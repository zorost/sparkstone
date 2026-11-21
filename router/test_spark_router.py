"""Tests for spark_router against in-process stub backends (no GPU, no network beyond localhost).

    python3 -m unittest discover -s router -v
"""
import http.client
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import spark_router as sr  # noqa: E402


class Stub(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def stub_backend(name: str, delay: float = 0.0):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health" and not self.server.down:
                data = b'{"ok":true}'
                self.send_response(200)
            else:
                data = b'{"down":true}'
                self.send_response(503)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            time.sleep(delay)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i in range(3):
                    ev = f"data: {json.dumps({'backend': name, 'i': i})}\n\n".encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            data = json.dumps({"backend": name, "echo": body.get("messages")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    s = Stub(("127.0.0.1", 0), H)
    s.down = False
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def chat(port, messages, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps({"model": "inkling-2-mini", "messages": messages, "stream": stream})
    c.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    return r.status, r.getheader("X-Spark-Backend"), data


class RouterTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b = stub_backend("a"), stub_backend("b")
        urls = [f"http://127.0.0.1:{s.server_address[1]}" for s in (self.a, self.b)]
        self.srv, self.stop = sr.serve(urls, "127.0.0.1", 0, spill_at=2, interval=0.1)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_address[1]

    def tearDown(self):
        self.stop.set()
        self.srv.shutdown()
        self.srv.server_close()
        for s in (self.a, self.b):
            s.shutdown()
            s.server_close()

    def test_affinity_is_sticky_across_turns(self):
        first = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
        _, be1, _ = chat(self.port, first)
        later = first + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
        for _ in range(5):
            _, be, _ = chat(self.port, later)
            self.assertEqual(be, be1)

    def test_conversations_spread_over_backends(self):
        seen = set()
        for i in range(40):
            _, be, _ = chat(self.port, [{"role": "user", "content": f"topic {i}"}])
            seen.add(be)
        self.assertEqual(len(seen), 2)

    def test_streaming_passthrough(self):
        status, _, data = chat(self.port, [{"role": "user", "content": "s"}], stream=True)
        self.assertEqual(status, 200)
        self.assertEqual(data.count(b"data: "), 3)

    def test_failover_when_backend_down(self):
        msgs = [{"role": "user", "content": "failover"}]
        _, be, _ = chat(self.port, msgs)
        victim = self.a if be.endswith(str(self.a.server_address[1])) else self.b
        victim.down = True
        victim.shutdown()
        victim.server_close()
        status, be2, _ = chat(self.port, msgs)
        self.assertEqual(status, 200)
        self.assertNotEqual(be2, be)

    def test_router_endpoints(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/router/stats")
        stats = json.loads(c.getresponse().read())
        self.assertEqual(len(stats["backends"]), 2)

    def test_affinity_key(self):
        k1 = sr.affinity_key("/v1/chat/completions", json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]}).encode())
        k2 = sr.affinity_key("/v1/chat/completions", json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
                                        {"role": "user", "content": "z"}]}).encode())
        self.assertEqual(k1, k2)
        self.assertIsNone(sr.affinity_key("/x", b"not json"))


class SpillTest(unittest.TestCase):
    def test_spill_to_less_loaded(self):
        pool = sr.Pool(["http://a:1", "http://b:1"], spill_at=2)
        key = "k"
        home = sr.rendezvous(key, pool.backends)[0]
        picks = [pool.choose(key).url for _ in range(4)]
        self.assertEqual(picks[:2], [home.url, home.url])
        self.assertNotEqual(picks[2], home.url)


if __name__ == "__main__":
    unittest.main()
