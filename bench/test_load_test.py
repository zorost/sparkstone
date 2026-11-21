"""load_test.py against an in-process fake OpenAI server, directly and through the Twin router.

    python3 -m unittest discover -s bench -v
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest

here = os.path.dirname(__file__)
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", "router"))
import load_test as lt  # noqa: E402
import spark_router as sr  # noqa: E402

TOKENS = 20


class Fake(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def fake_openai(send_usage=True):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = min(TOKENS, body.get("max_tokens", TOKENS))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def ev(obj):
                b = f"data: {json.dumps(obj)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                self.wfile.flush()
            for i in range(n):
                time.sleep(0.005)
                ev({"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]})
            last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
            if send_usage:
                last["usage"] = {"prompt_tokens": 30, "completion_tokens": n, "total_tokens": 30 + n}
            ev(last)
            b = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.write(b"0\r\n\r\n")

    s = Fake(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


class LoadTest(unittest.TestCase):
    def test_direct_with_usage(self):
        s = fake_openai()
        try:
            url = f"http://127.0.0.1:{s.server_address[1]}"
            r = lt.one_request(url, {}, "m", "struct", 0, 64, "none", 10)
            self.assertIsNone(r["error"])
            self.assertEqual(r["tokens"], TOKENS)
            self.assertEqual(r["tokens_from"], "usage")
            self.assertGreater(r["stream_tps"], 0)
            self.assertGreater(r["ttft_s"], 0)
        finally:
            s.shutdown()
            s.server_close()

    def test_chunk_count_fallback(self):
        s = fake_openai(send_usage=False)
        try:
            r = lt.one_request(f"http://127.0.0.1:{s.server_address[1]}", {}, "m", "chat", 1, 64, None, 10)
            self.assertEqual((r["tokens"], r["tokens_from"]), (TOKENS, "chunks"))
        finally:
            s.shutdown()
            s.server_close()

    def test_levels_through_router(self):
        a, b = fake_openai(), fake_openai()
        srv, stop = sr.serve([f"http://127.0.0.1:{x.server_address[1]}" for x in (a, b)], "127.0.0.1", 0,
                             interval=0.1)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "lt_test.json")
            rc = lt.main(["--url", f"http://127.0.0.1:{srv.server_address[1]}", "--workload", "chat",
                          "--concurrency", "1,4", "--requests-per-level", "8", "--max-tokens", "64", "--out", out])
            self.assertEqual(rc, 0)
            with open(out) as fh:
                res = json.load(fh)
            four = res["levels"][1]
            self.assertEqual(four["errors"], 0)
            self.assertEqual(four["completion_tokens"], 8 * TOKENS)
            self.assertEqual(len(four["backends"]), 2)        # distinct conversations spread over both "Sparks"
            os.remove(out)
        finally:
            stop.set()
            srv.shutdown()
            srv.server_close()
            for x in (a, b):
                x.shutdown()
                x.server_close()


if __name__ == "__main__":
    unittest.main()
