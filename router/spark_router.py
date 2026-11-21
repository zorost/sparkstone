#!/usr/bin/env python3
"""spark_router: one OpenAI/Anthropic-compatible endpoint in front of several Basalt servers (two DGX Sparks).

Data parallel: every Spark runs the whole model. The router sends each request to one of them:

* Conversation affinity. A request's affinity key is the model plus its first messages (system prompt and first
  user turn). Rendezvous hashing maps the key to a backend, so a conversation keeps landing on the Spark that
  holds its KV prefix cache, and adding or losing a backend moves only the keys it owned.
* Load spill. When the affinity backend already has `--spill-at` requests in flight and another healthy backend
  has fewer, the request goes to the least-loaded one instead (a cache miss beats a queue).
* Health. Each backend's /health is polled; a backend that fails a poll or a proxied connection is taken out
  until a poll succeeds again. A request whose backend refuses the connection before any byte was sent back is
  retried once on another backend.
* Streaming. Responses (SSE included) are copied through chunk by chunk; nothing is buffered.

Standard library only (Python 3.10+), like Basalt's own server.

    python3 router/spark_router.py --backend http://spark-a:8080 --backend http://spark-b:8080 --port 8000

Local endpoints: GET /router/health, GET /router/stats. Everything else is proxied.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import http.server
import json
import logging
import socketserver
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field

log = logging.getLogger("spark_router")

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


@dataclass
class Backend:
    url: str
    healthy: bool = True
    in_flight: int = 0
    served: int = 0
    failures: int = 0
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def parts(self) -> urllib.parse.SplitResult:
        return urllib.parse.urlsplit(self.url)

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        p = self.parts
        cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        return cls(p.hostname, p.port or (443 if p.scheme == "https" else 80), timeout=timeout)

    def snapshot(self) -> dict:
        return {"url": self.url, "healthy": self.healthy, "in_flight": self.in_flight, "served": self.served,
                "failures": self.failures, "last_error": self.last_error}


def affinity_key(path: str, body: bytes) -> str | None:
    """The conversation identity of a chat request: model + system prompt + first user turn.
    None for requests that carry no conversation (they go to the least-loaded backend)."""
    if not body:
        return None
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    parts: list = [req.get("model", "")]
    if isinstance(req.get("system"), (str, list)):          # Anthropic Messages API
        parts.append(req["system"])
    msgs = req.get("messages")
    if isinstance(msgs, list) and msgs:
        firsts = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            firsts.append([m.get("role"), m.get("content")])
            if m.get("role") == "user":
                break
        parts.append(firsts)
    elif isinstance(req.get("prompt"), str):                # completions API: the prompt's head
        parts.append(req["prompt"][:2048])
    else:
        return None
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def rendezvous(key: str, backends: list[Backend]) -> list[Backend]:
    """Backends ordered by highest-random-weight for this key."""
    return sorted(backends, key=lambda b: hashlib.sha256((key + "|" + b.url).encode()).digest(), reverse=True)


class Pool:
    def __init__(self, urls: list[str], spill_at: int):
        self.backends = [Backend(u.rstrip("/")) for u in urls]
        self.spill_at = spill_at
        self.lock = threading.Lock()

    def healthy(self) -> list[Backend]:
        return [b for b in self.backends if b.healthy]

    def choose(self, key: str | None, exclude: set[str] = frozenset()) -> Backend | None:
        with self.lock:
            live = [b for b in self.healthy() if b.url not in exclude]
            if not live:
                return None
            least = min(live, key=lambda b: (b.in_flight, b.served))
            if key is None:
                pick = least
            else:
                pick = rendezvous(key, live)[0]
                if pick.in_flight >= self.spill_at and least.in_flight < pick.in_flight:
                    pick = least
            pick.in_flight += 1
            return pick

    def release(self, b: Backend, ok: bool, err: str = "") -> None:
        with self.lock:
            b.in_flight -= 1
            if ok:
                b.served += 1
            else:
                b.failures += 1
                b.last_error = err
                b.healthy = False


def health_loop(pool: Pool, interval: float, timeout: float, stop: threading.Event) -> None:
    while not stop.is_set():
        for b in pool.backends:
            ok, err = False, ""
            try:
                c = b.connection(timeout)
                c.request("GET", (b.parts.path or "") + "/health")
                r = c.getresponse()
                r.read()
                ok = 200 <= r.status < 300
                err = "" if ok else f"/health {r.status}"
                c.close()
            except OSError as e:
                err = f"/health {e.__class__.__name__}: {e}"
            if ok != b.healthy:
                log.warning("backend %s is now %s %s", b.url, "healthy" if ok else "unhealthy", err)
            b.healthy = ok
            if err:
                b.last_error = err
        stop.wait(interval)


def make_handler(pool: Pool, timeout: float):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "spark_router/0.1"

        def log_message(self, fmt, *args):  # route through logging
            log.info("%s %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, obj) -> None:
            data = json.dumps(obj, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _proxy(self) -> None:
            if self.path == "/router/health":
                n = len(pool.healthy())
                return self._send_json(200 if n else 503, {"healthy_backends": n, "backends": len(pool.backends)})
            if self.path == "/router/stats":
                return self._send_json(200, {"backends": [b.snapshot() for b in pool.backends],
                                             "spill_at": pool.spill_at})
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            key = affinity_key(self.path, body) if self.command == "POST" else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            tried: set[str] = set()
            for attempt in range(2):
                b = pool.choose(key, tried)
                if b is None:
                    return self._send_json(503, {"error": {"message": "no healthy backend", "type": "unavailable"}})
                tried.add(b.url)
                started = False
                try:
                    c = b.connection(timeout)
                    c.request(self.command, (b.parts.path or "") + self.path, body=body or None, headers=headers)
                    r = c.getresponse()
                    started = True
                    self.send_response(r.status, r.reason)
                    for k, v in r.getheaders():
                        if k.lower() not in HOP_BY_HOP:
                            self.send_header(k, v)
                    self.send_header("X-Spark-Backend", b.url)
                    stream_len = r.getheader("Content-Length")
                    if stream_len is not None:
                        self.send_header("Content-Length", stream_len)
                        self.end_headers()
                        self.wfile.write(r.read())
                    else:                                   # SSE / chunked: re-chunk as it arrives
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        while True:
                            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                    c.close()
                    pool.release(b, True)
                    return
                except (ConnectionRefusedError, ConnectionResetError, TimeoutError, http.client.HTTPException,
                        OSError) as e:
                    pool.release(b, False, f"{e.__class__.__name__}: {e}")
                    log.warning("backend %s failed: %s", b.url, e)
                    if started:                             # bytes already went to the client: cannot retry
                        self.close_connection = True
                        return
            self._send_json(502, {"error": {"message": "backends failed", "type": "bad_gateway"}})

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = _proxy

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(backends: list[str], host: str, port: int, spill_at: int = 2, interval: float = 2.0,
          timeout: float = 600.0) -> tuple[Server, threading.Event]:
    pool = Pool(backends, spill_at)
    stop = threading.Event()
    threading.Thread(target=health_loop, args=(pool, interval, 5.0, stop), daemon=True).start()
    srv = Server((host, port), make_handler(pool, timeout))
    srv.pool = pool  # type: ignore[attr-defined]
    return srv, stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="append", required=True, help="a Basalt server URL; repeat per Spark")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--spill-at", type=int, default=2,
                    help="in-flight requests on the affinity backend before a request spills to a less-loaded one")
    ap.add_argument("--health-interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request backend timeout, seconds")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    srv, stop = serve(a.backend, a.host, a.port, a.spill_at, a.health_interval, a.timeout)
    print(f"spark_router on http://{a.host}:{a.port} -> {', '.join(a.backend)}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
