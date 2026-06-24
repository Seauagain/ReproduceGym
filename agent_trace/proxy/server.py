"""Stdlib passthrough HTTP proxy with response tee + native capture.

Forwards every request verbatim to a configured upstream model API. For a
captured path (default: ``*/v1/messages``) it persists a ``CompletionRecord``:
the native request body plus the response (reconstructed from SSE when streamed),
and the verbatim upstream bytes as a ``.raw`` sidecar.

No third-party dependencies: ``http.server`` for the listener, ``http.client``
for the upstream call. This runs anywhere CPython does, including stripped-down
boxes where FastAPI/httpx are not installed.
"""

from __future__ import annotations

import gzip
import http.client
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from agent_trace.proxy.anthropic_sse import reconstruct_from_raw
from agent_trace.proxy.capture_writer import CaptureWriter

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",
    }
)
@dataclass
class ProxyConfig:
    upstream_base: str
    writer: CaptureWriter
    default_session: str = "capture"
    capture_suffixes: tuple[str, ...] = ("/v1/messages",)
    connect_timeout: float = 30.0
    read_timeout: float = 900.0

    # Parsed upstream parts (filled in __post_init__).
    scheme: str = field(default="https", init=False)
    host: str = field(default="", init=False)
    port: int = field(default=443, init=False)
    base_path: str = field(default="", init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.upstream_base)
        self.scheme = parts.scheme or "https"
        self.host = parts.hostname or ""
        self.port = parts.port or (443 if self.scheme == "https" else 80)
        self.base_path = parts.path.rstrip("/")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _should_capture(cfg: ProxyConfig, method: str, path: str) -> bool:
    if method != "POST":
        return False
    path_only = path.split("?", 1)[0]
    return any(path_only.endswith(s) for s in cfg.capture_suffixes)


def make_handler(cfg: ProxyConfig) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Keep the proxy quiet; capture is the signal, not access logs.
            pass

        # All verbs route through one handler.
        def do_GET(self) -> None:  # noqa: N802
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._proxy("POST")

        def do_PUT(self) -> None:  # noqa: N802
            self._proxy("PUT")

        def do_DELETE(self) -> None:  # noqa: N802
            self._proxy("DELETE")

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _forward_headers(self) -> dict[str, str]:
            out: dict[str, str] = {}
            for key, val in self.headers.items():
                if key.lower() in _HOP_BY_HOP:
                    continue
                out[key] = val
            return out

        def _session_id(self) -> str:
            sid = self.headers.get("x-session-id")
            if sid:
                return sid
            qs = parse_qs(urlsplit(self.path).query)
            if qs.get("session_id"):
                return qs["session_id"][0]
            return cfg.default_session

        def _proxy(self, method: str) -> None:
            body = self._read_body()
            fwd_headers = self._forward_headers()
            upstream_path = cfg.base_path + urlsplit(self.path).path
            query = urlsplit(self.path).query
            if query:
                upstream_path = f"{upstream_path}?{query}"

            conn_cls = (
                http.client.HTTPSConnection
                if cfg.scheme == "https"
                else http.client.HTTPConnection
            )
            started = time.time()
            try:
                conn = conn_cls(cfg.host, cfg.port, timeout=cfg.read_timeout)
                conn.connect()
                conn.sock.settimeout(cfg.read_timeout)
                conn.request(method, upstream_path, body=body, headers=fwd_headers)
                resp = conn.getresponse()
            except Exception as exc:  # upstream unreachable / timeout
                self._fail(502, f"upstream error: {exc}")
                return

            ct = resp.getheader("Content-Type", "") or ""
            is_stream = "text/event-stream" in ct.lower()

            captured = bytearray()
            try:
                self.send_response(resp.status, resp.reason)
                for hk, hv in resp.getheaders():
                    if hk.lower() in _HOP_BY_HOP or hk.lower() == "content-length":
                        continue
                    self.send_header(hk, hv)

                if is_stream:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                    self.end_headers()
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        captured.extend(chunk)
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            # Client gave up; keep draining upstream for capture.
                            pass
                else:
                    payload = resp.read()
                    captured.extend(payload)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
            finally:
                conn.close()

            if _should_capture(cfg, method, self.path):
                try:
                    self._capture(
                        body=body,
                        upstream_bytes=bytes(captured),
                        is_stream=is_stream,
                        content_encoding=resp.getheader("Content-Encoding", ""),
                        status=resp.status,
                        latency_ms=int((time.time() - started) * 1000),
                    )
                except Exception:  # capture must never break the proxy
                    pass

        def _capture(
            self,
            *,
            body: bytes,
            upstream_bytes: bytes,
            is_stream: bool,
            content_encoding: str,
            status: int,
            latency_ms: int,
        ) -> None:
            decoded = upstream_bytes
            if content_encoding and "gzip" in content_encoding.lower():
                try:
                    decoded = gzip.decompress(upstream_bytes)
                except Exception:
                    decoded = upstream_bytes
            text = decoded.decode("utf-8", errors="replace")

            try:
                request_obj = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                request_obj = {}

            if is_stream:
                response_obj = reconstruct_from_raw(text)
            else:
                try:
                    response_obj = json.loads(text)
                except json.JSONDecodeError:
                    response_obj = {"_unparsed": text}

            completion_id = (
                str(response_obj.get("id"))
                if isinstance(response_obj, dict) and response_obj.get("id")
                else uuid.uuid4().hex
            )
            record = {
                "completion_id": completion_id,
                "timestamp": _now_iso(),
                "api_type": "anthropic",
                "original_request": request_obj,
                "request": request_obj,
                "response": response_obj,
                "metadata": {
                    "session_id": self._session_id(),
                    "model_requested": request_obj.get("model")
                    if isinstance(request_obj, dict)
                    else None,
                    "stream": bool(is_stream),
                    "status_code": status,
                    "latency_ms": latency_ms,
                    "anthropic_beta": self.headers.get("anthropic-beta"),
                    "anthropic_version": self.headers.get("anthropic-version"),
                    "user_agent": self.headers.get("user-agent"),
                },
            }
            cfg.writer.write(self._session_id(), record, raw_bytes=decoded)

        def _fail(self, code: int, message: str) -> None:
            payload = json.dumps({"error": {"message": message}}).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return _Handler


def serve(
    cfg: ProxyConfig,
    host: str = "127.0.0.1",
    port: int = 8788,
) -> ThreadingHTTPServer:
    """Start a threaded proxy server. Returns the server (call shutdown() to stop)."""
    cfg.writer.write_session_meta(
        cfg.default_session,
        {
            "created_at": _now_iso(),
            "upstream": f"{cfg.scheme}://{cfg.host}:{cfg.port}{cfg.base_path}",
        },
    )
    handler = make_handler(cfg)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
