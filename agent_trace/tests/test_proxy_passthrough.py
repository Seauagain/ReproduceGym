"""End-to-end: proxy forwards verbatim, tees the stream, and captures a record."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_trace.proxy.capture_writer import CaptureWriter
from agent_trace.proxy.server import ProxyConfig, serve

_SSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_x","role":"assistant","model":"opus","usage":{"input_tokens":3}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"pong"}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)


class _FakeUpstream(BaseHTTPRequestHandler):
    received_beta: str | None = None

    def log_message(self, *a):  # noqa: A003
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).received_beta = self.headers.get("anthropic-beta")
        body = _SSE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def fake_upstream():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def test_proxy_tees_and_captures(fake_upstream, tmp_path: Path):
    up_port = fake_upstream.server_address[1]
    writer = CaptureWriter(tmp_path)
    cfg = ProxyConfig(
        upstream_base=f"http://127.0.0.1:{up_port}",
        writer=writer,
        default_session="t1",
    )
    proxy = serve(cfg, host="127.0.0.1", port=0)
    try:
        proxy_port = proxy.server_address[1]
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            data=json.dumps({"model": "Vendor2/Claude-4.6-opus", "stream": True,
                             "messages": [{"role": "user", "content": "ping"}]}).encode(),
            headers={
                "Content-Type": "application/json",
                "anthropic-beta": "context-1m-2025-08-07",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            streamed = resp.read().decode("utf-8")
    finally:
        proxy.shutdown()

    # Verbatim tee: client got the exact SSE bytes.
    assert streamed == _SSE
    # Beta header forwarded untouched (1M passthrough).
    assert _FakeUpstream.received_beta == "context-1m-2025-08-07"

    # A record + raw sidecar were persisted.
    comp_dir = tmp_path / "sessions" / "t1" / "completions"
    jsons = sorted(comp_dir.glob("*.json"))
    raws = sorted(comp_dir.glob("*.raw"))
    assert len(jsons) == 1 and len(raws) == 1

    record = json.loads(jsons[0].read_text())
    assert record["api_type"] == "anthropic"
    assert record["metadata"]["anthropic_beta"] == "context-1m-2025-08-07"
    assert record["metadata"]["model_requested"] == "Vendor2/Claude-4.6-opus"
    assert record["original_request"]["messages"][0]["content"] == "ping"
    # Response reconstructed from the streamed SSE.
    assert record["response"]["content"] == [{"type": "text", "text": "pong"}]
    assert record["response"]["stop_reason"] == "end_turn"
    # Raw sidecar is the literal wire stream.
    assert raws[0].read_text() == _SSE
