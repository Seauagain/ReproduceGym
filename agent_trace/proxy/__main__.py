"""CLI: launch the passthrough capture proxy.

Example::

    python -m agent_trace.proxy \
        --upstream https://api.gpugeek.com \
        --save-dir ./captures \
        --session run-001 \
        --port 8788

Then point the agent at it::

    ANTHROPIC_BASE_URL=http://127.0.0.1:8788 claude ...
"""

from __future__ import annotations

import argparse
import signal
import sys

from agent_trace.proxy.capture_writer import CaptureWriter
from agent_trace.proxy.server import ProxyConfig, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_trace.proxy")
    parser.add_argument("--upstream", required=True, help="upstream base URL")
    parser.add_argument("--save-dir", required=True, help="capture output dir")
    parser.add_argument("--session", default="capture", help="session id for this run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--read-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    cfg = ProxyConfig(
        upstream_base=args.upstream,
        writer=CaptureWriter(args.save_dir),
        default_session=args.session,
        read_timeout=args.read_timeout,
    )
    httpd = serve(cfg, host=args.host, port=args.port)
    print(
        f"[agent_trace.proxy] listening on http://{args.host}:{args.port} "
        f"-> {args.upstream}  (session={args.session}, save_dir={args.save_dir})",
        flush=True,
    )

    try:
        if hasattr(signal, "pause"):
            signal.pause()
        else:
            import time

            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        print("[agent_trace.proxy] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
