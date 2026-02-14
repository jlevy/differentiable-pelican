from __future__ import annotations

import argparse
import sys
import webbrowser


def serve_command() -> None:
    """Launch the web UI server."""
    parser = argparse.ArgumentParser(
        prog="pelican serve",
        description="Start the Pelican web UI server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open the browser automatically"
    )

    args = parser.parse_args(sys.argv[2:])

    try:
        import uvicorn  # pyright: ignore[reportMissingImports]
    except ImportError:
        print(
            "Error: Web dependencies not installed.\n"
            'Install with: pip install "differentiable-pelican[web]"\n'
            '         or: uv sync --extra web',
            file=sys.stderr,
        )
        sys.exit(1)

    from differentiable_pelican.web.server import app

    url = f"http://{args.host}:{args.port}"
    print(f"Starting Pelican web UI at {url}")

    if not args.no_open:
        # Open browser after a short delay so server has time to start
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
