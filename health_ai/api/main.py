"""
main.py — Entry point for Health AI v3.

Usage:
    python -m health_ai.api.main                  # default port 8000
    python -m health_ai.api.main --port 9000      # custom port
    python -m health_ai.api.main --host 0.0.0.0   # explicit host
    python -m health_ai.api.main --reload         # hot reload (dev only)
"""

import argparse
import sys
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Health AI v3 — Dr. Aria personal health assistant server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0 — accessible on LAN)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload on code changes (development only — slow with large models)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of uvicorn workers (default: 1). "
            "Keep at 1 unless you have multiple GPUs — "
            "the LLM singleton is not designed for multi-process sharing."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.workers > 1:
        print(
            "⚠️  Warning: Multiple workers with a GGUF model will load the LLM "
            f"{args.workers}× into RAM. Ensure you have enough memory."
        )

    uvicorn.run(
        "health_ai.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
