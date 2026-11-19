"""``python -m factcheck_api [--host 127.0.0.1] [--port 8000] [--reload]``: start the API with Uvicorn."""

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m factcheck_api", description="Start the FactCheck REST API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = p.parse_args()
    uvicorn.run("factcheck_api.app:app", host=args.host, port=args.port, reload=args.reload)


main()
