"""Command-line interface.

Exit codes:
  0  completed; every checkable claim resolved (or nothing to check)
  1  completed with partial results or unresolved claims
  2  invalid input or configuration (including missing credentials)
  3  run-level failure (authentication, extraction failure, step limit, unexpected error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import envfile
from .config import Config, ConfigError, credential_status
from .graph import RunFailure, verify_text
from .models import UNRESOLVED_VERDICTS, Report
from .report import render_json, render_text
from .security import redact_secrets
from .validation import InputError

EXIT_OK, EXIT_PARTIAL, EXIT_INVALID, EXIT_FAILURE = 0, 1, 2, 3
MAX_FILE_BYTES = 64 * 1024


def exit_code_for(report: Report) -> int:
    if report.over_limit_claims or any(r.verdict in UNRESOLVED_VERDICTS for r in report.claims):
        return EXIT_PARTIAL
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="factcheck",
        description="Extract checkable claims from text, retrieve web evidence (Tavily) and assess it (Gemini).",
        epilog="Exit codes: 0 resolved, 1 partial/unresolved, 2 invalid input/config, 3 run failure.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("text", nargs="?", help="text to check (use '-' to read from standard input)")
    src.add_argument("-f", "--file", type=Path, help="read the text from a UTF-8 file")
    p.add_argument("--json-out", type=Path, help="also write the full report as JSON to this path")
    p.add_argument("--model", help="Gemini model ID (default from FACTCHECK_MODEL or built-in default)")
    p.add_argument("--max-claims", type=int, help="maximum checkable claims to verify (1-12, default 6)")
    p.add_argument("--search-budget", type=int, help="maximum Tavily credits for this run (0-40, default 40)")
    p.add_argument("--timeout", type=float, help="run time limit in seconds (default 180)")
    p.add_argument("--check-config", action="store_true", help="show configuration and which credentials are set, then exit")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the summary line")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="also show evidence IDs, every citation, score details and diagnostics")
    return p


def _read_input(args: argparse.Namespace) -> str:
    if args.file is not None:
        path: Path = args.file
        if not path.is_file():
            raise InputError(f"input file not found: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise InputError(f"input file is larger than {MAX_FILE_BYTES // 1024} KB")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise InputError("input file is not valid UTF-8") from None
    if args.text == "-":
        return sys.stdin.read(MAX_FILE_BYTES + 1)[: MAX_FILE_BYTES + 1]
    if args.text is None:
        raise InputError("no input: pass TEXT, '-' for stdin, or --file PATH")
    return args.text


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = _parser().parse_args(argv)
    for warning in envfile.load_env_file():
        print(f"warning: {warning}", file=sys.stderr)
    try:
        config = Config.from_env(
            model=args.model,
            max_checkable_claims=args.max_claims,
            search_credit_budget=args.search_budget,
            run_timeout_s=args.timeout,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if args.check_config:
        print("Configuration:")
        for key, value in config.model_dump(mode="json").items():
            print(f"  {key}: {value}")
        print("Credentials (presence only):")
        for name, present in credential_status().items():
            source = " (from .env)" if name in envfile.loaded_from_file else ""
            print(f"  {name}: {'set' + source if present else 'NOT SET'}")
        return EXIT_OK

    try:
        text = _read_input(args)
        if args.json_out is not None and args.json_out.exists() and args.json_out.is_dir():
            raise InputError(f"--json-out points to a directory: {args.json_out}")
        report = verify_text(text, config)
    except (InputError, ConfigError) as exc:
        print(f"error: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_INVALID
    except RunFailure as exc:
        print(f"run failed: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001
        print(f"run failed: unexpected {type(exc).__name__}: {redact_secrets(str(exc))}", file=sys.stderr)
        return EXIT_FAILURE

    if args.quiet:
        print(f"{report.summary_label.value}: {report.summary_text}")
    else:
        print(render_text(report, verbose=args.verbose))
    if args.json_out is not None:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(render_json(report), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write JSON report: {exc}", file=sys.stderr)
            return EXIT_FAILURE
    return exit_code_for(report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
