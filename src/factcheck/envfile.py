"""Minimal ``.env`` loader (no extra dependency).

Used only by the command-line entry points (``factcheck``, the diagnostics, the
live evaluation); the library function ``verify_text`` never reads files.

Rules:
* ``KEY=VALUE`` lines; blank lines and ``#`` comments are ignored; ``export KEY=...``
  is accepted; a value may be wrapped in single or double quotes; an unquoted
  value may end with `` # comment``. UTF-8 with or without BOM, LF or CRLF.
* A real environment variable wins over ``.env`` - unless it is unusable (blank,
  or containing non-printable characters, e.g. a broken clipboard paste).
* Empty values (``GEMINI_API_KEY=``) are placeholders and are skipped.
* Values with non-printable characters are rejected with a warning that names
  the variable and line - values are never printed.
* ``FACTCHECK_NO_DOTENV=1`` disables loading (the test suite sets it);
  ``FACTCHECK_ENV_FILE`` points at a different file (default: ``./.env``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DISABLE_ENV = "FACTCHECK_NO_DOTENV"
PATH_ENV = "FACTCHECK_ENV_FILE"
_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

# Names whose value came from the file during this process (for --check-config).
loaded_from_file: set[str] = set()


def _usable(value: str | None) -> bool:
    return bool(value and value.strip()) and value.isprintable()


def parse_env(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse .env text into (values, problems). Problems never contain values."""
    values: dict[str, str] = {}
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            problems.append(f"line {number}: not in KEY=VALUE form (ignored)")
            continue
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if not value.isprintable():
            problems.append(f"line {number}: {name} contains non-printable characters (ignored)")
            continue
        values[name] = value
    return values, problems


def load_env_file(path: str | os.PathLike | None = None) -> list[str]:
    """Load ``.env`` into ``os.environ``. Returns warnings (without values)."""
    if os.environ.get(DISABLE_ENV) == "1":
        return []
    target = Path(path or os.environ.get(PATH_ENV) or Path.cwd() / ".env")
    if not target.is_file():
        return []
    try:
        text = target.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"could not read {target.name}: {type(exc).__name__}"]
    values, problems = parse_env(text)
    for name, value in values.items():
        if not value:
            continue
        if _usable(os.environ.get(name)):
            continue  # a real, usable environment variable takes precedence
        os.environ[name] = value
        loaded_from_file.add(name)
    return [f"{target.name} {p}" for p in problems]
