"""Untrusted-content handling: sanitisation, delimiting, injection heuristics.

Everything the user types and everything retrieved from the web is data. This
module makes that data safer to place inside a prompt; it does not make the
system immune to prompt injection. The real guarantees come from code-side
validation (evidence IDs, verbatim quotes, deterministic verdict rules) which an
injected instruction cannot change.
"""

from __future__ import annotations

import os
import re
import secrets
import unicodedata
from urllib.parse import urlsplit

from .config import GEMINI_KEY_ENV, TAVILY_KEY_ENV

# Control characters except tab/newline; bidi overrides; zero-width chars.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_BIDI = re.compile(r"[‪-‮⁦-⁩‎‏]")
_ZW = re.compile(r"[​-‍⁠﻿]")
_MD_LINK = re.compile(r"!?\[([^\]]{0,300})\]\((?:[^)\s]{0,2000})\)")
_MD_NOISE = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s+|^>\s+)", re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_BOUNDARY_LIKE = re.compile(r"<<<[^<>]{0,80}>>>")


def sanitize_text(text: str, max_chars: int | None = None) -> str:
    """Normalise untrusted text for storage and prompting.

    Removes control, bidi-override and zero-width characters, flattens Markdown
    links/emphasis and HTML tags to their text, strips anything that looks like
    our prompt delimiters, and collapses blank-line runs. The result is the
    canonical text stored for evidence, so quotes are validated against exactly
    what the judge saw.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    text = _BIDI.sub("", text)
    text = _ZW.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MD_NOISE.sub("", text)
    text = _BOUNDARY_LIKE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def sanitize_user_input(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    text = _BIDI.sub("", text)
    text = _ZW.sub("", text)
    text = _BOUNDARY_LIKE.sub(" ", text)
    return text.strip()


_IGNORE_INSTRUCTIONS = (
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|system|the)\b"
    r"[^.\n]{0,30}\b(instructions?|prompts?|rules?|directions?)\b"
)
_FORCE_LABEL = (
    r"\b(mark|label|rate|classify|tag)\s+(this|that|every|each|all|the)\s+(claims?|statements?|text|paragraph)\b"
    r"[^.\n]{0,15}\b(as|to be)\b"
)
_ROLE_TOKENS = r"<\|?(im_start|im_end|system|assistant)\|?>|(^|\n)\s*(assistant|system)\s*:"
_INJECTION_PATTERNS = [
    _IGNORE_INSTRUCTIONS,
    _FORCE_LABEL,
    _ROLE_TOKENS,
    r"\b(system|developer)\s+(prompt|instructions?)\b",
    r"\byou are now (in|an?|the)\b",
    r"\bnew (instructions?|rules)\s*:",
    r"\b(act|behave|respond) as (an?|the) (assistant|ai|system|admin)",
    r"\b(mark|label|rate|classify|judge)\b[^.\n]{0,40}\b(as|to be)\b[^.\n]{0,15}\b(supported|refuted)\b",
    r"\bverdict\s*[:=]\s*\"?(supported|refuted|true|false)",
    r"\b(you|assistants?|ai models?|language models?|llms?|fact[- ]?checkers?|chatbots?)\s+(must|should|shall)\s+"
    r"(now\s+)?(rate|mark|label|classify|say|output|answer|respond|ignore|reveal|print)\b",
    r"\b(reveal|print|output|show|leak)\b[^.\n]{0,30}\b(api[_ ]?keys?|secrets?|passwords?|environment variables?|system prompt)",
    r"\b(run|execute)\b[^.\n]{0,20}\b(shell command|the following command|this command|this script)\b",
    r"\bevidence_id\b|\bproposed_verdict\b",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)
# Narrower check for extracted *claims*: only unmistakable instructions to the
# system, so that factual statements about AI or software are never dropped.
_INSTRUCTION = re.compile(
    "|".join(f"(?:{p})" for p in (_IGNORE_INSTRUCTIONS, _FORCE_LABEL, _ROLE_TOKENS, r"\byou are now (in|an?|the)\b")),
    re.IGNORECASE,
)


def is_instruction(text: str) -> bool:
    """True if text is an instruction aimed at the fact-checker rather than a claim."""
    return bool(_INSTRUCTION.search(text))


def detect_injection(text: str) -> bool:
    """Heuristic: does this text contain instruction-like content aimed at a model?

    False positives are tolerated (the evidence is down-weighted, not dropped);
    false negatives are expected for novel attacks, which is why verdicts never
    depend on this check alone.
    """
    return bool(_INJECTION.search(text))


def make_boundary() -> str:
    """A random per-run token used to delimit untrusted content in prompts."""
    return secrets.token_hex(8).upper()


def wrap_untrusted(label: str, text: str, boundary: str) -> str:
    safe_label = re.sub(r"[^A-Z0-9_]", "", label.upper()) or "DATA"
    body = text.replace(boundary, "")
    return f"<<<{safe_label}_{boundary}>>>\n{body}\n<<<END_{safe_label}_{boundary}>>>"


def is_safe_url(url: str) -> bool:
    """Only absolute http(s) URLs with a hostname are accepted as sources."""
    if not isinstance(url, str) or len(url) > 2048:
        return False
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.hostname)


def redact_secrets(message: str) -> str:
    """Remove configured API key values from a string before it is shown/logged."""
    out = str(message)
    for env in (GEMINI_KEY_ENV, TAVILY_KEY_ENV, "GOOGLE_API_KEY"):
        value = os.environ.get(env, "")
        if len(value) >= 8:
            out = out.replace(value, "[REDACTED]")
    out = re.sub(r"(key=)[A-Za-z0-9_\-]{12,}", r"\1[REDACTED]", out)
    out = re.sub(r"\btvly-[A-Za-z0-9_\-]{8,}", "[REDACTED]", out)
    out = re.sub(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}", r"\1 [REDACTED]", out)
    out = re.sub(r"\bAIza[0-9A-Za-z_\-]{20,}", "[REDACTED]", out)
    return out[:500]
