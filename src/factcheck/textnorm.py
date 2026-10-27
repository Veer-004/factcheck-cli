"""Text normalisation, tokenisation and number extraction.

Documented matching normalisation (used for excerpt anchoring and quote
validation), applied identically to both sides of every comparison:

1. Unicode NFKC normalisation (folds full-width forms, ligatures, NBSP, "...").
2. Zero-width characters and soft hyphens are removed.
3. Typographic single quotes/primes -> ``'``; double quotes -> ``"``;
   hyphen/dash variants and the minus sign -> ``-``.
4. All whitespace runs collapse to one space; leading/trailing space removed.
5. Case-folded.

Nothing else is forgiven: words, digits and their order must match exactly.
"""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_CHAR_MAP = str.maketrans(
    {
        **dict.fromkeys("‘’‚‛′ʼ`", "'"),
        **dict.fromkeys("“”„‟″«»", '"'),
        **dict.fromkeys("‐‑‒–—―−", "-"),
    }
)
_WS = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH).translate(_CHAR_MAP)
    return _WS.sub(" ", text).strip().casefold()


def strip_quote_wrapping(quote: str) -> str:
    """Remove whitespace and wrapping quotation marks / ellipses the model may add."""
    q = quote.strip()
    for _ in range(3):
        q2 = q.strip(" \"'“”‘’").strip()
        q2 = re.sub(r"^(\.\.\.|…)\s*", "", q2)
        q2 = re.sub(r"\s*(\.\.\.|…)$", "", q2)
        if q2 == q:
            break
        q = q2
    return q


def original_span(haystack: str, needle: str) -> str | None:
    """The exact substring of ``haystack`` that ``needle`` matches after normalisation.

    Used so a validated quotation is displayed in the source's own characters
    (case, punctuation, spacing) rather than the model's copy of it. Normalisation
    is applied per character with an index map back into ``haystack``; returns
    None if no mapping is found (callers then keep the already-validated text).
    """
    target = normalize_for_match(needle)
    if not target:
        return None
    chars: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, ch in enumerate(haystack):
        piece = unicodedata.normalize("NFKC", ch).translate(_ZERO_WIDTH).translate(_CHAR_MAP)
        for p in piece:
            if p.isspace():
                pending_space = bool(chars)
                continue
            if pending_space:
                chars.append(" ")
                index.append(i)
                pending_space = False
            for f in p.casefold():
                chars.append(f)
                index.append(i)
    pos = "".join(chars).find(target)
    if pos < 0:
        return None
    return haystack[index[pos] : index[pos + len(target) - 1] + 1]


def contains_normalized(haystack: str, needle: str) -> bool:
    n = normalize_for_match(needle)
    return bool(n) and n in normalize_for_match(haystack)


STOPWORDS = frozenset(
    """a an the and or but if then than that this these those is are was were be been being
    of in on at to for from by with as into about over under after before between through during
    it its it's he she they them his her their we our you your i me my not no nor so too very
    can could will would should may might must do does did done has have had having
    there here what which who whom whose when where why how all any both each few more most other
    some such only own same just also up down out off again further once per via""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(normalize_for_match(text)):
        if t in STOPWORDS:
            continue
        if t.endswith("'s"):
            t = t[:-2]
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) > 1}


NUMBER_WORDS = frozenset(
    "thousand million billion trillion percent per cent bn mn hundred dozen".split()
)


def subject_tokens(text: str) -> set[str]:
    """Content tokens minus figures and scale words: what the text is *about*."""
    return {t for t in content_tokens(text) if not any(ch.isdigit() for ch in t) and t not in NUMBER_WORDS}


_NUMBER = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"(?:\s*(%|percent|per cent|thousand|million|billion|trillion|bn|mn)\b)?",
    re.IGNORECASE,
)
_SCALE = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
}


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values, applying scale words (``3.2 million`` -> 3.2e6).

    Percent signs are dropped (``45%`` -> 45.0). Ordinals and identifiers that
    are glued to letters (``COVID19``) are ignored by the boundary rules.
    """
    return [v for v, _ in number_mentions(text)]


def number_mentions(text: str, context_chars: int = 25) -> list[tuple[float, str]]:
    """Like :func:`extract_numbers` but also returns the text preceding each number."""
    return [(value, before) for value, before, _after, _suffix in number_contexts(text, context_chars)]


def number_contexts(text: str, context_chars: int = 25) -> list[tuple[float, str, str, str]]:
    """Numbers with (value, text before, text after, scale/percent suffix)."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    out: list[tuple[float, str, str, str]] = []
    for m in _NUMBER.finditer(text):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        value *= _SCALE.get(suffix, 1.0)
        out.append((value, text[max(0, m.start() - context_chars) : m.start()],
                    text[m.end() : m.end() + context_chars], suffix))
    return out


def decimals_of(value: float) -> int:
    s = repr(float(value))
    if "e" in s or "." not in s:
        return 0
    frac = s.split(".")[1].rstrip("0")
    return len(frac)


def numbers_compatible(a: float, b: float, approximate: bool = False) -> bool:
    """True when ``a`` and ``b`` agree within rounding (or approximation) tolerance.

    Each figure is treated as the interval implied by its written precision
    (``3.2 million`` covers 3.15-3.25 million; values below one million are read
    at their literal decimal precision, so ``1,000`` is not a rounded 1,450).
    The figures agree when the intervals overlap, or within a 0.5% relative
    slack; approximate claims ("about", "nearly", "roughly") tolerate 10%.
    """
    if a == b:
        return True
    small_ints = float(a).is_integer() and float(b).is_integer() and max(abs(a), abs(b)) < 10_000
    if small_ints:
        # Years and small counts: no relative slack ("1887" is not "1889").
        both_years = all(1000 <= abs(v) <= 2100 for v in (a, b))
        if approximate:
            return abs(a - b) <= (2 if both_years else max(1.0, 0.10 * max(abs(a), abs(b))))
        return False
    rel_tol = 0.10 if approximate else 0.005
    if abs(a - b) <= rel_tol * max(abs(a), abs(b)):
        return True

    def half_step(v: float) -> float:
        mag = abs(v)
        scale = 1e12 if mag >= 1e12 else 1e9 if mag >= 1e9 else 1e6 if mag >= 1e6 else 1.0
        return 0.5 * (10 ** (-decimals_of(v / scale))) * scale

    return abs(a - b) <= half_step(a) + half_step(b) - 1e-9
