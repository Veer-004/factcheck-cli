"""Conservative, deterministic text comparison for de-duplication.

Two pieces of text are only treated as the same when nothing that can change
meaning differs. The "critical signature" collects exactly those details:

* numbers (as values, so "1,000" == "1000"), and the digits glued to letters;
* negation ("not", "never", "isn't", "no", "without" ...);
* comparison / quantity / direction words ("more", "less", "before", "after",
  "only", "all", "increase" ...), which the project's stopword list ignores;
* units ("km" vs "miles", "celsius" vs "fahrenheit", "%" ...) and month names.

Two claim statements are equivalent only when they are identical after removing
case, punctuation and spacing, or when they have the same content words in the
same order, the same critical signature, the same modal/tense/logic words and
the same capitalised words (names). Shared vocabulary alone never merges two
statements; synonyms and paraphrases are deliberately kept apart (checked twice).
"""

from __future__ import annotations

import re

from factcheck.textnorm import STOPWORDS, extract_numbers, normalize_for_match, tokens

NEGATIONS = frozenset("not no never none nobody nothing nowhere neither nor without cannot".split())
RELATIONAL = frozenset(
    """more less fewer most least before after over under above below higher lower larger smaller bigger greater
    longer shorter older younger earlier later first last only all every some most increase increased increases
    decrease decreased decreases rise rose risen fall fell fallen up down double doubled half halved""".split())
UNITS = {
    **dict.fromkeys("km kilometre kilometres kilometer kilometers".split(), "km"),
    **dict.fromkeys("m metre metres meter meters".split(), "m"),
    **dict.fromkeys("cm centimetre centimetres centimeter centimeters".split(), "cm"),
    **dict.fromkeys("mm millimetre millimetres millimeter millimeters".split(), "mm"),
    **dict.fromkeys("mi mile miles".split(), "mi"),
    **dict.fromkeys("ft foot feet".split(), "ft"),
    **dict.fromkeys("inch inches".split(), "in"),  # bare "in" is almost always the preposition
    **dict.fromkeys("kg kilogram kilograms kilo kilos".split(), "kg"),
    **dict.fromkeys("g gram grams".split(), "g"),
    **dict.fromkeys("lb lbs pound pounds".split(), "lb"),
    **dict.fromkeys("t tonne tonnes ton tons".split(), "t"),
    **dict.fromkeys("c celsius centigrade".split(), "c"),
    **dict.fromkeys("f fahrenheit".split(), "f"),
    **dict.fromkeys("k kelvin".split(), "k"),
    **dict.fromkeys("percent per cent %".split(), "%"),
    **dict.fromkeys("l litre litres liter liters".split(), "l"),
    **dict.fromkeys("s sec second seconds".split(), "s"),
    **dict.fromkeys("min minute minutes".split(), "min"),
    **dict.fromkeys("h hr hour hours".split(), "h"),
    **dict.fromkeys("day days".split(), "day"),
    **dict.fromkeys("year years yr yrs".split(), "year"),
    **dict.fromkeys("million millions mn".split(), "million"),
    **dict.fromkeys("billion billions bn".split(), "billion"),
    **dict.fromkeys("trillion trillions".split(), "trillion"),
    **dict.fromkeys("thousand thousands".split(), "thousand"),
    **dict.fromkeys("usd dollar dollars $".split(), "usd"),
    **dict.fromkeys("eur euro euros €".split(), "eur"),
    **dict.fromkeys("gbp £".split(), "gbp"),
    **dict.fromkeys("inr rupee rupees ₹".split(), "inr"),
}
MONTHS = {m[:3]: m[:3] for m in "january february march april may june july august september october november december".split()}
MONTHS.update({m: m[:3] for m in "january february march april june july august september october november december".split()})
MONTHS.update({"sept": "sep"})

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?|[%$€£₹°]")
_ALNUM_MIX = re.compile(r"\b(?=[a-z]*\d)(?=\d*[a-z])[a-z\d]+\b")


def words(text: str) -> list[str]:
    return _WORD.findall(normalize_for_match(text))


def loose_key(text: str) -> str:
    """Case, punctuation and spacing removed; every word and digit kept in order."""
    return " ".join(w for w in words(text) if w != "°")


def critical_signature(text: str) -> tuple:
    ws = words(text)
    negs = sorted(w for w in ws if w in NEGATIONS or w.endswith("n't"))
    rel = sorted({w for w in ws if w in RELATIONAL})
    units = sorted({UNITS[w] for w in ws if w in UNITS})
    months = sorted({MONTHS[w] for w in ws if w in MONTHS and w != "may"})  # "may" is usually the verb
    nums = sorted(round(n, 6) for n in extract_numbers(text))
    codes = sorted(set(_ALNUM_MIX.findall(normalize_for_match(text))))  # "covid19", "a380", "5g"
    # one/two-letter words are not content tokens, but "Plan A"/"Plan B" or "Vitamin C"/"D" differ
    short = sorted({w for w in ws if len(w) <= 2 and w.isalpha() and w not in STOPWORDS and w not in UNITS})
    return (tuple(nums), tuple(codes), tuple(negs), tuple(rel), tuple(units), tuple(months), tuple(short))


def _proper_words(text: str) -> set[str]:
    """Capitalised words, as content tokens (same stemming as ``content_tokens``).

    The first word of a sentence is included too: it is often a name ("Biden signed ...").
    That also blocks merges such as "Scientists found" vs "Researchers found", which is the
    conservative direction. All-caps text (common in OCR) carries no capitalisation signal.
    """
    found: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        ws = re.findall(r"[^\W\d_][\w'-]*", sentence)
        if sum(t.isupper() and len(t) > 1 for t in ws) > len(ws) / 2:
            continue
        found += [t for t in ws if t[0].isupper()]
    return set(tokens(" ".join(found)))


# Stopwords that still change meaning between two otherwise identical statements
# ("may cause" vs "will cause", "is" vs "was", "and" vs "or").
MODAL_TENSE_LOGIC = frozenset(
    """may might must can could will would should shall is are was were be been being has have had do does did
    and or if than each other same both either""".split())


def content_sequence(text: str) -> list[str]:
    """Content words in order (the project's stopword removal and light stemming).

    Figures are left out here ("8,849" tokenises as "8", "849"): they are compared by
    value, and letter-digit codes verbatim, in the signature. Unit words are left out
    too: the signature compares them in canonical form ("percent" == "%", "km" ==
    "kilometres").
    """
    return [t for t in tokens(text) if len(t) > 1 and not any(c.isdigit() for c in t) and t not in UNITS]


def statement_signature(text: str) -> tuple:
    ws = words(text)
    return critical_signature(text) + (tuple(sorted(w for w in ws if w in MODAL_TENSE_LOGIC)),)


def statements_equivalent(a: str, b: str) -> bool:
    """Do two claim statements say the same thing? False whenever unsure.

    Equivalent = identical after removing case/punctuation/spacing, or the same content
    words in the same order with the same critical details (numbers, negation, units,
    dates, comparison/quantity words, modal/tense/logic words) and the same names.
    Differences only in articles, prepositions, number formatting or plural endings are
    tolerated; synonyms, paraphrases and reordering are not (both are then checked).
    """
    if loose_key(a) == loose_key(b):
        return True
    if content_sequence(a) != content_sequence(b):
        return False
    if statement_signature(a) != statement_signature(b):
        return False
    return _proper_words(a) == _proper_words(b)


def claims_duplicate(earlier: tuple[str, str], new: tuple[str, str]) -> bool:
    """``validation.DuplicateCheck`` for the video workflow: (statement, excerpt) pairs.

    Both the model-written statements *and* the verbatim source excerpts must
    agree on every critical detail. The excerpt check stops a merge when the
    model wrote the same statement for two source passages that actually differ
    (e.g. spoken "1889" and on-screen "1887").
    """
    if not statements_equivalent(earlier[0], new[0]):
        return False
    return critical_signature(earlier[1]) == critical_signature(new[1])
