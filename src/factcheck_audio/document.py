"""The fact-check input built from a transcript, and tracing claim excerpts back to it.

The transcript is given to claim extraction as plain prose: segment texts joined
by spaces, with a paragraph break where the speaker paused for more than
``PARAGRAPH_GAP_S`` seconds. No timestamps or labels are put in the text, so
the model cannot copy them into a claim. Every claim's ``source_excerpt`` is then
located in the segments it was taken from (all occurrences, if the speaker
repeated it), and narrowed to the excerpt's own words when word timestamps exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from factcheck.textnorm import normalize_for_match, original_span

from .models import TranscriptSegment

PARAGRAPH_GAP_S = 2.0


@dataclass
class Occurrence:
    """One place the excerpt was spoken: the segments it spans and its time range."""

    segments: list[TranscriptSegment]
    start: float
    end: float
    word_level: bool = False


@dataclass
class TranscriptDocument:
    text: str
    segments: list[TranscriptSegment]  # included in ``text``, in time order
    omitted: list[TranscriptSegment] = field(default_factory=list)  # over the character limit

    def locate(self, excerpt: str) -> list[Occurrence]:
        """Every occurrence of the (normalised) excerpt; [] if it is not in the transcript."""
        target = normalize_for_match(excerpt)
        if not target:
            return []
        parts, spans, pos = [], [], 0
        for s in self.segments:
            norm = normalize_for_match(s.text)
            spans.append((pos, pos + len(norm)))
            parts.append(norm)
            pos += len(norm) + 1
        joined = " ".join(parts)
        found: list[Occurrence] = []
        start = joined.find(target)
        while start >= 0:
            end = start + len(target)
            hit = [s for s, (a, b) in zip(self.segments, spans) if a < end and b > start]
            if hit:
                occ = Occurrence(segments=hit, start=hit[0].start, end=hit[-1].end)
                narrowed = narrow_to_words(hit, excerpt)
                if narrowed:
                    occ.start, occ.end, occ.word_level = narrowed[0], narrowed[1], True
                if not any(o.segments == occ.segments and o.start == occ.start for o in found):
                    found.append(occ)
            start = joined.find(target, start + 1)
        return found


def narrow_to_words(segments: list[TranscriptSegment], excerpt: str) -> Optional[tuple[float, float]]:
    """Start/end of the excerpt's own words across ``segments``, when all have word timestamps."""
    if not segments or any(not s.words for s in segments):
        return None
    joined, offsets = "", []
    for w in (w for s in segments for w in s.words):
        piece = w.text.strip()
        if not piece:
            continue
        # Whisper marks a new word with a leading space; a token without one continues the previous
        # word (e.g. " 50" + "%" -> "50%", " $" + "307" -> "$307").
        if joined and w.text[:1].isspace():
            joined += " "
        offsets.append((len(joined), len(joined) + len(piece), w))
        joined += piece
    span = original_span(joined, excerpt)
    if span is None:
        return None
    a = joined.find(span)
    b = a + len(span)
    hit = [w for (s, e, w) in offsets if s < b and e > a]
    return (hit[0].start, hit[-1].end) if hit else None


def build_document(segments: list[TranscriptSegment], max_chars: int) -> TranscriptDocument:
    """Join segment texts (paragraph break after long pauses), stopping before ``max_chars``."""
    out = ""
    included: list[TranscriptSegment] = []
    omitted: list[TranscriptSegment] = []
    prev: Optional[TranscriptSegment] = None
    for s in segments:
        piece = s.text.strip()
        if not piece:
            continue
        sep = "" if prev is None else "\n\n" if s.start - prev.end > PARAGRAPH_GAP_S else " "
        if omitted or len(out) + len(sep) + len(piece) > max_chars:
            omitted.append(s)
            continue
        out += sep + piece
        included.append(s)
        prev = s
    return TranscriptDocument(text=out, segments=included, omitted=omitted)
