"""Alignment of speech and on-screen text into one time-ordered fact-check input.

Stage 1 (here, before claim extraction) merges identical text: a speech segment
and an on-screen block whose wording is the same after removing case,
punctuation and spacing become one unit carrying both sources. Burned-in
subtitles are attached to the speech they transcribe (``attach_subtitles``):
on-screen text seen while a segment plays whose words occur, in order, in that
segment. A caption whose spoken sentence contains a negation or denial word
outside the caption stays separate, because on its own it can say the opposite
("It is a myth that the Great Wall is visible from space" vs a caption "The
Great Wall is visible from space"). Everything else stays as separate units.

Stage 2 (``textmatch.claims_duplicate``, inside claim validation) merges claims
that the extraction model stated the same way, only when their critical
details agree. See the README's "Video fact-checking" section.

The text given to claim extraction labels each run of units with its source,
without timestamps, so no time figures can leak into claims:

    [speech]
    The Eiffel Tower was completed in 1889.

    [on-screen text]
    EIFFEL TOWER, PARIS
"""

from __future__ import annotations

from dataclasses import dataclass, field

from factcheck.textnorm import normalize_for_match

from .models import Modality, ScreenText, SourceRef, TranscriptSegment, Unit
from .textmatch import NEGATIONS, loose_key

LABELS = {
    frozenset({Modality.SPEECH}): "[speech]",
    frozenset({Modality.SCREEN}): "[on-screen text]",
    frozenset({Modality.SPEECH, Modality.SCREEN}): "[speech, also shown on screen]",
}


def speech_ref(seg: TranscriptSegment) -> SourceRef:
    return SourceRef(modality=Modality.SPEECH, start=seg.start, end=seg.end, text=seg.text,
                     confidence=seg.confidence, reliable=seg.reliable, segment_id=seg.segment_id)


def screen_ref(obs: ScreenText, interval: float) -> SourceRef:
    return SourceRef(modality=Modality.SCREEN, start=obs.first_seen, end=round(obs.last_seen + interval, 3),
                     text=obs.text, confidence=obs.min_score, reliable=obs.reliable,
                     observation_id=obs.observation_id, frame_times=list(obs.frame_times))


def build_units(segments: list[TranscriptSegment], observations: list[ScreenText], interval: float) -> tuple[list[Unit], int]:
    """Time-ordered units; identical speech/screen text merged. Returns (units, cross-modal merges)."""
    refs = [speech_ref(s) for s in segments if s.text.strip()]
    refs += [screen_ref(o, interval) for o in observations if o.text.strip()]
    refs.sort(key=lambda r: (r.start, r.modality.value))
    units: list[Unit] = []
    by_key: dict[str, Unit] = {}
    merges = 0
    for ref in refs:
        key = loose_key(ref.text)
        if not key:
            continue
        unit = by_key.get(key)
        if unit is not None:
            if ref.modality not in unit.modalities:
                merges += 1
                unit.modalities = sorted({*unit.modalities, ref.modality}, key=lambda m: m.value)
            unit.refs.append(ref)
            if ref.modality == Modality.SPEECH and not any(r.modality == Modality.SPEECH for r in unit.refs[:-1]):
                unit.text = ref.text  # prefer the speech wording (sentence case) as the checked text
            continue
        unit = Unit(unit_id=f"U{len(units) + 1}", text=ref.text, start=ref.start, modalities=[ref.modality], refs=[ref])
        units.append(unit)
        by_key[key] = unit
    return units, merges


# Words outside a caption that can reverse what the caption says on its own
# ("It is a MYTH that the Great Wall is visible ..." shown with the caption
# "The Great Wall is visible ..."). If the rest of the spoken sentence has one,
# the caption is kept as separate on-screen text.
DENIAL_WORDS = NEGATIONS | frozenset(
    """myth myths false untrue fake hoax wrong debunked misconception contrary incorrect rumor rumour rumors
    rumours claim claims claimed alleged allegedly supposedly""".split())


MIN_SUBTITLE_CHARS = 3  # shorter captions (after removing spaces/punctuation) are never treated as subtitles


def _caption_span(caption: str, sentence: str) -> tuple[int, int] | None:
    """Word positions [a, b) where ``caption`` occurs contiguously in ``sentence`` (loose keys)."""
    c, s = caption.split(), sentence.split()
    if not c or len(c) > len(s):
        return None
    for i in range(len(s) - len(c) + 1):
        if s[i:i + len(c)] == c:
            return i, i + len(c)
    return None


def attach_subtitles(units: list[Unit], interval: float) -> tuple[list[Unit], int]:
    """Attach on-screen text that is a subtitle of overlapping speech to that speech unit.

    A screen-only unit is a subtitle when it is seen while a speech segment plays
    (within one sampling interval) and its words occur, in order and contiguously,
    in that segment (or in that segment joined with the next one, for captions that
    straddle a segment boundary). It is then checked once, in the spoken sentence's
    full context, and stays in the unit's provenance as a screen source. Not
    attached when the rest of the spoken sentence contains a negation or denial word
    (``DENIAL_WORDS``). Returns (units in time order with fresh ids, captions attached).
    """
    speech = [u for u in units if Modality.SPEECH in u.modalities]
    speech.sort(key=lambda u: u.start)

    def span(u: Unit) -> tuple[float, float]:
        return min(r.start for r in u.refs), max(r.end for r in u.refs)

    def denies(sentence: str, hit: tuple[int, int]) -> bool:
        outside = sentence.split()[:hit[0]] + sentence.split()[hit[1]:]
        return any(w in DENIAL_WORDS or w.endswith("n't") for w in outside)

    kept: list[Unit] = []
    attached = 0
    for u in units:
        cap = loose_key(u.text)
        if u.modalities != [Modality.SCREEN] or len(cap.replace(" ", "")) < MIN_SUBTITLE_CHARS:
            kept.append(u)  # lone digits or letters ("1", "2") match spoken text by coincidence
            continue
        a0, a1 = span(u)
        near = [i for i, sp in enumerate(speech)
                if not (a1 < span(sp)[0] - interval or a0 > span(sp)[1] + interval)]
        target = None
        for i in near:  # the caption inside one segment
            sentence = loose_key(speech[i].text)
            hit = _caption_span(cap, sentence)
            if hit is not None and not denies(sentence, hit):
                target = speech[i]
                break
        if target is None:  # the caption crossing the boundary between two consecutive segments
            for i in near:
                if i + 1 >= len(speech):
                    continue
                first = loose_key(speech[i].text)
                sentence = f"{first} {loose_key(speech[i + 1].text)}"
                hit = _caption_span(cap, sentence)
                n_first = len(first.split())
                if hit is not None and hit[0] < n_first < hit[1] and not denies(sentence, hit):
                    target = speech[i]
                    break
        if target is None:
            kept.append(u)
            continue
        target.refs.extend(u.refs)
        target.modalities = sorted({*target.modalities, Modality.SCREEN}, key=lambda m: m.value)
        attached += 1
    kept.sort(key=lambda u: u.start)
    for n, u in enumerate(kept, 1):
        u.unit_id = f"U{n}"
    return kept, attached


@dataclass
class Document:
    text: str
    units: list[Unit]  # the units included in ``text``, in order
    omitted: list[Unit] = field(default_factory=list)  # over the character limit

    def locate(self, excerpt: str) -> list[Unit]:
        """Units spanned by any occurrence of the (normalised) excerpt; [] if not found."""
        target = normalize_for_match(excerpt)
        if not target:
            return []
        parts, spans, pos = [], [], 0
        for u in self.units:
            norm = normalize_for_match(u.text)
            spans.append((pos, pos + len(norm)))
            parts.append(norm)
            pos += len(norm) + 1
        joined = " ".join(parts)
        hits: list[Unit] = []
        start = joined.find(target)
        while start >= 0:
            end = start + len(target)
            for u, (a, b) in zip(self.units, spans):
                if a < end and b > start and u not in hits:
                    hits.append(u)
            start = joined.find(target, start + 1)
        return hits


def build_document(units: list[Unit], max_chars: int) -> Document:
    """Render units in time order with source labels, stopping before ``max_chars``."""
    out: list[str] = []
    included: list[Unit] = []
    omitted: list[Unit] = []
    size = 0
    prev_label = None
    for unit in units:
        label = LABELS[frozenset(unit.modalities)]
        piece = unit.text.strip()
        add = len(piece) + 1 + (len(label) + 2 if label != prev_label else 0)
        if omitted or size + add > max_chars:
            omitted.append(unit)
            continue
        if label != prev_label:
            if out:
                out.append("")
            out.append(label)
            prev_label = label
        out.append(piece)
        included.append(unit)
        size += add
    return Document(text="\n".join(out), units=included, omitted=omitted)
