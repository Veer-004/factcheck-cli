"""Human-readable and JSON audio reports.

The fact-check part is rendered by the shared ``factcheck.report.render_text``;
this module adds the audio metadata, language identification, the timed
transcript and per-claim "where was it said" lines.
"""

from __future__ import annotations

import json
import textwrap

from factcheck.report import WIDTH, render_text

from .models import TranscriptionStatus
from .pipeline import AudioReport, ClaimSource, fmt_time

MAX_ROWS = 60

_STATUS = {
    TranscriptionStatus.OK: "OK",
    TranscriptionStatus.LOW_CONFIDENCE: "LOW CONFIDENCE (some segments below the threshold)",
    TranscriptionStatus.UNRELIABLE: "UNRELIABLE (every segment below the threshold)",
    TranscriptionStatus.NO_SPEECH: "no speech found",
    TranscriptionStatus.UNSUPPORTED_LANGUAGE: "NOT TRANSCRIBED (language not supported)",
    TranscriptionStatus.FAILED: "FAILED",
}


def _fill(text: str, indent: str = "", sub: str = "   ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=sub or indent)


def _size(n: int) -> str:
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 1024:.0f} KB"


def provenance_lines(src: ClaimSource) -> list[str]:
    if not src.located:
        return ["Where in the audio: not traced to a specific transcript segment."]
    out = []
    for s in src.spans:
        flag = "" if s.reliable else ", below the threshold"
        words = " (word timing)" if s.word_level else ""
        segs = ", ".join(map(str, s.segment_ids))
        out.append(f"Spoken ~{fmt_time(s.start)}-{fmt_time(s.end)}{words}, segment {segs}, "
                   f"transcription confidence {s.confidence:.2f}{flag}")
    if len(src.spans) > 1:
        out.append(f"Said {len(src.spans)} times in the audio; checked once.")
    if src.merged:
        out.append(f"Merged repeat(s): {len(src.merged)} identical claim(s) from elsewhere in the audio were checked "
                   "once with this one.")
    return out


def render_audio_report(ar: AudioReport, verbose: bool = False) -> str:
    m, t = ar.media, ar.transcription
    out = ["AUDIO FACT-CHECK", ""]
    dur = f"{m.duration_s:.1f} s" if m.duration_s is not None else "unknown length"
    details = [dur, m.codec or "?", f"{m.sample_rate} Hz" if m.sample_rate else None,
               f"{m.channels} ch" if m.channels else None, _size(m.size_bytes)]
    out.append(_fill(f"Audio: {m.path} ({', '.join(d for d in details if d)})"))
    line = f"Transcription: {_STATUS[t.status]}"
    if t.model:
        line += f" - faster-whisper {t.model} ({t.compute_type})"
    if t.segments:
        line += f", {len(t.segments)} segment(s)"
    out.append(_fill(line))
    lang = t.language
    if lang.detected or lang.method != "detected":
        p = f" (p={lang.probability:.2f})" if lang.probability is not None and lang.method == "detected" else ""
        verdict = "supported" if lang.accepted else "NOT supported (English only)"
        out.append(_fill(f"Language: {lang.detected or 'en'}{p}, {lang.method}; {verdict}", "   ", "     "))
    out += [_fill(f"- {n}", "   ", "     ") for n in t.notes + ([t.error] if t.error else [])]
    out += [_fill(f"- warning: {w}", "   ", "     ") for w in ar.warnings]

    if t.segments:
        out += ["", f"Transcript (automatic; '?' marks segments below confidence {t.threshold:.2f}):"]
        rows = t.segments if verbose else t.segments[:MAX_ROWS]
        for seg in rows:
            mark = " " if seg.reliable else "?"
            conf = f" [{seg.confidence:.2f}]" if verbose else ""
            stamp = f"{fmt_time(seg.start)}-{fmt_time(seg.end)}"
            out.append(_fill(f"{mark} {stamp}{conf}  {seg.text}", "", " " * (len(stamp) + 4)))
        if len(rows) < len(t.segments):
            out.append(f"   ... {len(t.segments) - len(rows)} more segment(s) (use --verbose or --json-out)")

    out += ["", _fill(ar.timestamp_note, "", ""), _fill(ar.confidence_note, "", ""), ""]
    if ar.fact_check is None:
        out.append(_fill(f"NOT FACT-CHECKED: {ar.not_checked_reason}", "", "  "))
        return "\n".join(out)
    extras = {src.claim_id: provenance_lines(src) for src in ar.claim_sources}
    out.append(render_text(ar.fact_check, verbose=verbose, source_label="the audio", claim_extras=extras))
    tm = ar.timing
    rtf = f", {tm.real_time_factor:.2f}x real time" if tm.real_time_factor is not None else ""
    out += ["", _fill(f"Time: transcription {tm.extraction_total_s:.1f} s (decode {tm.decode_s:.1f} s, speech "
                      f"recognition {tm.transcribe_s:.1f} s{rtf}), fact-check {tm.fact_check_s:.1f} s", "", "")]
    return "\n".join(out)


def render_audio_json(ar: AudioReport) -> str:
    return json.dumps(ar.model_dump(mode="json"), ensure_ascii=False, indent=2)
