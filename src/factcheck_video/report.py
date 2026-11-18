"""Human-readable and JSON video reports.

The fact-check part is rendered by the shared ``factcheck.report.render_text``;
this module adds the extraction summary, the timed transcript / on-screen text,
and per-claim provenance lines.
"""

from __future__ import annotations

import json
import textwrap

from factcheck.report import WIDTH, render_text

from .models import Modality, StageStatus
from .pipeline import ClaimSource, VideoReport, fmt_time

MAX_ROWS = 40

_STATUS = {
    StageStatus.OK: "OK",
    StageStatus.LOW_CONFIDENCE: "LOW CONFIDENCE (some parts below the threshold)",
    StageStatus.UNRELIABLE: "UNRELIABLE (everything below the threshold)",
    StageStatus.NO_CONTENT: "nothing found",
    StageStatus.NO_STREAM: "no track",
    StageStatus.SKIPPED: "skipped",
    StageStatus.FAILED: "FAILED",
}


def _fill(text: str, indent: str = "", sub: str = "   ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=sub or indent)


def _span_range(start: float, end: float) -> str:
    return f"~{fmt_time(start)}-{fmt_time(end)}"


def appearances(frame_times: list[float], interval: float) -> str:
    """Sampled frame times as runs, e.g. "00:00.0-00:04.0, 00:08.0-00:10.0" (a gap = not seen in between)."""
    runs: list[list[float]] = []
    for t in sorted(frame_times):
        if runs and t - runs[-1][-1] <= interval * 1.5 + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return ", ".join(fmt_time(r[0]) if len(r) == 1 else f"{fmt_time(r[0])}-{fmt_time(r[-1])}" for r in runs)


def provenance_lines(src: ClaimSource, interval: float) -> list[str]:
    if not src.located:
        return ["Source in video: not traced to a specific segment or text block."]
    out = []
    for s in src.spans:
        flag = "" if s.reliable else ", below the threshold"
        if s.modality == Modality.SPEECH:
            words = " (word timing)" if s.word_level else ""
            out.append(f"Spoken {_span_range(s.start, s.end)}{words}, segment {s.segment_id}, "
                       f"speech confidence {s.confidence:.2f}{flag}")
        else:
            shown = (f"seen in sampled frames {appearances(s.frame_times, interval)}"
                     if len(s.frame_times) > 1 else f"seen in the frame at {fmt_time(s.frame_times[0])}")
            out.append(f"On screen: {shown}, text block {s.observation_id}, OCR confidence {s.confidence:.2f}{flag}")
    if src.merged:
        out.append(f"Merged duplicate(s): {len(src.merged)} identical claim(s) from other places in the video were "
                   "checked once with this one.")
    return out


def render_video_report(vr: VideoReport, verbose: bool = False) -> str:
    ex = vr.extraction
    m, a, s = ex.media, ex.audio, ex.screen
    out = ["VIDEO FACT-CHECK", ""]
    dims = f"{m.width}x{m.height}" if m.width else "no video track"
    dur = f"{m.duration_s:.1f} s" if m.duration_s is not None else "unknown length"
    out.append(_fill(f"Video: {m.path} ({dur}, {dims}, {m.container or '?'}; "
                     f"audio {'yes' if m.has_audio else 'none'})"))
    speech = f"Speech: {_STATUS[a.status]}"
    if a.model:
        speech += f" - faster-whisper {a.model} ({a.compute_type})"
    if a.language:
        speech += f", language {a.language} (p={a.language_probability:.2f})"
    if a.segments:
        speech += f", {len(a.segments)} segment(s)"
    out.append(_fill(speech))
    out += [_fill(f"- {n}", "   ", "     ") for n in a.notes + ([a.error] if a.error else [])]
    screen = f"On-screen text: {_STATUS[s.status]}"
    if s.engine:
        screen += (f" - OCR {s.engine}, {s.frames_sampled} frame(s) sampled every {s.interval_s:.2f} s, "
                   f"{s.frames_ocr} OCR'd, {len(s.observations)} distinct text block(s)")
    out.append(_fill(screen))
    out += [_fill(f"- {n}", "   ", "     ") for n in s.notes + ([s.error] if s.error else [])]
    out += [_fill(f"- warning: {w}", "   ", "     ") for w in ex.warnings]
    if ex.exact_merges:
        out.append(f"   - {ex.exact_merges} spoken segment(s) were also shown on screen word for word (merged)")
    if ex.subtitle_merges:
        out.append(_fill(f"- {ex.subtitle_merges} on-screen caption(s) were subtitles of the speech and are checked "
                         "with the spoken sentence", "   ", "     "))

    if a.segments:
        out += ["", f"Transcript (automatic; '?' marks segments below confidence {a.threshold:.2f}):"]
        rows = a.segments if verbose else a.segments[:MAX_ROWS]
        for seg in rows:
            mark = " " if seg.reliable else "?"
            conf = f" [{seg.confidence:.2f}]" if verbose else ""
            out.append(_fill(f"{mark} {fmt_time(seg.start)}-{fmt_time(seg.end)}{conf}  {seg.text}", "", " " * 17))
        if len(rows) < len(a.segments):
            out.append(f"   ... {len(a.segments) - len(rows)} more segment(s) (use --verbose or --json-out)")
    if s.observations:
        out += ["", f"On-screen text (OCR of sampled frames; '?' marks text below confidence {s.threshold:.2f}):"]
        rows = s.observations if verbose else s.observations[:MAX_ROWS]
        for o in rows:
            mark = " " if o.reliable else "?"
            when = appearances(o.frame_times, s.interval_s) if len(o.frame_times) > 1 else f"{fmt_time(o.first_seen)}      "
            conf = f" [{o.min_score:.2f}]" if verbose else ""
            out.append(_fill(f"{mark} {when}{conf}  {' / '.join(o.text.splitlines())}", "", " " * 17))
        if len(rows) < len(s.observations):
            out.append(f"   ... {len(s.observations) - len(rows)} more block(s) (use --verbose or --json-out)")

    out += ["", _fill(vr.timestamp_note, "", ""), _fill(vr.confidence_note, "", ""), ""]
    if vr.fact_check is None:
        out.append(_fill(f"NOT FACT-CHECKED: {vr.not_checked_reason}", "", "  "))
        return "\n".join(out)
    extras = {src.claim_id: provenance_lines(src, s.interval_s) for src in vr.claim_sources}
    out.append(render_text(vr.fact_check, verbose=verbose, source_label="the video", claim_extras=extras))
    t = vr.timing
    out.append("")
    out.append(_fill(f"Time: extraction {t.extraction_total_s:.1f} s (audio {t.audio_extract_s:.1f} s, transcription "
                     f"{t.transcribe_s:.1f} s, frames+OCR {t.frames_ocr_s:.1f} s), fact-check {t.fact_check_s:.1f} s",
                     "", ""))
    return "\n".join(out)


def render_video_json(vr: VideoReport) -> str:
    return json.dumps(vr.model_dump(mode="json"), ensure_ascii=False, indent=2)
