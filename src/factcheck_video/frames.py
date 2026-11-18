"""On-screen text: sampled frames -> the project's OCR pipeline -> merged observations.

Only text is read; objects, scenes and events in the frames are never
interpreted. Each OCR block (a caption, title card or label) is tracked across
sampled frames:

* a frame that looks unchanged from the last OCR'd frame (tiny thumbnail
  difference) reuses that frame's OCR result instead of running OCR again;
* a block whose text is identical after normalisation to one already seen is
  merged into it (repeated subtitles, persistent titles), keeping every frame
  time;
* a block seen in the *next* sampled frame with nearly identical text (OCR
  jitter such as "1889" vs "1889." or one changed letter) is merged too, but
  only when its digits and negation words are identical, so "1889" vs "1887"
  or "is" vs "is not" always stay separate observations.
"""

from __future__ import annotations

import difflib
import time
from typing import Any, Iterable, Optional

from factcheck_ocr.engines import EngineFailure, OcrEngine
from factcheck_ocr.models import PageResult, PageStats
from factcheck_ocr.pipeline import filter_notes, process_boxes, render_blocks

from .media import Deadline, SampledFrame
from .models import ScreenLine, ScreenStage, ScreenText, StageStatus
from .textmatch import critical_signature, loose_key

THUMB_SIZE = (160, 90)  # RGB thumbnail used to detect unchanged frames
PIXEL_DELTA = 0.10  # change (0-1) in any colour channel that counts a thumbnail pixel as changed
CHANGED_FRACTION = 0.0005  # frames with fewer changed pixels than this reuse the previous OCR result
JITTER_RATIO = 0.92  # character similarity for merging OCR variants seen in consecutive samples


def thumbnail(image: Any) -> Any:
    import numpy as np

    # RGB, not grayscale: coloured text on a background of similar brightness must still count as a change
    return np.asarray(image.convert("RGB").resize(THUMB_SIZE), dtype=np.float32) / 255.0


def frames_differ(a: Any, b: Any) -> bool:
    """True unless (almost) no thumbnail pixel changed visibly.

    A subtitle changes few pixels, so an average difference would miss it; this
    counts pixels whose brightness changed by more than ``PIXEL_DELTA`` instead
    (compression noise stays below that).
    """
    import numpy as np

    if a is None or b is None or a.shape != b.shape:
        return True
    return float((np.abs(a - b).max(axis=-1) > PIXEL_DELTA).mean()) > CHANGED_FRACTION


def ocr_frame(image: Any, engine: OcrEngine, *, min_score: float, layout: str = "auto",
              noise_filter: bool = True) -> PageResult:
    """The OCR module's processing for an in-memory frame (same box filtering, English clean-up and
    reading order as ``factcheck-ocr``)."""
    started = time.perf_counter()
    raw = engine.recognize(image)
    blocks, used, stats = process_boxes(raw, scale=1.0, min_score=min_score, layout=layout,
                                        noise_filter=noise_filter)
    stats.seconds = round(time.perf_counter() - started, 3)
    return PageResult(source="frame", width=image.width, height=image.height, engine=engine.name, layout=used,
                      text=render_blocks(blocks), blocks=blocks, stats=stats, warnings=raw.notes)


class ObservationTracker:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.observations: list[ScreenText] = []
        self._by_key: dict[str, ScreenText] = {}
        self._previous: list[ScreenText] = []  # observations seen in the previous sampled frame

    def add_frame(self, t: float, page: PageResult) -> None:
        current: list[ScreenText] = []
        for block in page.blocks:
            lines = [ScreenLine(text=l.text, score=round(l.score, 4)) for l in block.lines if l.text.strip()]
            if not lines:
                continue
            text = "\n".join(l.text for l in lines)
            obs = self._match(text)
            min_score = min(l.score for l in lines)
            if obs is None:
                obs = ScreenText(observation_id=len(self.observations) + 1, text=text, lines=lines,
                                 min_score=min_score, reliable=min_score >= self.threshold, frame_times=[t])
                self.observations.append(obs)
            else:
                if t not in obs.frame_times:
                    obs.frame_times.append(t)
                obs.readings += 1
                if text != obs.text and text not in obs.variants and len(obs.variants) < 5:
                    obs.variants.append(text)
                if min_score > obs.min_score:  # keep the best reading as the representative text
                    if obs.text not in obs.variants and text != obs.text and len(obs.variants) < 5:
                        obs.variants.append(obs.text)
                    obs.text, obs.lines, obs.min_score = text, lines, min_score
                    obs.reliable = min_score >= self.threshold
                    if text in obs.variants:
                        obs.variants.remove(text)
            self._by_key.setdefault(loose_key(text), obs)
            current.append(obs)
        self._previous = current

    def _match(self, text: str) -> Optional[ScreenText]:
        key = loose_key(text)
        if not key:
            return None
        if key in self._by_key:
            return self._by_key[key]
        sig = critical_signature(text)
        for obs in self._previous:
            if (critical_signature(obs.text) == sig
                    and difflib.SequenceMatcher(None, key, loose_key(obs.text)).ratio() >= JITTER_RATIO):
                return obs
        return None


def read_screen_text(frames: Iterable[SampledFrame], engine: OcrEngine, *, threshold: float, min_score: float,
                     interval: float, planned: int, deadline: Optional[Deadline] = None,
                     noise_filter: bool = True) -> ScreenStage:
    stage = ScreenStage(status=StageStatus.NO_CONTENT, engine=engine.name, threshold=threshold, interval_s=interval)
    tracker = ObservationTracker(threshold)
    started = time.perf_counter()
    last_thumb, last_page = None, None
    totals = PageStats()
    try:
        for frame in frames:
            stage.frames_sampled += 1
            thumb = thumbnail(frame.image)
            if last_page is None or frames_differ(thumb, last_thumb):
                if deadline is not None:
                    deadline.check("on-screen text OCR")
                last_page = ocr_frame(frame.image, engine, min_score=min_score, noise_filter=noise_filter)
                last_thumb = thumb
                stage.frames_ocr += 1
                totals.boxes_filtered_noise += last_page.stats.boxes_filtered_noise
                totals.boxes_small_figures += last_page.stats.boxes_small_figures
            tracker.add_frame(frame.time, last_page)
    except EngineFailure as exc:
        stage.status, stage.error = StageStatus.FAILED, str(exc)
    stage.seconds = round(time.perf_counter() - started, 2)
    stage.observations = tracker.observations
    if stage.frames_sampled < planned:
        stage.notes.append(f"{stage.frames_sampled} of {planned} planned frames could be decoded")
    stage.notes += [f"across all frames: {n}" for n in filter_notes(totals)]
    if stage.status == StageStatus.FAILED:
        return stage
    if stage.observations:
        reliable = sum(o.reliable for o in stage.observations)
        stage.status = (StageStatus.OK if reliable == len(stage.observations)
                        else StageStatus.UNRELIABLE if reliable == 0 else StageStatus.LOW_CONFIDENCE)
    return stage
