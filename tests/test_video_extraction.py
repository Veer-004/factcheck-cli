"""On-screen text tracking, speech/screen alignment and the fact-check input document (no models)."""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402
from video_helpers import ColorEngine, box, seg  # noqa: E402

from factcheck_ocr.engines import EngineFailure, EngineOutput  # noqa: E402
from factcheck_ocr.models import Block, Line, PageResult, PageStats  # noqa: E402
from factcheck_video.frames import ObservationTracker, frames_differ, ocr_frame, read_screen_text, thumbnail  # noqa: E402
from factcheck_video.media import SampledFrame  # noqa: E402
from factcheck_video.merge import build_document, build_units  # noqa: E402
from factcheck_video.models import Modality, ScreenText, StageStatus  # noqa: E402


def page(*blocks: list[tuple[str, float]]) -> PageResult:
    bs = []
    for lines in blocks:
        ls = [Line(text=t, score=s, bbox=(0, 0, 10, 10), box_count=1) for t, s in lines]
        bs.append(Block(text="\n".join(t for t, _ in lines), lines=ls, bbox=(0, 0, 10, 10)))
    return PageResult(source="f", width=10, height=10, engine="x", layout="geometric",
                      text="\n\n".join(b.text for b in bs), blocks=bs, stats=PageStats())


def frame(t: float, colour=(200, 30, 30), text_box=None) -> SampledFrame:
    im = Image.new("RGB", (320, 180), colour)
    if text_box:
        from PIL import ImageDraw

        ImageDraw.Draw(im).rectangle(text_box, fill="white")
    return SampledFrame(requested=t, time=t, image=im)


# ------------------------------------------------------------------ observation tracking


def test_repeated_caption_is_one_observation_with_all_frame_times():
    tr = ObservationTracker(0.8)
    for t in (0.0, 2.0, 4.0):
        tr.add_frame(t, page([("The Amazon is the longest river in South America", 0.97)]))
    tr.add_frame(6.0, page([("Fact check demo", 0.99)]))
    tr.add_frame(8.0, page([("The Amazon is the longest river in South America", 0.95)]))  # reappears later
    assert len(tr.observations) == 2
    amazon = tr.observations[0]
    assert amazon.frame_times == [0.0, 2.0, 4.0, 8.0] and amazon.readings == 4
    assert amazon.min_score == 0.97  # best reading kept


def test_ocr_jitter_in_consecutive_frames_is_merged_and_variant_kept():
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([("Water boils at 100 degrees Celsius", 0.90)]))
    tr.add_frame(2.0, page([("Water bolls at 100 degrees Celsius", 0.86)]))  # one misread letter
    assert len(tr.observations) == 1
    obs = tr.observations[0]
    assert obs.text == "Water boils at 100 degrees Celsius" and "Water bolls at 100 degrees Celsius" in obs.variants


def test_digit_letter_confusion_is_conservatively_kept_apart():
    # "boi1s" looks like a code (letters+digits, as in "A380"), so it is not merged: an extra
    # observation is preferred over silently merging a possible change of figure.
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([("Water boils at 100 degrees Celsius", 0.90)]))
    tr.add_frame(2.0, page([("Water boi1s at 100 degrees Celsius", 0.86)]))
    assert len(tr.observations) == 2


@pytest.mark.parametrize("a,b", [
    ("Completed in 1889", "Completed in 1887"),
    ("The wall is visible from the Moon", "The wall is not visible from the Moon"),
    ("Speed limit 50 km/h", "Speed limit 50 mph"),
])
def test_near_identical_but_meaningfully_different_text_is_not_merged(a, b):
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([(a, 0.95)]))
    tr.add_frame(2.0, page([(b, 0.95)]))
    assert [o.text for o in tr.observations] == [a, b]


def test_jitter_merge_only_between_consecutive_samples():
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([("Water boils at 100 degrees Celsius", 0.9)]))
    tr.add_frame(2.0, page([("Something else entirely", 0.9)]))
    tr.add_frame(4.0, page([("Water bolls at 100 degrees Celsius", 0.9)]))
    assert len(tr.observations) == 3  # not consecutive, not identical -> kept apart


def test_title_and_subtitle_tracked_independently():
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([("WORLD FACTS", 0.99)], [("Everest is 8849 m tall", 0.95)]))
    tr.add_frame(2.0, page([("WORLD FACTS", 0.99)], [("K2 is 8611 m tall", 0.95)]))
    texts = {o.text: o.frame_times for o in tr.observations}
    assert texts == {"WORLD FACTS": [0.0, 2.0], "Everest is 8849 m tall": [0.0], "K2 is 8611 m tall": [2.0]}


def test_low_confidence_observation_is_marked_unreliable():
    tr = ObservationTracker(0.8)
    tr.add_frame(0.0, page([("Th3 tovver", 0.55)]))
    assert tr.observations[0].reliable is False


# ------------------------------------------------------------------ frame change detection and OCR reuse


def test_frame_change_detection():
    a = thumbnail(frame(0).image)
    assert not frames_differ(a, thumbnail(frame(1).image))
    assert frames_differ(a, thumbnail(frame(1, text_box=(100, 150, 220, 165)).image))  # a small subtitle appears
    assert frames_differ(a, thumbnail(frame(1, colour=(30, 30, 200)).image))
    # similar brightness, different colour (grayscale would call these equal)
    assert frames_differ(thumbnail(frame(0, colour=(200, 30, 30)).image), thumbnail(frame(1, colour=(30, 60, 200)).image))


def test_unchanged_frames_reuse_ocr_and_keep_timestamps():
    engine = ColorEngine({(200, 30, 30): [("The Eiffel Tower was completed in 1889", 0.96)]})
    frames = [frame(t) for t in (0.0, 2.0, 4.0, 6.0)]
    stage = read_screen_text(frames, engine, threshold=0.8, min_score=0.5, interval=2.0, planned=4)
    assert stage.status == StageStatus.OK and stage.frames_sampled == 4
    assert engine.calls == 1 and stage.frames_ocr == 1
    [obs] = stage.observations
    assert obs.frame_times == [0.0, 2.0, 4.0, 6.0]


def test_no_text_and_failure_statuses():
    stage = read_screen_text([frame(0.0)], ColorEngine({}), threshold=0.8, min_score=0.5, interval=2.0, planned=3)
    assert stage.status == StageStatus.NO_CONTENT and "1 of 3 planned frames" in stage.notes[0]

    class Broken:
        name = "broken"

        def recognize(self, image):
            raise EngineFailure("RapidOCR failed: boom")

    stage = read_screen_text([frame(0.0)], Broken(), threshold=0.8, min_score=0.5, interval=2.0, planned=1)
    assert stage.status == StageStatus.FAILED and "boom" in stage.error


def test_ocr_frame_uses_ocr_module_filtering():
    class Two:
        name = "two"

        def recognize(self, image):
            return EngineOutput(boxes=[box("Readable caption here", 0.95, 20), box("x#", 0.2, 120)])

    p = ocr_frame(Image.new("RGB", (400, 200)), Two(), min_score=0.5)
    assert p.text == "Readable caption here" and p.stats.boxes_below_min_score == 1


def test_unchanged_frame_detection_does_not_hide_low_score_drift():
    # mixed reliability: one reliable block, one unreliable -> LOW_CONFIDENCE
    engine = ColorEngine({(200, 30, 30): [("Good caption text", 0.95)], (30, 30, 200): [("blurry txt", 0.5)]})
    stage = read_screen_text([frame(0.0), frame(2.0, colour=(30, 30, 200))], engine, threshold=0.8, min_score=0.3,
                             interval=2.0, planned=2)
    assert stage.status == StageStatus.LOW_CONFIDENCE


# ------------------------------------------------------------------ alignment and the checked document


def obs(i, text, times, score=0.95) -> ScreenText:
    return ScreenText(observation_id=i, text=text, min_score=score, reliable=score >= 0.8, frame_times=times)


def test_identical_speech_and_screen_text_become_one_unit():
    units, merges = build_units([seg(1, 0.2, 3.7, "Water boils at 100 degrees Celsius at sea level.")],
                                [obs(1, "WATER BOILS AT 100 DEGREES CELSIUS AT SEA LEVEL", [2.0, 4.0])], 2.0)
    assert merges == 1 and len(units) == 1
    u = units[0]
    assert set(u.modalities) == {Modality.SPEECH, Modality.SCREEN}
    assert u.text == "Water boils at 100 degrees Celsius at sea level."  # speech wording preferred
    assert {r.modality for r in u.refs} == {Modality.SPEECH, Modality.SCREEN}
    screen = next(r for r in u.refs if r.modality == Modality.SCREEN)
    assert screen.frame_times == [2.0, 4.0] and screen.end == 6.0  # last seen + one interval


def test_contained_or_similar_text_is_not_merged_before_extraction():
    # The caption is contained in the spoken sentence but means the opposite.
    units, merges = build_units([seg(1, 0.0, 4.0, "It is a myth that the Great Wall is visible from the Moon.")],
                                [obs(1, "The Great Wall is visible from the Moon", [0.0])], 2.0)
    assert merges == 0 and len(units) == 2


def test_units_are_time_ordered_and_document_is_labelled():
    units, _ = build_units([seg(1, 0.5, 3.0, "The Pacific is the largest ocean."), seg(2, 8.0, 10.0, "Light is fast.")],
                           [obs(1, "Canberra is the capital of Australia", [4.0, 6.0])], 2.0)
    assert [u.text for u in units] == ["The Pacific is the largest ocean.", "Canberra is the capital of Australia",
                                       "Light is fast."]
    doc = build_document(units, 4000)
    assert doc.text == ("[speech]\nThe Pacific is the largest ocean.\n\n[on-screen text]\n"
                        "Canberra is the capital of Australia\n\n[speech]\nLight is fast.")
    assert not any(ch.isdigit() for ch in doc.text.replace("Light", ""))  # no timestamps leak into the text


def test_document_character_limit_omits_later_units():
    units, _ = build_units([seg(i, i * 5.0, i * 5.0 + 4, f"Sentence number {'x' * 40} {i}.") for i in range(1, 6)], [], 2.0)
    doc = build_document(units, 150)
    assert len(doc.text) <= 150 and doc.units and doc.omitted
    assert [u.unit_id for u in doc.units + doc.omitted] == [u.unit_id for u in units]  # nothing lost or reordered


def test_locate_finds_units_including_across_boundaries_and_repeats():
    units, _ = build_units([seg(1, 0.0, 2.0, "The Eiffel Tower was completed"), seg(2, 2.0, 3.0, "in 1889."),
                            seg(3, 9.0, 11.0, "Bats are blind.")],
                           [obs(1, "Bats are blind", [20.0])], 2.0)
    doc = build_document(units, 4000)
    assert [u.unit_id for u in doc.locate("was completed in 1889")] == ["U1", "U2"]
    [bats] = doc.locate("Bats are blind")  # spoken, then shown 11 s later with the same words: one unit
    assert set(bats.modalities) == {Modality.SPEECH, Modality.SCREEN} and len(bats.refs) == 2
    assert doc.locate("not in the video") == []


def test_transcript_segments_split_into_sentences_with_word_times():
    from factcheck_video.models import Word
    from factcheck_video.transcribe import split_sentences

    words = [Word(text=f" {t}", start=s, end=s + 0.3, probability=0.9) for t, s in
             [("The", 0.2), ("Pacific", 0.5), ("is", 0.8), ("huge.", 1.1), ("Light", 8.0), ("is", 8.3), ("fast.", 8.6),
              ("The", 9.0), ("U.S.", 9.3), ("has", 9.6), ("50", 9.9), ("states.", 10.2)]]
    parent = seg(7, 0.2, 10.5, "The Pacific is huge. Light is fast. The U.S. has 50 states.", conf=0.8, words=False)
    parent = parent.model_copy(update={"words": words})
    out = split_sentences([parent, seg(8, 11.0, 12.0, "No word times here.", words=False)])
    assert [(s.segment_id, s.text, s.start, round(s.end, 1)) for s in out] == [
        (1, "The Pacific is huge.", 0.2, 1.4), (2, "Light is fast.", 8.0, 8.9),
        (3, "The U.S. has 50 states.", 9.0, 10.5), (4, "No word times here.", 11.0, 12.0)]
    assert all(s.confidence == 0.8 for s in out[:3])


def test_subtitle_fragments_are_attached_to_the_speech_they_transcribe():
    from factcheck_video.merge import attach_subtitles

    spoken = "The fastest a human has ever run is 44.7 kilometers an hour."
    units, _ = build_units([seg(1, 0.0, 3.4, spoken)],
                           [obs(1, "The fastest", [0.0]), obs(2, "has ever run", [2.0]),
                            obs(3, "is 44.7 km/h,", [2.0]),  # different wording: not a subtitle match
                            obs(4, "JAMAICA / BOLT", [0.0, 2.0])], 2.0)
    units, n = attach_subtitles(units, 2.0)
    assert n == 2
    speech = next(u for u in units if Modality.SPEECH in u.modalities)
    assert speech.text == spoken and set(speech.modalities) == {Modality.SPEECH, Modality.SCREEN}
    assert sorted(r.observation_id for r in speech.refs if r.observation_id) == [1, 2]  # provenance kept
    assert sorted(u.text for u in units if u.modalities == [Modality.SCREEN]) == ["JAMAICA / BOLT", "is 44.7 km/h,"]
    assert [u.unit_id for u in units] == [f"U{i}" for i in range(1, len(units) + 1)]
    doc = build_document(units, 4000)
    assert "The fastest\n" not in doc.text and doc.text.startswith("[on-screen text]\nJAMAICA / BOLT")


def test_caption_is_not_attached_when_the_speech_denies_it():
    from factcheck_video.merge import attach_subtitles

    units, _ = build_units([seg(1, 0.0, 4.0, "It is a myth that the Great Wall is visible from the Moon.")],
                           [obs(1, "The Great Wall is visible from the Moon", [0.0, 2.0])], 2.0)
    units, n = attach_subtitles(units, 2.0)
    assert n == 0 and len(units) == 2  # checked separately: on its own the caption asserts the myth


def test_caption_is_not_attached_without_time_overlap():
    from factcheck_video.merge import attach_subtitles

    units, _ = build_units([seg(1, 0.0, 3.0, "Light travels faster than sound.")],
                           [obs(1, "Light travels faster", [20.0])], 2.0)
    assert attach_subtitles(units, 2.0)[1] == 0


def test_caption_straddling_two_segments_is_attached():
    from factcheck_video.merge import attach_subtitles

    units, _ = build_units([seg(1, 0.0, 2.0, "In 1995 he and his brother"), seg(2, 2.0, 4.0, "founded Zip2.")],
                           [obs(1, "his brother founded Zip2", [2.0])], 2.0)
    units, n = attach_subtitles(units, 2.0)
    assert n == 1 and len(units) == 2 and all(Modality.SPEECH in u.modalities for u in units)


def test_caption_goes_to_the_segment_that_contains_it_not_the_previous_one():
    from factcheck_video.merge import attach_subtitles

    # Seen at 18-22 s while both segments are near; "Pretoria" is only in segment 2.
    units, _ = build_units([seg(1, 13.9, 17.2, "But did you know that Elon Musk was raised in South Africa?"),
                            seg(2, 17.8, 22.6, "Elon Musk was born on June 28, 1971 in Pretoria, South Africa.")],
                           [obs(1, "Pretoria", [18.0, 20.0, 22.0])], 2.0)
    units, n = attach_subtitles(units, 2.0)
    assert n == 1
    holder = next(u for u in units if any(r.observation_id == 1 for r in u.refs))
    assert "Pretoria" in holder.text


def test_lone_digits_are_never_subtitles():
    from factcheck_video.merge import attach_subtitles

    units, _ = build_units([seg(1, 0.0, 3.0, "In 1995 he founded Zip 2.")], [obs(1, "2", [0.0])], 2.0)
    assert attach_subtitles(units, 2.0)[1] == 0


def test_word_timing_handles_tokens_split_by_whisper():
    from factcheck_video.models import TranscriptSegment, Word
    from factcheck_video.pipeline import _narrow_to_words

    words = [Word(text=t, start=float(i), end=i + 0.5, probability=0.9)
             for i, t in enumerate([" Gratitude", " boosts", " immunity", " by", " 50", "%", " fast."])]
    s = TranscriptSegment(segment_id=1, start=0, end=7, text="Gratitude boosts immunity by 50% fast.",
                          confidence=0.9, words=words)
    assert _narrow_to_words(s, "boosts immunity by 50%") == (1.0, 5.5)


def test_unicode_text_survives_alignment():
    units, merges = build_units([seg(1, 0.0, 2.0, "La tour Eiffel a été achevée en 1889.")],
                                [obs(1, "LA TOUR EIFFEL A ÉTÉ ACHEVÉE EN 1889", [0.0])], 2.0)
    assert merges == 1 and units[0].text == "La tour Eiffel a été achevée en 1889."
