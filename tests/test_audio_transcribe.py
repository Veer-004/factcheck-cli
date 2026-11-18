"""English-only transcription policy, transcript document and timestamp tracing (no real model)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from audio_helpers import seg

from factcheck_audio.document import PARAGRAPH_GAP_S, build_document, narrow_to_words
from factcheck_audio.models import TranscriptSegment
from factcheck_audio.pipeline import fmt_time, language_check
from factcheck_audio.transcribe import AudioTranscript, EnglishTranscriber, latin_share, mark_non_latin
from factcheck_video.models import Word

# ------------------------------------------------------------------ the English-only transcriber


def raw_segment(start, end, text, logprob=-0.1):
    toks = text.split()
    step = (end - start) / len(toks)
    words = [SimpleNamespace(word=" " + t, start=start + i * step, end=start + (i + 1) * step, probability=0.9)
             for i, t in enumerate(toks)]
    return SimpleNamespace(start=start, end=end, text=" " + text, avg_logprob=logprob, no_speech_prob=0.01,
                           compression_ratio=1.3, words=words)


class FakeWhisperModel:
    def __init__(self, language, probability, segments, all_probs=None):
        self.info = SimpleNamespace(language=language, language_probability=probability, duration=10.0,
                                    all_language_probs=all_probs if all_probs is not None else [(language, probability)])
        self.segments = segments
        self.calls = []
        self.iterated = False

    def transcribe(self, path, **kw):
        self.calls.append(kw)

        def gen():
            self.iterated = True
            yield from self.segments

        return gen(), self.info


def transcriber(model, language=None, method="detected") -> EnglishTranscriber:
    t = object.__new__(EnglishTranscriber)  # skip loading a real model
    t._model, t.language, t.method = model, language, method
    t.name, t.compute_type, t.beam_size, t.vad, t.word_timestamps, t.threshold = "fake", "int8", 5, True, True, 0.4
    return t


def test_english_audio_is_detected_and_decoded_with_sentence_times():
    model = FakeWhisperModel("en", 0.97, [raw_segment(0.0, 4.0, "Water boils at 100 degrees. Ice melts at zero.")],
                             [("en", 0.97), ("de", 0.01)])
    tr = transcriber(model).transcribe("x.wav")
    assert model.calls[0]["language"] is None and model.calls[0]["language_detection_segments"] == 3
    assert model.calls[0]["condition_on_previous_text"] is False
    assert [s.text for s in tr.segments] == ["Water boils at 100 degrees.", "Ice melts at zero."]
    # 9 evenly spaced words over 4 s: "Ice" is word 6 -> starts at 5 * 4/9 s
    assert tr.segments[1].start == pytest.approx(5 * 4 / 9, abs=0.01) and tr.english_probability == 0.97
    assert tr.decoded


def test_non_english_audio_is_not_decoded_at_all():
    model = FakeWhisperModel("hi", 0.99, [raw_segment(0.0, 2.0, "yah ek missile hai")], [("hi", 0.99), ("en", 0.004)])
    tr = transcriber(model).transcribe("x.wav")
    assert tr.segments == [] and tr.language == "hi" and not tr.decoded
    assert model.iterated is False  # stopped before any decoding work
    check = language_check(tr)
    assert not check.accepted and "only English is supported" in check.note and "--assume-english" in check.note


def test_assume_english_forces_the_language():
    model = FakeWhisperModel("en", 1.0, [raw_segment(0.0, 2.0, "Hello there.")], all_probs=[])
    tr = transcriber(model, language="en", method="assumed").transcribe("x.wav")
    assert model.calls[0]["language"] == "en" and len(tr.segments) == 1
    check = language_check(tr)
    assert check.accepted and check.method == "assumed" and "--assume-english" in check.note


def test_uncertain_english_is_accepted_with_a_note():
    check = language_check(AudioTranscript([], "en", 0.45, 1.0, [], english_probability=0.45))
    assert check.accepted and "low certainty" in check.note


def test_transcription_errors_become_transcription_failed():
    from factcheck_audio.transcribe import TranscriptionFailed

    class Broken(FakeWhisperModel):
        def transcribe(self, path, **kw):
            raise RuntimeError("ctranslate2 out of memory")

    with pytest.raises(TranscriptionFailed, match="out of memory"):
        transcriber(Broken("en", 1, [])).transcribe("x.wav")


@pytest.mark.parametrize("text,share", [("The Eiffel Tower", 1.0), ("یہ ہے اگنی", 0.0), ("Agni 5 یہ ہے", 0.5),
                                        ("1889 1990", 1.0), ("Café déjà vu", 1.0)])
def test_latin_share(text, share):
    assert latin_share(text) == pytest.approx(share, abs=0.01)


def test_non_latin_segments_are_marked_unreliable():
    segs, n = mark_non_latin([seg(1, 0, 1, "The Moon orbits Earth."), seg(2, 1, 2, "یہ ہے اگنی 5 missiles")])
    assert n == 1 and [s.reliable for s in segs] == [True, False]


# ------------------------------------------------------------------ document and timestamps


def test_document_is_plain_prose_with_paragraphs_after_long_pauses():
    segs = [seg(1, 0.0, 2.0, "The Pacific is the largest ocean."), seg(2, 2.2, 4.0, "It covers a third of Earth."),
            seg(3, 4.0 + PARAGRAPH_GAP_S + 1, 9.0, "Light is fast.")]
    doc = build_document(segs, 4000)
    assert doc.text == "The Pacific is the largest ocean. It covers a third of Earth.\n\nLight is fast."
    assert not any(ch.isdigit() for ch in doc.text)  # no timestamps leak into the text


def test_document_character_limit_keeps_order_and_reports_omitted():
    segs = [seg(i, i * 3.0, i * 3.0 + 2, f"Sentence number {'x' * 40} {i}.") for i in range(1, 6)]
    doc = build_document(segs, 150)
    assert len(doc.text) <= 150 and doc.omitted
    assert [s.segment_id for s in doc.segments + doc.omitted] == [1, 2, 3, 4, 5]


def test_locate_with_word_level_times_inside_one_segment():
    # 9 evenly spaced words over 0-9 s -> one word per second; the claim starts at word 3
    doc = build_document([seg(1, 0.0, 9.0, "Hello everyone. The Eiffel Tower was completed in 1889.")], 4000)
    [occ] = doc.locate("The Eiffel Tower was completed in 1889")
    assert occ.word_level and (occ.start, occ.end) == (pytest.approx(2.0), pytest.approx(9.0))


def test_locate_across_a_segment_boundary():
    doc = build_document([seg(1, 0.0, 3.0, "In 1995 he and his brother"), seg(2, 3.0, 5.0, "founded Zip2."),
                          seg(3, 9.0, 11.0, "Bats are blind.")], 4000)
    [occ] = doc.locate("his brother founded Zip2")
    assert [s.segment_id for s in occ.segments] == [1, 2] and occ.word_level
    assert occ.start == pytest.approx(2.0) and occ.end == pytest.approx(5.0)  # "his" = word 5 of 6 over 0-3 s
    assert doc.locate("not in the audio") == []


def test_locate_finds_every_repeat():
    doc = build_document([seg(1, 0.0, 2.0, "Bats are blind."), seg(2, 5.0, 6.0, "Okay."),
                          seg(3, 30.0, 32.0, "Bats are blind.")], 4000)
    assert [o.segments[0].segment_id for o in doc.locate("Bats are blind")] == [1, 3]


def test_segment_times_are_used_without_word_timestamps():
    s = TranscriptSegment(segment_id=1, start=3.0, end=6.5, text="The Nile is in Africa.", confidence=0.8)
    [occ] = build_document([s], 4000).locate("The Nile is in Africa")
    assert (occ.start, occ.end, occ.word_level) == (3.0, 6.5, False)
    assert narrow_to_words([s], "Nile") is None


def test_word_timing_survives_punctuation_differences():
    words = [Word(text=t, start=i, end=i + 0.5, probability=0.9) for i, t in
             enumerate([" It", " cost", " $307", " million,", " in", " 1999."])]
    s = TranscriptSegment(segment_id=1, start=0, end=6, text="It cost $307 million, in 1999.", confidence=0.9,
                          words=words)
    assert narrow_to_words([s], "cost $307 million") == (1, 3.5)


def test_word_timing_with_split_tokens_across_segments():
    # Observed on a real podcast: Whisper emits "50%" as " 50" + "%" and the claim spans two segments.
    a = TranscriptSegment(segment_id=36, start=128.0, end=134.0, confidence=0.88,
                          text="I know that gratitude strengthens my immune system",
                          words=[Word(text=t, start=128 + i, end=128.5 + i, probability=0.9) for i, t in
                                 enumerate([" I", " know", " that", " gratitude", " strengthens", " my", " immune",
                                            " system"])])
    b = TranscriptSegment(segment_id=37, start=136.0, end=140.0, confidence=0.88, text="by 50% and if",
                          words=[Word(text=t, start=136 + i, end=136.5 + i, probability=0.9) for i, t in
                                 enumerate([" by", " 50", "%", " and", " if"])])
    [occ] = build_document([a, b], 4000).locate("gratitude strengthens my immune system by 50%")
    assert occ.word_level and (occ.start, occ.end) == (131, 138.5)


@pytest.mark.parametrize("seconds,text", [(0, "00:00.0"), (75.25, "01:15.2"), (3725.0, "1:02:05.0")])
def test_time_format_handles_long_recordings(seconds, text):
    assert fmt_time(seconds) == text
