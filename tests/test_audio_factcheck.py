"""Audio fact-checking end to end, fully offline.

Real encoded audio files are decoded by PyAV into a temporary 16 kHz WAV. A
scripted transcriber stands in for faster-whisper and the fake LLM/search
providers from ``fakes.py`` for Gemini and Tavily. Everything else (the shared
fact-checking engine included) is the real code.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("av")

from audio_helpers import FakeTranscriber, noise, seg, silence, tone, write_audio  # noqa: E402
from conftest import load_page  # noqa: E402
from fakes import FakeLLM, FakeSearch, claim_dict, extraction, result  # noqa: E402

from factcheck.graph import RunFailure  # noqa: E402
from factcheck.llm import LLMAuthError, LLMUnavailableError  # noqa: E402
from factcheck.models import SummaryLabel, Verdict  # noqa: E402
from factcheck_audio import cli as audio_cli  # noqa: E402
from factcheck_audio.audio_io import AudioInputError  # noqa: E402
from factcheck_audio.models import AudioLimits, TranscriptionStatus  # noqa: E402
from factcheck_audio.pipeline import AUDIO_SOURCE_NOTE, AudioReport, verify_audio  # noqa: E402
from factcheck_audio.report import render_audio_json, render_audio_report  # noqa: E402
from factcheck_audio.transcribe import ModelUnavailable, TranscriptionFailed  # noqa: E402

GOV = "https://www.tourism.example.gov/eiffel"
BBC = "https://www.bbc.co.uk/news/eiffel"
CORPUS = {"eiffel": [result(GOV, "The Eiffel Tower - official history", "The Eiffel Tower was completed in 1889."),
                     result(BBC, "Visitors flock to Paris landmark", "The Eiffel Tower, completed in 1889, remains...")]}
PAGES = {GOV: load_page("eiffel_official.txt"), BBC: load_page("eiffel_news.txt")}

SPOKEN = "The Eiffel Tower was completed in 1889."
Q = ["Eiffel Tower completed 1889"]


def year_judge(claim, passages, attempt, user):
    """SUPPORTED for 1889, REFUTED (with a grounded correction) for any other year."""
    wrong = "1889" not in claim.statement
    stance = "REFUTES" if wrong else "SUPPORTS"
    assessments = []
    for p in passages:
        if "completed in 1889" in p.text:
            quote = next(s for s in p.text.replace("\n", " ").split(". ") if "completed in 1889" in s)
            assessments.append({"evidence_id": p.evidence_id, "stance": stance, "directness": "EXPLICIT",
                                "quote": quote.strip(), "explanation": ""})
        else:
            assessments.append({"evidence_id": p.evidence_id, "stance": "IRRELEVANT", "directness": "INFERENTIAL",
                                "quote": p.text[:30], "explanation": ""})
    out = {"assessments": assessments, "proposed_verdict": "REFUTED" if wrong else "SUPPORTED", "rationale": ""}
    if wrong:
        out["correction"] = {"text": "The Eiffel Tower was completed in 1889.",
                             "evidence_ids": [p.evidence_id for p in passages]}
    return out


@pytest.fixture
def audio(tmp_path):
    return write_audio(tmp_path / "talk.mp3", tone(4.0))


def run(path, config, transcriber, llm, search=None, **kw) -> AudioReport:
    return verify_audio(path, config, llm, search or FakeSearch(CORPUS, PAGES),
                        make_transcriber=lambda: transcriber, **kw)


# ------------------------------------------------------------------ verdicts and traceability


def test_supported_claim_is_fully_traceable(audio, config):
    tr = FakeTranscriber([seg(1, 0.0, 1.5, "Hi everyone, welcome back."), seg(2, 1.6, 4.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    ar = run(audio, config, tr, llm)
    [r] = ar.fact_check.claims
    assert r.verdict == Verdict.SUPPORTED
    [t] = ar.traced_claims
    assert t.statement == SPOKEN and t.transcript_excerpt == SPOKEN and t.verdict == Verdict.SUPPORTED
    assert t.segment_ids == [2] and t.transcript == SPOKEN and t.times_spoken == 1
    assert 1.6 <= t.start < t.end <= 4.0 and t.transcription_confidence == 0.9
    assert t.supporting and t.supporting[0].url in (GOV, BBC) and t.contradicting == []
    text = render_audio_report(ar)
    assert "AUDIO FACT-CHECK" in text and "Spoken ~00:01.6-00:04.0 (word timing), segment 2" in text
    assert "Language: en (p=0.99), detected; supported" in text and "transcription confidence 0.90" in text


def test_wrong_date_is_refuted_with_correction(audio, config):
    tr = FakeTranscriber([seg(1, 0.0, 3.0, "The Eiffel Tower was completed in 1899.")])
    llm = FakeLLM(extraction(claim_dict("The Eiffel Tower was completed in 1899.", queries=Q)), year_judge)
    ar = run(audio, config, tr, llm)
    [r] = ar.fact_check.claims
    assert r.verdict == Verdict.REFUTED and "1889" in r.correction
    assert ar.traced_claims[0].contradicting and ar.fact_check.summary_label == SummaryLabel.CONTAINS_FALSE_CLAIMS


def test_repeated_claim_is_checked_once_with_every_place_it_was_said(audio, config):
    tr = FakeTranscriber([seg(1, 0.0, 2.0, SPOKEN), seg(2, 2.0, 2.5, "Remember that."), seg(3, 2.6, 4.0, SPOKEN)])
    # the model extracts each occurrence, as the audio note asks
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q), claim_dict(SPOKEN, queries=Q)), year_judge)
    ar = run(audio, config, tr, llm)
    assert len(ar.fact_check.claims) == 1 and len(llm.calls_for("judge")) == 1
    [src] = ar.claim_sources
    assert [s.segment_ids for s in src.spans] == [[1], [3]]
    assert ar.traced_claims[0].times_spoken == 2
    assert "Said 2 times in the audio" in render_audio_report(ar)


def test_numbers_dates_and_negations_are_never_merged(audio, config):
    neg = "The Great Wall of China is not visible from the Moon."
    pos = "The Great Wall of China is visible from the Moon."
    tr = FakeTranscriber([seg(1, 0.0, 1.0, SPOKEN), seg(2, 1.0, 2.0, "The Eiffel Tower was completed in 1887."),
                          seg(3, 2.0, 3.0, neg), seg(4, 3.0, 4.0, pos)])
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q),
                             claim_dict("The Eiffel Tower was completed in 1887.", queries=Q),
                             claim_dict(neg, checkability="VAGUE"), claim_dict(pos, checkability="VAGUE")),
                  year_judge)
    ar = run(audio, config, tr, llm)
    assert ar.duplicates_merged == 0
    statements = {r.statement for r in ar.fact_check.claims}
    assert {SPOKEN, "The Eiffel Tower was completed in 1887.", neg, pos} == statements
    verdicts = {r.statement: r.verdict for r in ar.fact_check.claims}
    assert verdicts[SPOKEN] == Verdict.SUPPORTED and verdicts["The Eiffel Tower was completed in 1887."] == Verdict.REFUTED


def test_prompt_has_the_audio_note_and_no_timestamps(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    ar = run(audio, config, tr, llm)
    prompt = llm.calls_for("extract_claims")[0]["user"]
    assert AUDIO_SOURCE_NOTE.splitlines()[0] in prompt and "promotional calls to action" in prompt
    assert ar.checked_text == SPOKEN and "00:" not in prompt


# ------------------------------------------------------------------ claim extraction safeguards


def test_opinions_greetings_and_calls_to_action_are_not_checked(audio, config):
    text = "Hi everyone! I think pineapple pizza is the best. Subscribe and use code PODCAST20."
    tr = FakeTranscriber([seg(1, 0.0, 4.0, text)])
    llm = FakeLLM(extraction(claim_dict("Pineapple pizza is the best.", "pineapple pizza is the best",
                                        checkability="OPINION"),
                             claim_dict("Listeners should subscribe.", "Subscribe and use code PODCAST20",
                                        checkability="OTHER_NOT_CHECKABLE")))
    search = FakeSearch()
    ar = run(audio, config, tr, llm, search)
    assert ar.fact_check.summary_label == SummaryLabel.NOTHING_TO_CHECK and search.search_calls == []


def test_invented_claim_is_dropped(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict("The Louvre opened in 1793.")))
    ar = run(audio, config, tr, llm)
    assert ar.fact_check.claims == [] and "does not occur in the input" in ar.fact_check.dropped_claims[0].reason


def test_claim_with_a_changed_number_is_dropped(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict("The Eiffel Tower was completed in 1887.", SPOKEN)))
    ar = run(audio, config, tr, llm)
    assert ar.fact_check.claims == [] and "figures that are not in the input" in ar.fact_check.dropped_claims[0].reason


# ------------------------------------------------------------------ transcription reliability


def test_low_confidence_claim_is_withheld_and_not_searched(audio, config):
    tr = FakeTranscriber([seg(1, 0.0, 2.0, "The Eiffel Tower was completed in 1899.", conf=0.2),
                          seg(2, 2.0, 4.0, SPOKEN, conf=0.9)])
    llm = FakeLLM(extraction(claim_dict("The Eiffel Tower was completed in 1899.", queries=["Eiffel 1899"]),
                             claim_dict(SPOKEN, queries=Q)), year_judge)
    search = FakeSearch(CORPUS, PAGES)
    ar = run(audio, config, tr, llm, search)
    by = {r.statement: r for r in ar.fact_check.claims}
    withheld = by["The Eiffel Tower was completed in 1899."]
    assert withheld.verdict == Verdict.PROCESSING_FAILED and withheld.failure_reason.startswith("transcription:")
    assert by[SPOKEN].verdict == Verdict.SUPPORTED
    assert all("1899" not in q for q in search.search_calls)
    assert ar.transcription.status == TranscriptionStatus.LOW_CONFIDENCE
    text = " ".join(render_audio_report(ar).split())  # ignore line wrapping
    assert "may have been misheard (low transcription confidence)" in text and "below the threshold" in text


def test_everything_unreliable_means_no_model_or_search_calls(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 1.8, "mumble mumble tower", conf=0.15)])
    llm, search = FakeLLM(), FakeSearch(CORPUS, PAGES)
    ar = run(audio, config, tr, llm, search)
    assert ar.fact_check is None and "below the transcription confidence" in ar.not_checked_reason
    assert llm.calls == [] and search.search_calls == []


def test_transcription_confidence_never_changes_the_evidence_score(audio, config):
    scores = []
    for conf in (0.95, 0.45):
        tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN, conf=conf)])
        ar = run(audio, config, tr, FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge))
        scores.append((ar.fact_check.claims[0].verdict, ar.fact_check.claims[0].score))
    assert scores[0] == scores[1]


# ------------------------------------------------------------------ no speech / language / failures


@pytest.mark.parametrize("samples", [silence(3.0), noise(3.0)], ids=["silence", "noise"])
def test_silence_or_noise_is_not_checked(tmp_path, config, samples):
    path = write_audio(tmp_path / "x.wav", samples)
    llm, search = FakeLLM(), FakeSearch()
    ar = run(path, config, FakeTranscriber([]), llm, search)
    assert ar.fact_check is None and ar.transcription.status == TranscriptionStatus.NO_SPEECH
    assert "no speech was recognised" in ar.not_checked_reason and llm.calls == [] and search.search_calls == []
    assert "NOT FACT-CHECKED" in render_audio_report(ar)


def test_non_english_audio_is_not_checked(audio, config):
    llm = FakeLLM()
    ar = run(audio, config, FakeTranscriber([seg(1, 0, 2, "yah ek missile hai")], language="hi"), llm)
    assert ar.transcription.status == TranscriptionStatus.UNSUPPORTED_LANGUAGE and ar.transcription.segments == []
    assert ar.fact_check is None and "only English is supported" in ar.not_checked_reason and llm.calls == []
    assert "NOT supported (English only)" in render_audio_report(ar)


def test_transcription_failure_is_reported(audio, config):
    ar = run(audio, config, FakeTranscriber([], error=TranscriptionFailed("decoder crashed")), FakeLLM())
    assert ar.transcription.status == TranscriptionStatus.FAILED and "decoder crashed" in ar.transcription.error
    assert ar.fact_check is None and "transcription failed" in ar.not_checked_reason


def test_corrupted_and_empty_files_raise_input_errors(tmp_path, config):
    bad = tmp_path / "bad.mp3"
    bad.write_text("not audio")
    with pytest.raises(AudioInputError):
        run(bad, config, FakeTranscriber([]), FakeLLM())
    empty = tmp_path / "empty.wav"
    import wave

    with wave.open(str(empty), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
    made = []
    with pytest.raises(AudioInputError, match="empty audio"):
        verify_audio(empty, config, FakeLLM(), FakeSearch(), make_transcriber=lambda: made.append(1))
    assert made == []  # no model is loaded for an unusable file


def test_temporary_wav_is_real_and_removed(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, "Hi there.")])
    run(audio, config, tr, FakeLLM(extraction(claim_dict("Hi there.", checkability="OTHER_NOT_CHECKABLE"))))
    [wav] = tr.wav_paths
    assert tr.wav_sizes[0] > 100_000  # ~4 s of 16 kHz 16-bit audio
    assert not wav.exists() and not wav.parent.exists()
    tr = FakeTranscriber([], error=TranscriptionFailed("boom"))
    run(audio, config, tr, FakeLLM())
    assert not tr.wav_paths[0].parent.exists()


def test_long_transcript_is_truncated_with_a_warning(audio, config):
    segs = [seg(i, i * 0.5, i * 0.5 + 0.4, f"Statement number {i} is about the tower and other things.")
            for i in range(1, 8)]
    ar = run(audio, config, FakeTranscriber(segs), FakeLLM(extraction()), max_text_chars=200)
    assert ar.omitted_segment_ids and len(ar.checked_text) <= 200
    assert any("were not checked" in w and "--max-text-chars" in w for w in ar.warnings)


def test_extraction_timeout_is_a_failed_transcription(audio, config, monkeypatch):
    import factcheck_audio.pipeline as pipeline
    from factcheck_audio.audio_io import ExtractionTimeout

    def slow(*a, **k):
        raise ExtractionTimeout("extraction time limit of 1 s reached during audio extraction")

    monkeypatch.setattr(pipeline, "decode_to_wav", slow)
    made = []
    ar = verify_audio(audio, config, FakeLLM(), FakeSearch(), make_transcriber=lambda: made.append(1),
                      limits=AudioLimits(extract_timeout_s=1))
    assert ar.transcription.status == TranscriptionStatus.FAILED and "time limit" in ar.transcription.error
    assert made == [] and ar.fact_check is None


# ------------------------------------------------------------------ API failures (shared engine behaviour)


def test_llm_auth_failure_fails_the_run(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    with pytest.raises(RunFailure, match="authentication"):
        run(audio, config, tr, FakeLLM(LLMAuthError("bad key")))


def test_search_auth_failure_fails_the_run(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    with pytest.raises(RunFailure, match="authentication"):
        run(audio, config, tr, FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge),
            FakeSearch(CORPUS, PAGES, auth_fail=True))


def test_judge_unavailable_marks_the_claim_processing_failed(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])

    def down(*a):
        return LLMUnavailableError("503 UNAVAILABLE: high demand")

    ar = run(audio, config, tr, FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), down))
    [r] = ar.fact_check.claims
    assert r.verdict == Verdict.PROCESSING_FAILED
    assert ar.traced_claims[0].failure_reason and ar.traced_claims[0].start is not None  # still traceable


def test_search_outage_is_retrieval_failed_not_a_crash(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    ar = run(audio, config, tr, llm, FakeSearch(CORPUS, PAGES, fail_queries=lambda q: True))
    assert ar.fact_check.claims[0].verdict == Verdict.RETRIEVAL_FAILED  # never a truth verdict
    assert llm.calls_for("judge") == [] and ar.traced_claims[0].segment_ids == [1]


# ------------------------------------------------------------------ JSON report


def test_json_report_is_complete_valid_and_round_trips(audio, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    ar = run(audio, config, tr, FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge))
    raw = render_audio_json(ar)
    data = json.loads(raw)
    assert data["input_kind"] == "audio" and data["media"]["codec"] and data["media"]["duration_s"]
    assert data["transcription"]["language"] == {"detected": "en", "probability": 0.99, "english_probability": 0.99,
                                                 "accepted": True, "method": "detected", "note": None}
    assert data["transcription"]["segments"][0]["start"] == 0.2
    t = data["traced_claims"][0]
    assert set(t) >= {"statement", "start", "end", "segment_ids", "transcript", "transcription_confidence",
                      "verdict", "evidence_strength", "supporting", "contradicting",
                      "citations_rejected_by_validation"}
    assert data["fact_check"]["claims"][0]["verdict"] == "SUPPORTED"
    assert AudioReport.model_validate(data).model_dump(mode="json") == data
    assert render_audio_json(AudioReport.model_validate(data)) == raw  # stable


def test_json_for_a_not_checked_result(tmp_path, config):
    path = write_audio(tmp_path / "q.flac", silence(1.0))
    data = json.loads(render_audio_json(run(path, config, FakeTranscriber([]), FakeLLM())))
    assert data["fact_check"] is None and data["traced_claims"] == [] and data["not_checked_reason"]
    assert data["transcription"]["status"] == "NO_SPEECH"


def test_extract_only_needs_no_providers(audio, config):
    llm = FakeLLM()
    ar = run(audio, config, FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)]), llm, fact_check=False)
    assert ar.fact_check is None and llm.calls == [] and ar.checked_text == SPOKEN
    assert "transcription only" in ar.not_checked_reason


# ------------------------------------------------------------------ CLI


@pytest.fixture
def fake_model(monkeypatch):
    import factcheck_audio.transcribe as transcribe

    made = {}

    def factory(*a, **k):
        made.update(k, model=a[0] if a else None)
        return FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])

    monkeypatch.setattr(transcribe, "EnglishTranscriber", factory)
    return made


def test_cli_extract_only_writes_json(audio, tmp_path, fake_model, capsys):
    out = tmp_path / "r" / "report.json"
    assert audio_cli.main([str(audio), "--extract-only", "--json-out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "AUDIO FACT-CHECK" in printed and SPOKEN in printed and "NOT FACT-CHECKED" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["checked_text"] == SPOKEN
    assert fake_model["model"] == "base" and fake_model["compute_type"] == "int8" and not fake_model["assume_english"]


def test_cli_passes_speech_options(audio, fake_model):
    assert audio_cli.main([str(audio), "--transcribe-only", "--whisper-model", "small", "--assume-english",
                           "--beam-size", "1", "--threads", "4", "--no-word-timestamps", "-q"]) == 0
    assert fake_model["model"] == "small" and fake_model["assume_english"] and fake_model["beam_size"] == 1
    assert fake_model["threads"] == 4 and fake_model["word_timestamps"] is False


def test_cli_fact_check_without_keys_is_a_config_error(audio, fake_model, capsys):
    assert audio_cli.main([str(audio)]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


@pytest.mark.parametrize("args,needle", [
    (["missing.mp3"], "not found"),
    (["x.mp3"], "corrupted"),
    (["clip.mp4"], "factcheck-video"),
    (["--speech-threshold", "1.5", "x.mp3"], "--speech-threshold"),
    (["--max-duration", "0", "x.mp3"], "invalid limit"),
    (["--beam-size", "0", "x.mp3"], "--beam-size"),
    ([], "an audio file is required"),
])
def test_cli_invalid_arguments(tmp_path, monkeypatch, capsys, args, needle):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.mp3").write_bytes(b"\0" * 10)
    (tmp_path / "clip.mp4").write_bytes(b"\0" * 10)
    assert audio_cli.main(args) == 2
    assert needle in capsys.readouterr().err


def test_cli_missing_model_is_reported_not_downloaded(audio, monkeypatch, capsys):
    import factcheck_audio.transcribe as transcribe

    def missing(*a, **k):
        raise ModelUnavailable("speech model 'base' is not downloaded. Re-run with --allow-download")

    monkeypatch.setattr(transcribe, "EnglishTranscriber", missing)
    assert audio_cli.main([str(audio), "--extract-only"]) == 2
    assert "--allow-download" in capsys.readouterr().err


@pytest.mark.parametrize("transcriber,code,needle", [
    (FakeTranscriber([]), 1, "no speech"),
    (FakeTranscriber([seg(1, 0, 2, "x")], language="hi"), 1, "only English"),
    (FakeTranscriber([], error=TranscriptionFailed("boom")), 3, "transcription failed"),
])
def test_cli_exit_codes_when_nothing_is_checked(audio, monkeypatch, capsys, transcriber, code, needle):
    import factcheck_audio.transcribe as transcribe

    monkeypatch.setattr(transcribe, "EnglishTranscriber", lambda *a, **k: transcriber)
    assert audio_cli.main([str(audio), "-q"]) == code
    out = capsys.readouterr().out
    assert out.startswith("NOT_FACT_CHECKED") and needle in out


def test_cli_help_and_check(capsys):
    with pytest.raises(SystemExit) as exc:
        audio_cli.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "factcheck-audio" in text and "--extract-only" in text and "--whisper-model" in text
    assert audio_cli.main(["--check"]) == 0
    assert "speech model" in capsys.readouterr().out


def test_module_separation():
    # Importing the audio command loads no heavy dependency, no OCR and nothing from the video pipeline;
    # the other three commands never import the audio package.
    code = ("import sys, factcheck_audio, factcheck_audio.cli, factcheck_audio.pipeline, factcheck_audio.report;"
            "import factcheck_audio.transcribe;"
            "print(sorted(m for m in ('av', 'faster_whisper', 'ctranslate2', 'rapidocr', 'onnxruntime', 'PIL',"
            " 'factcheck_ocr', 'factcheck_video.pipeline', 'factcheck_video.frames', 'factcheck_video.cli')"
            " if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
    code = ("import sys, factcheck.cli, factcheck_ocr.cli, factcheck.image, factcheck_video.cli, factcheck_video.pipeline;"
            "print([m for m in sys.modules if m.startswith('factcheck_audio')])")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
