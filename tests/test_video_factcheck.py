"""Video fact-checking end to end, fully offline.

Real generated video files are decoded by PyAV (probe, audio extraction to a temp
WAV, frame sampling). A scripted transcriber stands in for faster-whisper, a
colour-keyed fake engine for RapidOCR, and the fake LLM/search providers from
``fakes.py`` for Gemini and Tavily. Everything else is the real pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("av")
pytest.importorskip("PIL")

from conftest import load_page  # noqa: E402
from fakes import FakeLLM, FakeSearch, claim_dict, extraction, result  # noqa: E402
from video_helpers import BLUE, GREEN, RED, ColorEngine, FakeTranscriber, seg, silence, tone, write_video  # noqa: E402

from factcheck.graph import verify_text  # noqa: E402
from factcheck.models import SummaryLabel, Verdict  # noqa: E402
from factcheck_video import cli as video_cli  # noqa: E402
from factcheck_video.media import VideoInputError  # noqa: E402
from factcheck_video.models import Limits, Modality, StageStatus  # noqa: E402
from factcheck_video.pipeline import VIDEO_SOURCE_NOTE, VideoReport, verify_video  # noqa: E402
from factcheck_video.report import render_video_json, render_video_report  # noqa: E402
from factcheck_video.transcribe import ModelUnavailable, TranscriptionFailed  # noqa: E402

GOV = "https://www.tourism.example.gov/eiffel"
BBC = "https://www.bbc.co.uk/news/eiffel"
CORPUS = {"eiffel": [result(GOV, "The Eiffel Tower - official history", "The Eiffel Tower was completed in 1889."),
                     result(BBC, "Visitors flock to Paris landmark", "The Eiffel Tower, completed in 1889, remains...")]}
PAGES = {GOV: load_page("eiffel_official.txt"), BBC: load_page("eiffel_news.txt")}

SPOKEN = "The Eiffel Tower was completed in 1889."
SHOWN = "EIFFEL TOWER COMPLETED IN 1889"


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
def video(tmp_path):
    """3 s, audio track (tone), background red for 0-2 s and blue for 2-3 s."""
    return write_video(tmp_path / "clip.mp4", duration=4.0, audio=tone(4.0),
                       backgrounds=[(0, 2, RED), (2, 4, BLUE)])


def run(video, config, transcriber, engine, llm, search=None, **kw) -> VideoReport:
    return verify_video(video, config, llm, search or FakeSearch(CORPUS, PAGES),
                        make_transcriber=(lambda: transcriber) if transcriber else None,
                        make_engine=(lambda: engine) if engine else None, **kw)


# ------------------------------------------------------------------ de-duplication across modalities


def test_same_claim_spoken_and_shown_is_checked_once_with_both_sources(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    engine = ColorEngine({RED: [(SHOWN, 0.97)], BLUE: [(SHOWN, 0.97)]})
    stmt = "The Eiffel Tower was completed in 1889."
    llm = FakeLLM(extraction(claim_dict(stmt, SPOKEN, queries=["Eiffel Tower completed 1889"]),
                             claim_dict(stmt, SHOWN, queries=["Eiffel Tower completed 1889"])), year_judge)
    search = FakeSearch(CORPUS, PAGES)
    vr = run(video, config, tr, engine, llm, search)
    [r] = vr.fact_check.claims
    assert r.verdict == Verdict.SUPPORTED
    assert vr.duplicates_merged == 1 and vr.fact_check.dropped_claims == []
    [src] = vr.claim_sources
    assert set(src.modalities) == {Modality.SPEECH, Modality.SCREEN}
    assert src.merged[0].source_excerpt == SHOWN
    assert len(llm.calls_for("judge")) == 1  # verified once
    assert any("merged and checked once" in w for w in vr.fact_check.warnings)
    assert not any("dropped by validation" in w for w in vr.fact_check.warnings)


def test_different_numbers_in_speech_and_screen_are_both_checked(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    engine = ColorEngine({RED: [("EIFFEL TOWER COMPLETED IN 1887", 0.97)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=["Eiffel Tower completed"]),
                             claim_dict("The Eiffel Tower was completed in 1887.", "EIFFEL TOWER COMPLETED IN 1887",
                                        queries=["Eiffel Tower completed"])), year_judge)
    vr = run(video, config, tr, engine, llm)
    verdicts = {r.statement: r.verdict for r in vr.fact_check.claims}
    assert verdicts == {SPOKEN: Verdict.SUPPORTED, "The Eiffel Tower was completed in 1887.": Verdict.REFUTED}
    assert vr.duplicates_merged == 0
    assert vr.fact_check.summary_label == SummaryLabel.CONTAINS_FALSE_CLAIMS
    refuted = next(r for r in vr.fact_check.claims if r.verdict == Verdict.REFUTED)
    assert refuted.correction and "1889" in refuted.correction


def test_same_statement_written_for_differing_sources_is_not_merged(video, config):
    # The model (wrongly) wrote the 1889 statement for the 1887 caption too; the excerpt check keeps both,
    # and validation then drops the mismatched one for its figures, so nothing is silently merged away.
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    engine = ColorEngine({RED: [("EIFFEL TOWER COMPLETED IN 1887", 0.97)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=["Eiffel Tower completed"]),
                             claim_dict(SPOKEN, "EIFFEL TOWER COMPLETED IN 1887", queries=["Eiffel Tower completed"])),
                  year_judge)
    vr = run(video, config, tr, engine, llm)
    assert vr.duplicates_merged == 0
    assert len(vr.fact_check.claims) == 2


def test_negated_contradiction_is_kept_where_the_text_rule_would_merge(video, config):
    pos, neg = "The Great Wall of China is visible from the Moon.", "The Great Wall of China is not visible from the Moon."
    # Regression baseline: the text workflow's rule (unchanged) treats these as near-duplicates.
    text_llm = FakeLLM(extraction(claim_dict(pos, checkability="OPINION"), claim_dict(neg, checkability="OPINION")))
    rep = verify_text(f"{pos} {neg}", config, text_llm, FakeSearch())
    assert len(rep.claims) == 1 and "near-duplicate" in rep.dropped_claims[0].reason
    # Video: both are kept.
    tr = FakeTranscriber([seg(1, 0.2, 3.0, neg)])
    engine = ColorEngine({RED: [("The Great Wall of China is visible from the Moon", 0.95)]})
    llm = FakeLLM(extraction(claim_dict(neg, checkability="OPINION"),
                             claim_dict(pos, "The Great Wall of China is visible from the Moon", checkability="OPINION")))
    vr = run(video, config, tr, engine, llm)
    assert len(vr.fact_check.claims) == 2 and vr.duplicates_merged == 0


# ------------------------------------------------------------------ provenance and timestamps


def test_claim_provenance_has_modality_ids_and_times(video, config):
    tr = FakeTranscriber([seg(1, 0.4, 3.6, "Hello everyone. " + SPOKEN)])
    engine = ColorEngine({BLUE: [("Bats are blind", 0.93)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=["Eiffel Tower completed 1889"]),
                             claim_dict("Bats are blind.", "Bats are blind", checkability="VAGUE")), year_judge)
    vr = run(video, config, tr, engine, llm)
    speech_src, screen_src = vr.claim_sources
    [sp] = speech_src.spans
    assert sp.modality == Modality.SPEECH and sp.segment_id == 1 and sp.word_level
    assert 0.4 < sp.start < sp.end <= 3.6  # narrowed to the claim's words (after "Hello everyone.")
    [sc] = screen_src.spans
    assert sc.modality == Modality.SCREEN and sc.observation_id == 1
    assert sc.frame_times == [2.0] and sc.start == 2.0
    text = render_video_report(vr)
    assert "Spoken ~00:0" in text and "(word timing)" in text and "On screen: seen in the frame at 00:02.0" in text


def test_prompt_has_video_note_labels_and_no_timestamps(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    engine = ColorEngine({BLUE: [("Paris, France", 0.95)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=["Eiffel Tower completed 1889"])), year_judge)
    vr = run(video, config, tr, engine, llm)
    prompt = llm.calls_for("extract_claims")[0]["user"]
    assert VIDEO_SOURCE_NOTE.splitlines()[0] in prompt
    assert "[speech]\nThe Eiffel Tower was completed in 1889.\n\n[on-screen text]\nParis, France" in prompt
    assert vr.checked_text == "[speech]\nThe Eiffel Tower was completed in 1889.\n\n[on-screen text]\nParis, France"
    assert "00:" not in prompt and "0.2" not in vr.checked_text


# ------------------------------------------------------------------ extraction reliability gate


def test_low_confidence_speech_claim_is_withheld_and_not_searched(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 1.8, "The Eiffel Tower was completed in 1899.", conf=0.2),
                          seg(2, 2.0, 3.8, "Hello and welcome to the channel.", conf=0.9)])
    engine = ColorEngine({RED: [(SHOWN, 0.97)]})
    llm = FakeLLM(extraction(claim_dict("The Eiffel Tower was completed in 1899.", queries=["Eiffel 1899"]),
                             claim_dict("The Eiffel Tower was completed in 1889.", SHOWN, queries=["Eiffel 1889"])),
                  year_judge)
    search = FakeSearch(CORPUS, PAGES)
    vr = run(video, config, tr, engine, llm, search)
    by = {r.statement: r for r in vr.fact_check.claims}
    withheld = by["The Eiffel Tower was completed in 1899."]
    assert withheld.verdict == Verdict.PROCESSING_FAILED and withheld.failure_reason.startswith("extraction:")
    assert by["The Eiffel Tower was completed in 1889."].verdict == Verdict.SUPPORTED
    assert all("1899" not in q for q in search.search_calls)
    assert vr.extraction.audio.status == StageStatus.LOW_CONFIDENCE
    assert "misheard or misread" in render_video_report(vr)


def test_everything_unreliable_means_no_model_or_search_calls(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 1.8, "mumble mumble tower", conf=0.15)])
    engine = ColorEngine({RED: [("Th3 tovv3r", 0.4)]})
    llm, search = FakeLLM(), FakeSearch(CORPUS, PAGES)
    vr = run(video, config, tr, engine, llm, search, min_score=0.3)
    assert vr.fact_check is None and "below the recognition confidence" in vr.not_checked_reason
    assert llm.calls == [] and search.search_calls == []


# ------------------------------------------------------------------ empty / missing modalities


def test_silent_video_without_text_is_not_checked(tmp_path, config):
    path = write_video(tmp_path / "quiet.mp4", duration=2.0, audio=silence(2.0))
    llm, search = FakeLLM(), FakeSearch()
    vr = run(path, config, FakeTranscriber([]), ColorEngine({}), llm, search)
    assert vr.fact_check is None and llm.calls == [] and search.search_calls == []
    assert vr.extraction.audio.status == StageStatus.NO_CONTENT
    assert vr.extraction.screen.status == StageStatus.NO_CONTENT
    assert "speech: none found; on-screen text: none found" in vr.not_checked_reason
    assert "NOT FACT-CHECKED" in render_video_report(vr)


def test_video_without_audio_stream_uses_screen_text_only(tmp_path, config):
    path = write_video(tmp_path / "mute.mp4", duration=2.0, audio=None, backgrounds=[(0, 2, GREEN)])
    made = []

    def factory():
        made.append(1)
        return FakeTranscriber([])

    llm = FakeLLM(extraction(claim_dict(SPOKEN, "The Eiffel Tower was completed in 1889",
                                        queries=["Eiffel Tower completed 1889"])), year_judge)
    vr = verify_video(path, config, llm, FakeSearch(CORPUS, PAGES), make_transcriber=factory,
                      make_engine=lambda: ColorEngine({GREEN: [("The Eiffel Tower was completed in 1889", 0.96)]}))
    assert made == []  # no model loaded for a file without audio
    assert vr.extraction.audio.status == StageStatus.NO_STREAM
    assert vr.fact_check.claims[0].verdict == Verdict.SUPPORTED
    assert vr.claim_sources[0].modalities == [Modality.SCREEN]


def test_transcription_failure_keeps_screen_text(video, config):
    tr = FakeTranscriber([], error=TranscriptionFailed("decoder crashed"))
    engine = ColorEngine({RED: [("The Eiffel Tower was completed in 1889", 0.96)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, "The Eiffel Tower was completed in 1889",
                                        queries=["Eiffel Tower completed 1889"])), year_judge)
    vr = run(video, config, tr, engine, llm)
    assert vr.extraction.audio.status == StageStatus.FAILED and "decoder crashed" in vr.extraction.audio.error
    assert any("audio extraction failed" in w for w in vr.extraction.warnings)
    assert vr.fact_check.claims[0].verdict == Verdict.SUPPORTED


def test_temporary_audio_file_is_removed(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, "Hi there.")])
    llm = FakeLLM(extraction(claim_dict("Hi there.", checkability="OTHER_NOT_CHECKABLE")))
    run(video, config, tr, None, llm)
    [wav] = tr.wav_paths
    assert tr.wav_sizes[0] > 100_000  # ~4 s of 16 kHz 16-bit audio was really written
    assert not wav.exists() and not wav.parent.exists()


def test_temporary_audio_file_is_removed_on_failure(video, config):
    tr = FakeTranscriber([], error=TranscriptionFailed("boom"))
    run(video, config, tr, None, FakeLLM(), fact_check=False)
    assert not tr.wav_paths[0].parent.exists()


def test_extract_only_needs_no_providers(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM()
    vr = run(video, config, tr, ColorEngine({}), llm, fact_check=False)
    assert vr.fact_check is None and llm.calls == []
    assert vr.checked_text == "[speech]\n" + SPOKEN and "extraction only" in vr.not_checked_reason


def test_opinions_and_greetings_are_not_checked(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, "Hi everyone! I think pineapple pizza is the best.")])
    llm = FakeLLM(extraction(claim_dict("Pineapple pizza is the best.", "pineapple pizza is the best",
                                        checkability="OPINION")))
    search = FakeSearch()
    vr = run(video, config, tr, None, llm, search)
    assert vr.fact_check.summary_label == SummaryLabel.NOTHING_TO_CHECK and search.search_calls == []


def test_invented_claim_is_dropped(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    llm = FakeLLM(extraction(claim_dict("The Louvre opened in 1793.", "The Louvre opened in 1793.")))
    vr = run(video, config, tr, None, llm)
    assert vr.fact_check.claims == [] and "does not occur in the input" in vr.fact_check.dropped_claims[0].reason


# ------------------------------------------------------------------ limits and errors


def test_text_over_the_limit_is_truncated_with_a_warning(video, config):
    segs = [seg(i, i * 0.5, i * 0.5 + 0.4, f"Statement number {i} is about the tower and other things.") for i in range(1, 8)]
    tr = FakeTranscriber(segs)
    llm = FakeLLM(extraction())
    vr = run(video, config, tr, None, llm, max_text_chars=200)
    assert vr.omitted_units and len(vr.checked_text) <= 200
    assert any("were not checked" in w and "--max-text-chars" in w for w in vr.extraction.warnings)


def test_extraction_timeout_marks_stage_failed(video, config):
    tr = FakeTranscriber([])
    vr = run(video, config, tr, ColorEngine({}), FakeLLM(), limits=Limits(extract_timeout_s=0.0001), fact_check=False)
    statuses = {vr.extraction.audio.status, vr.extraction.screen.status}
    assert StageStatus.FAILED in statuses
    assert any("time limit" in w for w in vr.extraction.warnings)


def test_invalid_video_raises_input_error(tmp_path, config):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(VideoInputError):
        run(bad, config, FakeTranscriber([]), ColorEngine({}), FakeLLM())


# ------------------------------------------------------------------ reports


def test_json_report_is_valid_stable_and_unicode(video, config):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, "La tour Eiffel a été achevée en 1889 — « vrai ».")])
    llm = FakeLLM(extraction(claim_dict("La tour Eiffel a été achevée en 1889.", "La tour Eiffel a été achevée en 1889",
                                        checkability="VAGUE")))
    vr = run(video, config, tr, None, llm)
    raw = render_video_json(vr)
    data = json.loads(raw)
    assert "été" in raw and "«" in raw  # not escaped
    assert data["input_kind"] == "video" and data["extraction"]["media"]["has_audio"] is True
    assert data["extraction"]["audio"]["segments"][0]["start"] == 0.2
    assert data["claim_sources"][0]["spans"][0]["modality"] == "speech"
    assert VideoReport.model_validate(data).model_dump(mode="json") == data  # round-trips
    assert render_video_json(VideoReport.model_validate(data)) == raw  # stable


def test_json_for_empty_result(tmp_path, config):
    path = write_video(tmp_path / "quiet.mp4", duration=1.0, audio=None)
    vr = run(path, config, None, ColorEngine({}), FakeLLM())
    data = json.loads(render_video_json(vr))
    assert data["fact_check"] is None and data["claim_sources"] == [] and data["extraction"]["units"] == []
    assert data["extraction"]["audio"]["status"] == "SKIPPED"


# ------------------------------------------------------------------ CLI


@pytest.fixture
def fake_models(monkeypatch):
    import factcheck_ocr.engines as engines
    import factcheck_video.transcribe as transcribe

    monkeypatch.setattr(transcribe, "WhisperTranscriber",
                        lambda *a, **k: FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)]))
    monkeypatch.setattr(engines, "create_engine", lambda *a, **k: ColorEngine({RED: [("Paris, France", 0.95)]}))


def test_cli_extract_only_writes_json(video, tmp_path, fake_models, capsys):
    out = tmp_path / "r" / "report.json"
    code = video_cli.main([str(video), "--extract-only", "--json-out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "VIDEO FACT-CHECK" in printed and SPOKEN in printed and "Paris, France" in printed
    # the caption is visible from 0.0 s, the speech starts at 0.2 s: time order
    assert json.loads(out.read_text(encoding="utf-8"))["checked_text"] == (
        "[on-screen text]\nParis, France\n\n[speech]\n" + SPOKEN)


def test_cli_fact_check_without_keys_is_a_config_error(video, fake_models, capsys):
    assert video_cli.main([str(video)]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


@pytest.mark.parametrize("args,needle", [
    (["missing.mp4"], "not found"),
    (["--no-audio", "--no-ocr", "x.mp4"], "nothing to extract"),
    (["--ocr-threshold", "1.5", "x.mp4"], "--ocr-threshold"),
    (["--frame-interval", "0.01", "x.mp4"], "invalid limit"),
    ([], "a video file is required"),
])
def test_cli_invalid_arguments(tmp_path, monkeypatch, capsys, args, needle):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.mp4").write_bytes(b"\0" * 10)
    assert video_cli.main(args) == 2
    assert needle in capsys.readouterr().err


def test_cli_missing_model_is_reported_not_downloaded(video, monkeypatch, capsys):
    import factcheck_video.transcribe as transcribe

    def missing(*a, **k):
        raise ModelUnavailable("speech model 'base' is not downloaded. Re-run with --allow-download")

    monkeypatch.setattr(transcribe, "WhisperTranscriber", missing)
    assert video_cli.main([str(video), "--no-ocr", "--extract-only"]) == 2
    assert "--allow-download" in capsys.readouterr().err


def test_cli_nothing_found_exits_1(tmp_path, monkeypatch, capsys):
    import factcheck_video.transcribe as transcribe

    monkeypatch.setattr(transcribe, "WhisperTranscriber", lambda *a, **k: FakeTranscriber([]))
    path = write_video(tmp_path / "q.mp4", duration=1.0, audio=silence(1.0))
    assert video_cli.main([str(path), "--no-ocr", "-q"]) == 1
    assert capsys.readouterr().out.startswith("NOT_FACT_CHECKED")


def test_heavy_video_dependencies_are_not_imported_by_other_commands():
    code = ("import sys, factcheck.cli, factcheck_ocr.cli, factcheck.image, factcheck_video, factcheck_video.cli;"
            "print(sorted(m for m in ('av', 'faster_whisper', 'ctranslate2') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
