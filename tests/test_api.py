"""REST API (factcheck_api) tests, fully offline.

The app runs in-process (FastAPI TestClient). Gemini/Tavily are replaced by the
fake providers from ``fakes.py``; faster-whisper and RapidOCR by the scripted
fakes from the video/audio helpers. Uploaded files are real (generated PNG,
MP4 and MP3 files), and everything between the HTTP layer and the fakes is the
real code: upload handling, the four module functions, the shared engine and
the response conversion.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
pytest.importorskip("PIL")

from audio_helpers import FakeTranscriber as FakeAudioTranscriber  # noqa: E402
from audio_helpers import tone as audio_tone  # noqa: E402
from audio_helpers import write_audio  # noqa: E402
from conftest import AS_OF, load_page  # noqa: E402
from fakes import FakeLLM, FakeSearch, claim_dict, extraction, result  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from video_helpers import BLUE, RED, ColorEngine, FakeTranscriber, seg, tone, write_video  # noqa: E402

from factcheck.config import Config  # noqa: E402
from factcheck.llm import LLMAuthError  # noqa: E402
from factcheck_api.app import create_app  # noqa: E402
from factcheck_api.config import MB, Settings, SettingsError  # noqa: E402
from factcheck_api.services import Backend  # noqa: E402

GOV = "https://www.tourism.example.gov/eiffel"
BBC = "https://www.bbc.co.uk/news/eiffel"
CORPUS = {"eiffel": [result(GOV, "The Eiffel Tower - official history", "The Eiffel Tower was completed in 1889."),
                     result(BBC, "Visitors flock to Paris landmark", "The Eiffel Tower, completed in 1889, remains...")]}
PAGES = {GOV: load_page("eiffel_official.txt"), BBC: load_page("eiffel_news.txt")}
SPOKEN = "The Eiffel Tower was completed in 1889."
WRONG = "The Eiffel Tower was completed in 1899."
Q = ["Eiffel Tower completed 1889"]
TOP_KEYS = {"status", "input_type", "filename", "summary", "claims", "dropped_claims", "over_limit_claims",
            "not_checked_reason", "warnings", "notes", "processing_seconds", "run_plan", "report"}
CLAIM_KEYS = {"claim_id", "claim", "source_excerpt", "verdict", "evidence_score", "explanation", "citations",
              "citation_validation", "timestamps", "provenance", "correction", "failure_reason"}
ERROR_KEYS = {"status", "error", "request_id"}

judge_threads: list[str] = []


def year_judge(claim, passages, attempt, user):
    judge_threads.append(threading.current_thread().name)
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
    out = {"assessments": assessments, "proposed_verdict": "REFUTED" if wrong else "SUPPORTED", "rationale": "r"}
    if wrong:
        out["correction"] = {"text": "The Eiffel Tower was completed in 1889.",
                             "evidence_ids": [p.evidence_id for p in passages]}
    return out


def config_factory(**kw):
    return Config(as_of=AS_OF, **{k: v for k, v in kw.items() if v is not None})


@pytest.fixture
def uploads(tmp_path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def settings(uploads) -> Settings:
    return Settings(temp_dir=str(uploads), max_image_mb=2, max_video_mb=5, max_audio_mb=5)


def client_for(settings, **backend) -> TestClient:
    backend.setdefault("make_config", config_factory)
    return TestClient(create_app(settings, Backend(**backend)), raise_server_exceptions=False)


def png_bytes(colour=RED) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 120), colour).save(buf, format="PNG")
    return buf.getvalue()


def no_temp_files(uploads: Path) -> bool:
    return not uploads.exists() or not any(uploads.iterdir())


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == ERROR_KEYS and body["status"] == "error" and body["error"]["code"] == code
    assert body["request_id"] == r.headers["x-request-id"]
    return body["error"]["message"]


# ------------------------------------------------------------------ health / docs / generic errors


def test_health(settings):
    with client_for(settings) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and set(body["modules"]) == {"text", "image", "video", "audio"}
    assert all(m["available"] for m in body["modules"].values())
    assert body["fact_check_configured"] is False  # offline tests remove the keys
    assert "KEY" not in r.text


def test_openapi_documents_every_endpoint(settings):
    with client_for(settings) as c:
        spec = c.get("/openapi.json").json()
        assert c.get("/docs").status_code == 200
    assert set(spec["paths"]) == {"/health", "/api/v1/fact-check/text", "/api/v1/fact-check/image",
                                  "/api/v1/fact-check/video", "/api/v1/fact-check/audio", "/api/v1/plan",
                                  "/api/v1/plan/image", "/api/v1/plan/video", "/api/v1/plan/audio"}
    for path in ("text", "image", "video", "audio"):
        op = spec["paths"][f"/api/v1/fact-check/{path}"]["post"]
        assert op["description"] and op["summary"]
        assert "200" in op["responses"] and "422" in op["responses"] and "503" in op["responses"]
    assert "FactCheckResponse" in spec["components"]["schemas"] and "ErrorResponse" in spec["components"]["schemas"]


def test_unknown_route_and_method_use_the_error_shape(settings):
    with client_for(settings) as c:
        assert_error(c.get("/nope"), 404, "not_found")
        assert_error(c.get("/api/v1/fact-check/text"), 405, "method_not_allowed")


# ------------------------------------------------------------------ text


def test_text_supported_and_refuted_with_citations_preserved(settings):
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q), claim_dict(WRONG, queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES)) as c:
        r = c.post("/api/v1/fact-check/text", json={"text": f"{SPOKEN} {WRONG}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == TOP_KEYS and body["status"] == "completed" and body["input_type"] == "text"
    assert body["filename"] is None
    s = body["summary"]
    assert (s["total_claims"], s["supported"], s["refuted"], s["label"]) == (2, 1, 1, "CONTAINS_FALSE_CLAIMS")
    sup, ref = body["claims"]
    assert set(sup) >= CLAIM_KEYS and sup["verdict"] == "SUPPORTED" and ref["verdict"] == "REFUTED"
    assert sup["timestamps"] == [] and sup["provenance"] is None
    assert "1889" in ref["correction"]
    # citations: same URLs and verbatim quotes as the module report
    module = body["report"]["claims"][0]
    assert [(c["url"], c["quote"]) for c in sup["citations"]] == [(c["url"], c["quote"]) for c in module["supporting"]]
    assert all(c["role"] == "supporting" for c in sup["citations"]) and ref["citations"][0]["role"] == "refuting"
    assert sup["evidence_score"] == module["score"] and sup["explanation"].startswith("The retrieved evidence supports")
    assert body["report"]["summary_label"] == "CONTAINS_FALSE_CLAIMS" and body["report"]["claims"][0]["evidence"]
    assert judge_threads and "MainThread" not in judge_threads  # module work ran in the threadpool


def test_insufficient_evidence_is_a_successful_result(settings):
    llm = FakeLLM(extraction(claim_dict("The Zorblax treaty was signed in 1771.", queries=["Zorblax treaty"])),
                  year_judge)
    with client_for(settings, llm=llm, search=FakeSearch()) as c:
        r = c.post("/api/v1/fact-check/text", json={"text": "The Zorblax treaty was signed in 1771."})
    assert r.status_code == 200
    [claim] = r.json()["claims"]
    assert claim["verdict"] == "INSUFFICIENT_EVIDENCE" and r.json()["summary"]["insufficient_evidence"] == 1
    assert r.json()["status"] == "completed"


def test_text_options_reach_the_engine(settings):
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return config_factory(**kw)

    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES), make_config=factory) as c:
        r = c.post("/api/v1/fact-check/text", json={"text": SPOKEN, "max_claims": 2, "search_budget": 5})
    assert r.status_code == 200 and seen["max_checkable_claims"] == 2 and seen["search_credit_budget"] == 5


def test_a_run_refreshes_the_planner_credit_count(settings):
    class Credits:
        invalidated = 0

        def __call__(self):
            return None

        def invalidate(self):
            Credits.invalidated += 1

    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES), credits=Credits()) as c:
        assert c.post("/api/v1/fact-check/text", json={"text": SPOKEN}).status_code == 200
    assert Credits.invalidated == 1


@pytest.mark.parametrize("payload,status,code", [
    ({"text": ""}, 422, "validation_error"),
    ({}, 422, "validation_error"),
    ({"text": 5}, 422, "validation_error"),
    ({"text": "Paris is in France.", "max_claims": 99}, 422, "validation_error"),
    ({"text": "   "}, 400, "invalid_input"),
    ({"text": "x" * 4001}, 400, "invalid_input"),  # over the engine's own 4000-character limit
])
def test_text_request_validation(settings, payload, status, code):
    with client_for(settings, llm=FakeLLM(), search=FakeSearch()) as c:
        r = c.post("/api/v1/fact-check/text", json=payload)
    assert_error(r, status, code)
    if status == 422:
        assert all("input" not in d for d in r.json()["error"]["details"])  # submitted values are not echoed


def test_text_over_the_api_limit_is_413(uploads):
    with client_for(Settings(temp_dir=str(uploads), max_text_chars=50)) as c:
        assert_error(c.post("/api/v1/fact-check/text", json={"text": "a" * 51}), 413, "text_too_large")


def test_malformed_json_is_422(settings):
    with client_for(settings) as c:
        r = c.post("/api/v1/fact-check/text", content=b"{not json", headers={"content-type": "application/json"})
    assert_error(r, 422, "validation_error")


def test_missing_credentials_is_503_without_revealing_configuration(settings, monkeypatch):
    with client_for(settings, make_config=lambda **kw: Config.from_env(**kw)) as c:  # real providers, no keys
        r = c.post("/api/v1/fact-check/text", json={"text": SPOKEN})
    msg = assert_error(r, 503, "service_not_configured")
    assert "API_KEY" not in msg and "GEMINI" not in r.text and "TAVILY" not in r.text


def test_external_auth_failure_is_503(settings):
    with client_for(settings, llm=FakeLLM(LLMAuthError("API key AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE12 invalid")),
                    search=FakeSearch(CORPUS, PAGES)) as c:
        r = c.post("/api/v1/fact-check/text", json={"text": SPOKEN})
    assert_error(r, 503, "external_service_error")
    assert "AIza" not in r.text


def test_unexpected_exception_is_a_generic_500(settings):
    def boom(**kw):
        raise RuntimeError(r"secret path C:\Users\someone\private\thing.txt and key sk-abcdefghijklmnopqrstuv")

    with client_for(settings, make_config=boom) as c:
        r = c.post("/api/v1/fact-check/text", json={"text": SPOKEN})
    msg = assert_error(r, 500, "internal_error")
    assert msg == "An unexpected internal error occurred." and "Traceback" not in r.text and "Users" not in r.text


# ------------------------------------------------------------------ image


def test_image_fact_check_with_ocr_provenance_and_cleanup(settings, uploads):
    engine = ColorEngine({RED: [(SPOKEN, 0.97)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES), make_ocr_engine=lambda: engine) as c:
        r = c.post("/api/v1/fact-check/image", files={"file": ("../../etc/meme.png", png_bytes(), "image/png")},
                   data={"max_claims": "3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == TOP_KEYS and body["input_type"] == "image" and body["filename"] == "meme.png"
    [claim] = body["claims"]
    assert claim["verdict"] == "SUPPORTED" and claim["timestamps"] == []
    prov = claim["provenance"]
    assert prov["modality"] == "image" and prov["line_numbers"] == [1] and prov["ocr_reliable"] and prov["min_ocr_score"]
    assert body["report"]["extraction"]["image"] == "meme.png" and body["report"]["extraction"]["text"] == SPOKEN
    assert "factcheck-api-" not in r.text and str(uploads) not in json.dumps(body)
    assert no_temp_files(uploads)


def test_image_without_text_is_not_checked(settings, uploads):
    llm = FakeLLM()
    with client_for(settings, llm=llm, make_ocr_engine=lambda: ColorEngine({})) as c:
        r = c.post("/api/v1/fact-check/image", files={"file": ("blank.png", png_bytes(BLUE), "image/png")})
    assert r.status_code == 200 and r.json()["status"] == "not_checked" and r.json()["not_checked_reason"]
    assert r.json()["claims"] == [] and r.json()["summary"]["total_claims"] == 0 and llm.calls == []
    assert no_temp_files(uploads)


@pytest.mark.parametrize("name,ctype,content,status,code", [
    ("notes.txt", "text/plain", b"hello", 415, "unsupported_media_type"),
    ("photo.png", "text/plain", b"x" * 50, 415, "unsupported_media_type"),
    ("photo.png", "image/png", b"", 400, "empty_file"),
    ("photo.png", "image/png", b"not really a png" * 20, 400, "invalid_input"),
    ("photo.png", "image/png", b"\x89PNG" + b"\0" * (3 * MB), 413, "file_too_large"),
], ids=["wrong-extension", "wrong-content-type", "empty", "corrupted", "too-large"])
def test_image_upload_errors_and_cleanup(settings, uploads, name, ctype, content, status, code):
    with client_for(settings, make_ocr_engine=lambda: ColorEngine({})) as c:
        r = c.post("/api/v1/fact-check/image", files={"file": (name, content, ctype)})
    msg = assert_error(r, status, code)
    assert str(uploads) not in msg and "factcheck-api-" not in msg and "upload.png" not in msg
    if code == "invalid_input":
        assert "photo.png" in msg  # the client's own file name instead of the temporary one
    assert no_temp_files(uploads)


def test_missing_file_is_422(settings):
    with client_for(settings) as c:
        assert_error(c.post("/api/v1/fact-check/image", data={"max_claims": "2"}), 422, "validation_error")
        assert_error(c.post("/api/v1/fact-check/audio"), 422, "validation_error")


def test_temp_files_are_removed_when_processing_raises(settings, uploads):
    def crash():
        raise RuntimeError("engine exploded")

    with client_for(settings, make_ocr_engine=crash) as c:
        r = c.post("/api/v1/fact-check/image", files={"file": ("a.png", png_bytes(), "image/png")})
    assert_error(r, 500, "internal_error")
    assert no_temp_files(uploads)


# ------------------------------------------------------------------ video


@pytest.fixture
def video_bytes(tmp_path) -> bytes:
    path = write_video(tmp_path / "clip.mp4", duration=4.0, audio=tone(4.0), backgrounds=[(0, 2, RED), (2, 4, BLUE)])
    return path.read_bytes()


def test_video_preserves_speech_and_screen_timestamps(settings, uploads, video_bytes):
    tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    engine = ColorEngine({RED: [("EIFFEL TOWER COMPLETED IN 1889", 0.97)], BLUE: [("EIFFEL TOWER COMPLETED IN 1889", 0.97)]})
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q),
                             claim_dict(SPOKEN, "EIFFEL TOWER COMPLETED IN 1889", queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES), make_video_transcriber=lambda: tr,
                    make_ocr_engine=lambda: engine) as c:
        r = c.post("/api/v1/fact-check/video", files={"file": ("talk.mp4", video_bytes, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == TOP_KEYS and body["input_type"] == "video" and body["filename"] == "talk.mp4"
    [claim] = body["claims"]  # spoken and shown: checked once
    modalities = {t["modality"] for t in claim["timestamps"]}
    assert modalities == {"speech", "screen"}
    speech = next(t for t in claim["timestamps"] if t["modality"] == "speech")
    screen = next(t for t in claim["timestamps"] if t["modality"] == "screen")
    assert speech["segment_ids"] == [1] and speech["word_level"] and 0.2 <= speech["start"] < speech["end"] <= 3.0
    assert screen["observation_id"] and screen["frame_times"] and screen["confidence"] == pytest.approx(0.97)
    assert claim["provenance"]["modality"] == "video" and len(claim["provenance"]["merged"]) == 1
    ex = body["report"]["extraction"]
    assert ex["media"]["path"] == "talk.mp4" and ex["audio"]["segments"][0]["text"] == SPOKEN
    assert ex["screen"]["observations"] and body["report"]["checked_text"]
    assert str(uploads) not in r.text and "factcheck-api-" not in r.text
    assert no_temp_files(uploads)
    assert not tr.wav_paths[0].exists()  # the module's own temporary WAV is gone too


def test_corrupted_video_is_400(settings, uploads):
    with client_for(settings, make_video_transcriber=lambda: FakeTranscriber([]),
                    make_ocr_engine=lambda: ColorEngine({})) as c:
        r = c.post("/api/v1/fact-check/video", files={"file": ("bad.mp4", b"not a video" * 100, "video/mp4")})
    msg = assert_error(r, 400, "invalid_input")
    assert "upload.mp4" not in msg and str(uploads) not in msg
    assert no_temp_files(uploads)


def test_video_upload_wrong_type_is_415(settings):
    with client_for(settings) as c:
        r = c.post("/api/v1/fact-check/video", files={"file": ("song.mp3", b"x" * 100, "audio/mpeg")})
    assert_error(r, 415, "unsupported_media_type")


def test_oversized_video_rejected_before_processing(settings, uploads):
    made = []
    with client_for(settings, make_video_transcriber=lambda: made.append(1)) as c:
        r = c.post("/api/v1/fact-check/video", files={"file": ("big.mp4", b"\0" * (6 * MB), "video/mp4")})
    assert_error(r, 413, "file_too_large")
    assert made == [] and no_temp_files(uploads)


# ------------------------------------------------------------------ audio


@pytest.fixture
def audio_bytes(tmp_path) -> bytes:
    return write_audio(tmp_path / "talk.mp3", audio_tone(4.0)).read_bytes()


def test_audio_preserves_transcript_timestamps_and_confidence(settings, uploads, audio_bytes):
    tr = FakeAudioTranscriber([seg(1, 0.0, 1.5, "Hi everyone."), seg(2, 1.6, 4.0, SPOKEN, conf=0.83)])
    llm = FakeLLM(extraction(claim_dict(SPOKEN, queries=Q)), year_judge)
    with client_for(settings, llm=llm, search=FakeSearch(CORPUS, PAGES), make_audio_transcriber=lambda: tr) as c:
        r = c.post("/api/v1/fact-check/audio", files={"file": ("podcast.mp3", audio_bytes, "audio/mpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == TOP_KEYS and body["input_type"] == "audio" and body["filename"] == "podcast.mp3"
    [claim] = body["claims"]
    [span] = claim["timestamps"]
    assert span["modality"] == "speech" and span["segment_ids"] == [2] and span["word_level"]
    assert span["confidence"] == 0.83 and span["text"] == SPOKEN and 1.6 <= span["start"] < span["end"] <= 4.0
    assert claim["citations"] and claim["provenance"]["modality"] == "audio"
    rep = body["report"]
    assert rep["media"]["path"] == "podcast.mp3" and rep["transcription"]["language"]["detected"] == "en"
    assert [s["segment_id"] for s in rep["transcription"]["segments"]] == [1, 2]
    assert rep["traced_claims"][0]["transcription_confidence"] == 0.83
    assert any("not evidence" in n for n in body["notes"])  # confidence is not factual confidence
    assert str(uploads) not in r.text and no_temp_files(uploads)


def test_non_english_audio_is_not_checked(settings, uploads, audio_bytes):
    llm = FakeLLM()
    tr = FakeAudioTranscriber([seg(1, 0, 2, "yah ek missile hai")], language="hi")
    with client_for(settings, llm=llm, make_audio_transcriber=lambda: tr) as c:
        r = c.post("/api/v1/fact-check/audio", files={"file": ("hindi.m4a", audio_bytes, "audio/mp4")})
    assert r.status_code == 200 and r.json()["status"] == "not_checked"
    assert "only English" in r.json()["not_checked_reason"] and llm.calls == []
    assert no_temp_files(uploads)


def test_transcription_failure_is_500_and_cleaned_up(settings, uploads, audio_bytes):
    from factcheck_audio.transcribe import TranscriptionFailed

    tr = FakeAudioTranscriber([], error=TranscriptionFailed("decoder crashed"))
    with client_for(settings, make_audio_transcriber=lambda: tr) as c:
        r = c.post("/api/v1/fact-check/audio", files={"file": ("x.mp3", audio_bytes, "audio/mpeg")})
    assert "decoder crashed" in assert_error(r, 500, "processing_failed")
    assert no_temp_files(uploads)


def test_missing_speech_model_is_503(settings, audio_bytes):
    from factcheck_audio.transcribe import ModelUnavailable

    def missing():
        raise ModelUnavailable(r"speech model 'base' is not downloaded under C:\Users\x\.cache")

    with client_for(settings, make_audio_transcriber=missing) as c:
        r = c.post("/api/v1/fact-check/audio", files={"file": ("x.wav", audio_bytes, "audio/wav")})
    assert_error(r, 503, "dependency_unavailable")
    assert ".cache" not in r.text


def test_corrupted_and_mislabelled_audio(settings, uploads):
    with client_for(settings, make_audio_transcriber=lambda: FakeAudioTranscriber([])) as c:
        bad = c.post("/api/v1/fact-check/audio", files={"file": ("x.mp3", b"garbage" * 100, "audio/mpeg")})
        video = c.post("/api/v1/fact-check/audio", files={"file": ("x.mp4", b"\0" * 100, "video/mp4")})
        ctype = c.post("/api/v1/fact-check/audio", files={"file": ("x.mp3", b"\0" * 100, "image/png")})
    assert_error(bad, 400, "invalid_input")
    assert_error(video, 415, "unsupported_media_type")
    assert_error(ctype, 415, "unsupported_media_type")
    assert no_temp_files(uploads)


# ------------------------------------------------------------------ CORS and settings


def test_cors_allows_configured_origins_only_and_never_credentials(uploads):
    s = Settings(temp_dir=str(uploads), cors_origins=["https://app.example.com"])
    with client_for(s) as c:
        ok = c.options("/api/v1/fact-check/text", headers={"Origin": "https://app.example.com",
                                                           "Access-Control-Request-Method": "POST",
                                                           "Access-Control-Request-Headers": "content-type"})
        bad = c.options("/api/v1/fact-check/text", headers={"Origin": "https://evil.example",
                                                            "Access-Control-Request-Method": "POST"})
        simple = c.get("/health", headers={"Origin": "https://app.example.com"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-credentials" not in ok.headers
    assert "access-control-allow-origin" not in bad.headers
    assert simple.headers["access-control-allow-origin"] == "https://app.example.com"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FACTCHECK_API_CORS_ORIGINS", "https://a.example, https://b.example/")
    monkeypatch.setenv("FACTCHECK_API_MAX_VIDEO_MB", "100")
    s = Settings.from_env()
    assert s.cors_origins == ["https://a.example", "https://b.example"] and s.max_bytes("video") == 100 * MB
    monkeypatch.setenv("FACTCHECK_API_CORS_ORIGINS", "*")
    with pytest.raises(SettingsError, match="FACTCHECK_API_CORS_ORIGINS"):
        Settings.from_env()
    monkeypatch.setenv("FACTCHECK_API_CORS_ORIGINS", "https://a.example")
    monkeypatch.setenv("FACTCHECK_API_MAX_IMAGE_MB", "99")  # above the OCR module's own 25 MB limit
    with pytest.raises(SettingsError, match="FACTCHECK_API_MAX_IMAGE_MB"):
        Settings.from_env()


# ------------------------------------------------------------------ upload layer units


def test_display_name_never_becomes_a_path():
    from factcheck_api.uploads import display_name

    assert display_name("../../etc/passwd.png") == "passwd.png"
    assert display_name(r"C:\Windows\system32\evil.mp3") == "evil.mp3"
    assert display_name("a\x00b\x1f.mp3") == "ab.mp3"
    assert display_name("") == "upload" and display_name(None) == "upload"


@pytest.mark.anyio
async def test_streaming_copy_stops_at_the_limit_when_size_is_unknown(tmp_path):
    from fastapi import UploadFile

    from factcheck_api.errors import APIError
    from factcheck_api.uploads import stored_upload

    s = Settings(temp_dir=str(tmp_path / "u"), max_audio_mb=1)
    f = UploadFile(io.BytesIO(b"\0" * (2 * MB)), filename="x.mp3", size=None,
                   headers={"content-type": "audio/mpeg"})  # e.g. a chunked upload
    with pytest.raises(APIError) as exc:
        async with stored_upload(f, "audio", s):
            pass
    assert exc.value.status_code == 413 and not any((tmp_path / "u").iterdir())


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_cli_commands_do_not_load_the_api():
    import subprocess
    import sys

    code = ("import sys, factcheck.cli, factcheck_ocr.cli, factcheck_video.cli, factcheck_audio.cli;"
            "print(sorted(m for m in ('fastapi', 'uvicorn', 'starlette', 'factcheck_api') if m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


def test_no_persistent_infrastructure_is_used():
    # The API keeps no state between requests: no database, cache or queue libraries are imported.
    import subprocess
    import sys

    code = ("import sys, factcheck_api.app;"
            "print(sorted(m for m in ('sqlite3', 'redis', 'celery', 'sqlalchemy', 'pymongo', 'psycopg2')"
            " if m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


# ------------------------------------------------------------------ "Calculate with AI" for files


class RecordingPlanner:
    """Stands in for the AI planning agent: records the content it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, mode, text, duration_s=None, credits=None):
        from factcheck_api.planner import rule_plan

        self.seen.append((mode, text, duration_s))
        return rule_plan("text", text=text, credits=credits).model_copy(update={"mode": mode, "planner": "agent"})


def test_ai_plan_for_an_image_reads_it_with_ocr_first(settings, uploads):
    agent = RecordingPlanner()
    engine = ColorEngine({RED: [(SPOKEN, 0.97)]})
    with client_for(settings, make_ocr_engine=lambda: engine, planner=agent, credits=lambda: None) as c:
        r = c.post("/api/v1/plan/image", files={"file": ("meme.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["planner"] == "agent" and r.json()["mode"] == "image"
    assert agent.seen == [("image", SPOKEN, None)] and no_temp_files(uploads)


def test_ai_plan_for_audio_and_video_reads_the_transcript(settings, uploads, audio_bytes, video_bytes):
    agent = RecordingPlanner()
    audio_tr = FakeAudioTranscriber([seg(1, 0.0, 3.5, SPOKEN)])
    video_tr = FakeTranscriber([seg(1, 0.2, 3.0, SPOKEN)])
    with client_for(settings, planner=agent, credits=lambda: None, make_audio_transcriber=lambda: audio_tr,
                    make_video_transcriber=lambda: video_tr, make_ocr_engine=lambda: ColorEngine({})) as c:
        a = c.post("/api/v1/plan/audio", files={"file": ("talk.mp3", audio_bytes, "audio/mpeg")})
        v = c.post("/api/v1/plan/video", files={"file": ("talk.mp4", video_bytes, "video/mp4")})
    assert a.status_code == 200 and v.status_code == 200, (a.text, v.text)
    (am, at, ad), (vm, vt, vd) = agent.seen
    assert (am, vm) == ("audio", "video") and SPOKEN in at and SPOKEN in vt
    assert ad and vd and 3.5 < ad < 4.5 and 3.5 < vd < 4.5  # media length is passed on
    assert no_temp_files(uploads)


def test_ai_plan_for_a_file_without_text_needs_no_model_call(settings, uploads):
    agent = RecordingPlanner()
    with client_for(settings, make_ocr_engine=lambda: ColorEngine({}), planner=agent, credits=lambda: None) as c:
        r = c.post("/api/v1/plan/image", files={"file": ("blank.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and agent.seen == []
    assert "nothing to check" in r.json()["reasons"][0]


def test_ai_plan_upload_errors(settings, uploads):
    with client_for(settings, planner=RecordingPlanner(), credits=lambda: None) as c:
        assert_error(c.post("/api/v1/plan/image", files={"file": ("x.exe", b"MZ", "application/octet-stream")}),
                     415, "unsupported_media_type")
        assert_error(c.post("/api/v1/plan/audio"), 422, "validation_error")
    assert no_temp_files(uploads)
