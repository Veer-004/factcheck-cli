"""Gemini -> Ollama fallback: policy, graph behaviour, wiring, reporting (fully offline)."""

from __future__ import annotations

import json

import pytest
from conftest import load_page
from fakes import MALFORMED, FakeLLM, FakeSearch, claim_dict, extraction, result, rule_judge

from factcheck import cli, diagnostics
from factcheck import graph as graph_mod
from factcheck.config import Config, ConfigError
from factcheck.graph import RunFailure, verify_text
from factcheck.limits import CallBudget, Deadline
from factcheck.llm import (
    FallbackLLM,
    GeminiClient,
    LLMAuthError,
    LLMNetworkError,
    LLMOutputError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    is_recoverable,
)
from factcheck.models import LLMExtraction, Verdict
from factcheck.ollama import OllamaClient
from factcheck.report import render_json, render_text

GOV = "https://www.tourism.example.gov/eiffel"
BBC = "https://www.bbc.co.uk/news/eiffel"
CORPUS = {"eiffel": [result(GOV, "The Eiffel Tower - official history", "x"),
                     result(BBC, "Visitors flock to Paris landmark", "y")]}
PAGES = {GOV: load_page("eiffel_official.txt"), BBC: load_page("eiffel_news.txt")}
TEXT = "The Eiffel Tower was completed in 1889."
RULES = {"example.gov": ("SUPPORTS", "EXPLICIT", "completed in 1889"),
         "bbc.co.uk": ("SUPPORTS", "EXPLICIT", "completed in 1889")}


class Failing:
    """A primary that always raises the given error (and counts calls)."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def generate_structured(self, **kw):
        self.calls.append(kw["purpose"])
        raise self.exc


def chain(primary, fallback, budget=None, seconds=180):
    return FallbackLLM(primary, fallback, budget or CallBudget(14), Deadline(seconds),
                       primary_label="gemini:test", fallback_label="ollama:qwen3:4b")


def good_llm():
    return FakeLLM(extraction(claim_dict(TEXT, queries=["Eiffel 1889"])), rule_judge(RULES))


def _call(llm):
    return llm.generate_structured(system="s", user="u", schema=LLMExtraction, purpose="extract_claims")


# --------------------------------------------------------------------------- policy unit tests


@pytest.mark.parametrize("exc,expected", [
    (LLMRateLimitError("429", status_code=429), True),
    (LLMTimeoutError("read timeout"), True),
    (LLMNetworkError("connect"), True),
    (LLMUnavailableError("503", status_code=503), True),
    (LLMUnavailableError("no status"), True),
    (LLMUnavailableError("bad request", status_code=400), False),
    (LLMAuthError("bad key", status_code=403), False),
    (LLMOutputError("malformed"), False),
])
def test_recoverable_classification(exc, expected):
    assert is_recoverable(exc) is expected


def test_1_primary_success_never_calls_fallback():
    primary, fallback = good_llm(), good_llm()
    llm = chain(primary, fallback)
    _call(llm)
    _call(llm)
    assert len(primary.calls) == 2 and fallback.calls == [] and llm.fallback_calls == 0


def test_2_recoverable_failure_uses_fallback_then_skips_failed_primary():
    primary = Failing(LLMUnavailableError("503 UNAVAILABLE high demand", status_code=503))
    fallback = good_llm()
    budget = CallBudget(14)
    llm = chain(primary, fallback, budget)
    assert _call(llm).claims
    assert _call(llm).claims
    assert primary.calls == ["extract_claims"]  # marked down after the first failure
    assert len(fallback.calls) == 2 and llm.fallback_calls == 2
    assert budget.used == 1  # only the fallback attempt after a failed primary costs an extra unit
    assert "503" in llm.primary_down_reason


def test_3_both_providers_fail_with_a_clear_combined_error():
    llm = chain(Failing(LLMTimeoutError("Gemini sent no response within 50s")),
                Failing(LLMNetworkError("could not connect to the Ollama API. Is Ollama running?")))
    with pytest.raises(LLMNetworkError) as exc:
        _call(llm)
    msg = str(exc.value)
    assert "gemini:test failed (Gemini sent no response" in msg and "ollama:qwen3:4b fallback also failed" in msg
    assert "Is Ollama running?" in msg


@pytest.mark.parametrize("exc", [LLMAuthError("API key not valid", status_code=400),
                                 LLMUnavailableError("400 INVALID_ARGUMENT", status_code=400),
                                 LLMOutputError("malformed JSON")])
def test_6_non_recoverable_primary_errors_do_not_fall_back(exc):
    primary, fallback = Failing(exc), good_llm()
    with pytest.raises(type(exc)):
        _call(chain(primary, fallback))
    assert fallback.calls == []


def test_fallback_respects_call_budget_and_deadline():
    budget = CallBudget(1)
    budget.acquire()  # budget already spent by the logical call
    fallback = good_llm()
    with pytest.raises(LLMTimeoutError, match="budget exhausted, ollama:qwen3:4b fallback not attempted"):
        _call(chain(Failing(LLMTimeoutError("slow")), fallback, budget))
    assert fallback.calls == []
    fallback2 = good_llm()
    with pytest.raises(LLMTimeoutError, match="no run time left for the ollama:qwen3:4b fallback"):
        _call(chain(Failing(LLMTimeoutError("slow")), fallback2, seconds=1))
    assert fallback2.calls == []


@pytest.mark.parametrize("bad", [MALFORMED, json.dumps({"claims": [{"statement": 7}]})])
def test_4_5_fallback_malformed_or_schema_invalid_output_is_output_error(bad):
    llm = chain(Failing(LLMTimeoutError("slow")), FakeLLM(bad))
    with pytest.raises(LLMOutputError):
        _call(llm)


# --------------------------------------------------------------------------- through the graph


def test_graph_fallback_judges_retrieved_evidence(config):
    primary = Failing(LLMRateLimitError("Gemini rate limit (429)", status_code=429))
    fallback = good_llm()
    report = verify_text(TEXT, config, llm=chain(primary, fallback), search=FakeSearch(CORPUS, PAGES))
    [r] = report.claims
    assert r.verdict == Verdict.SUPPORTED and {c.url for c in r.supporting} == {GOV, BBC}
    assert [c["purpose"] for c in fallback.calls] == ["extract_claims", "judge"]
    assert primary.calls == ["extract_claims"]
    assert report.stats.fallback_model == "ollama:qwen3:4b" and report.stats.fallback_calls == 2
    assert any("handled 2 model call(s)" in w for w in report.warnings)
    out = render_text(report)
    assert "local fallback model ollama:qwen3:4b handled 2 model call(s)" in " ".join(out.split())
    data = json.loads(render_json(report))
    assert data["stats"]["fallback_model"] == "ollama:qwen3:4b" and data["stats"]["fallback_calls"] == 2


def test_7_retrieval_failure_is_not_replaced_by_model_knowledge(config):
    fallback = good_llm()
    search = FakeSearch(CORPUS, PAGES, fail_queries=lambda q: True)
    report = verify_text(TEXT, config, llm=chain(Failing(LLMTimeoutError("slow")), fallback), search=search)
    [r] = report.claims
    assert r.verdict == Verdict.RETRIEVAL_FAILED and not r.supporting and r.score is None
    assert [c["purpose"] for c in fallback.calls] == ["extract_claims"]  # never asked to judge without evidence


def test_7b_no_evidence_stays_insufficient_with_fallback(config):
    text = "The Zorblax festival was first held in 1742."
    search = FakeSearch({"zorblax": [result("https://x.example/a", "Bananas", "Bananas are rich in potassium.")]})
    fallback = FakeLLM(extraction(claim_dict(text, queries=["Zorblax 1742"])), rule_judge({}))
    [r] = verify_text(text, config, llm=chain(Failing(LLMTimeoutError("slow")), fallback), search=search).claims
    assert r.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert all(c["purpose"] != "judge" for c in fallback.calls)


def test_8_fallback_fabricated_citations_are_rejected(config):
    def lying(claim, passages, attempt, user):
        return {"proposed_verdict": "SUPPORTED", "rationale": "trust me", "assessments": [
            {"evidence_id": "C1-E42", "stance": "SUPPORTS", "directness": "EXPLICIT", "quote": TEXT},
            {"evidence_id": passages[0].evidence_id, "stance": "SUPPORTS", "directness": "EXPLICIT",
             "quote": "Records held by UNESCO prove the tower was finished in 1889."}]}

    fallback = FakeLLM(extraction(claim_dict(TEXT, queries=["Eiffel 1889"])), lying)
    [r] = verify_text(TEXT, config, llm=chain(Failing(LLMTimeoutError("slow")), fallback),
                      search=FakeSearch(CORPUS, PAGES)).claims
    assert r.verdict == Verdict.PROCESSING_FAILED and not r.supporting


def test_9_verdict_rules_unchanged_when_fallback_proposes_wrong_verdict(config):
    blog = "https://someblog.example/eiffel"
    search = FakeSearch({"eiffel": [result(blog, "Blog", "x")]}, {blog: load_page("eiffel_news.txt")})
    fallback = FakeLLM(extraction(claim_dict(TEXT, queries=["Eiffel 1889"])),
                       rule_judge({"someblog": ("SUPPORTS", "EXPLICIT", "completed in 1889")}, verdict="SUPPORTED"))
    [r] = verify_text(TEXT, config, llm=chain(Failing(LLMTimeoutError("slow")), fallback), search=search).claims
    assert r.verdict == Verdict.INSUFFICIENT_EVIDENCE and r.score == 40  # one weak source, as without fallback


def test_both_down_during_extraction_is_run_failure(config):
    llm = chain(Failing(LLMTimeoutError("Gemini sent no response")), Failing(LLMNetworkError("Is Ollama running?")))
    with pytest.raises(RunFailure, match="claim extraction failed: .*fallback also failed"):
        verify_text(TEXT, config, llm=llm, search=FakeSearch(CORPUS, PAGES))


def test_6b_gemini_auth_error_stops_run_without_fallback(config):
    fallback = good_llm()
    with pytest.raises(RunFailure, match="authentication"):
        verify_text(TEXT, config, llm=chain(Failing(LLMAuthError("API key not valid", status_code=400)), fallback),
                    search=FakeSearch(CORPUS, PAGES))
    assert fallback.calls == []


def test_10_limits_still_enforced_with_fallback(config):
    cfg = config.model_copy(update={"max_llm_calls": 2, "search_credit_budget": 2, "max_concurrency": 1})
    fallback = good_llm()
    budget_report = verify_text(TEXT, cfg, llm=FallbackLLM(Failing(LLMTimeoutError("slow")), fallback,
                                                           CallBudget(2), Deadline(180)),
                                search=FakeSearch(CORPUS, PAGES))
    assert budget_report.stats.llm_calls <= 2 and budget_report.stats.search_credits_used <= 2


def test_10b_shared_budget_counts_fallback_attempts(config):
    """With the budget shared as in verify_text, primary attempt + fallback attempt both count."""
    cfg = config.model_copy(update={"max_llm_calls": 3})
    budget = CallBudget(cfg.max_llm_calls)
    deadline = Deadline(cfg.run_timeout_s)
    llm = FallbackLLM(Failing(LLMTimeoutError("slow")), good_llm(), budget, deadline)
    _call(llm)
    assert budget.used == 1  # fallback attempt (the logical call is charged by BudgetedLLM)


# --------------------------------------------------------------------------- wiring and configuration


def test_config_defaults_and_env(monkeypatch):
    cfg = Config()
    assert (cfg.provider, cfg.ollama_fallback, cfg.ollama_url, cfg.ollama_model, cfg.ollama_timeout_s) == \
        ("gemini", True, "http://localhost:11434", "qwen3:4b", 90.0)
    monkeypatch.setenv("FACTCHECK_OLLAMA_FALLBACK", "false")
    monkeypatch.setenv("FACTCHECK_OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("FACTCHECK_OLLAMA_URL", "http://127.0.0.1:11500/")
    monkeypatch.setenv("FACTCHECK_OLLAMA_TIMEOUT", "45")
    cfg = Config.from_env()
    assert (cfg.ollama_fallback, cfg.ollama_model, cfg.ollama_url, cfg.ollama_timeout_s) == \
        (False, "qwen3:8b", "http://127.0.0.1:11500", 45.0)


@pytest.mark.parametrize("env,value", [("FACTCHECK_PROVIDER", "nvidia"), ("FACTCHECK_PROVIDER", "openai"),
                                       ("FACTCHECK_OLLAMA_URL", "ftp://x"), ("FACTCHECK_OLLAMA_MODEL", "../x"),
                                       ("FACTCHECK_OLLAMA_TIMEOUT", "0")])
def test_invalid_provider_settings_rejected(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ConfigError):
        Config.from_env()


def test_build_llm_default_is_gemini_with_ollama_fallback(monkeypatch, config):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFAKEKEYFORTESTSONLY000000000000")
    llm, label = graph_mod.build_llm(config, Deadline(180), CallBudget(14))
    assert isinstance(llm, FallbackLLM) and isinstance(llm.primary, GeminiClient)
    assert isinstance(llm.fallback, OllamaClient) and llm.fallback.model == "qwen3:4b"
    assert label == config.model


def test_build_llm_fallback_disabled_and_local_only(monkeypatch, config):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFAKEKEYFORTESTSONLY000000000000")
    llm, _ = graph_mod.build_llm(config.model_copy(update={"ollama_fallback": False}), Deadline(180), CallBudget(14))
    assert isinstance(llm, GeminiClient)
    monkeypatch.delenv("GEMINI_API_KEY")
    local, label = graph_mod.build_llm(config.model_copy(update={"provider": "ollama"}), Deadline(180), CallBudget(14))
    assert isinstance(local, OllamaClient) and label == "ollama:qwen3:4b"


def test_missing_gemini_key_is_config_error_not_silent_local_fallback(config):
    with pytest.raises(ConfigError, match="GEMINI_API_KEY is not set"):
        verify_text(TEXT, config, search=FakeSearch(CORPUS, PAGES))


# --------------------------------------------------------------------------- CLI and diagnostics


def test_11_cli_text_verbose_json_and_exit_codes_with_fallback(monkeypatch, tmp_path, capsys, config):
    def fake_verify(text, cfg):
        return verify_text(text, config, llm=chain(Failing(LLMTimeoutError("slow")), good_llm()),
                           search=FakeSearch(CORPUS, PAGES))

    monkeypatch.setattr(cli, "verify_text", fake_verify)
    out_file = tmp_path / "r.json"
    assert cli.main([TEXT, "--json-out", str(out_file)]) == 0
    assert "handled 2 model call(s)" in " ".join(capsys.readouterr().out.split())
    assert json.loads(out_file.read_text(encoding="utf-8"))["stats"]["fallback_calls"] == 2
    assert cli.main([TEXT, "-v"]) == 0
    assert "fallback ollama:qwen3:4b served 2 call(s)" in capsys.readouterr().out


def test_cli_help_and_check_config_mention_no_nvidia(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert cli.main(["--check-config"]) == 0
    out = capsys.readouterr().out
    assert "ollama_model: qwen3:4b" in out and "nvidia" not in out.lower()


def test_nvidia_diagnostic_is_gone():
    with pytest.raises(SystemExit) as exc:
        diagnostics.main(["nvidia"])
    assert exc.value.code == 2


def test_ollama_diagnostic_unreachable_server_is_reported():
    # 127.0.0.1:9 (discard port) is closed: a real, local-only connection refusal.
    r = diagnostics.diagnose_ollama("qwen3:4b", "http://127.0.0.1:9", timeout_s=5)
    assert not r.ok and r.error_type == "LLMNetworkError" and "ollama serve" in r.detail


def test_ollama_diagnostic_detects_missing_model(monkeypatch):
    import httpx

    from factcheck.llm import PhaseTimeoutClient

    monkeypatch.setattr(diagnostics.socket, "getaddrinfo",
                        lambda *a, **kw: [(diagnostics.socket.AF_INET, None, None, "", ("127.0.0.1", 11434))])

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.11"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:8b", "model": "llama3:8b"}]})
        raise AssertionError("chat must not be called when the model is missing")

    def factory(_k, m, *, timeout_s, max_retries, deadline, max_output_tokens):
        return OllamaClient(m, deadline=deadline, http_client=PhaseTimeoutClient(transport=httpx.MockTransport(handler)))

    r = diagnostics.diagnose_ollama("qwen3:4b", "http://localhost:11434", timeout_s=5, client_factory=factory)
    assert not r.ok and "ollama pull qwen3:4b" in r.detail
