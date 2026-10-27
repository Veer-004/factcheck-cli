from __future__ import annotations

import json

import pytest

from factcheck import cli
from factcheck.config import ConfigError
from factcheck.graph import RunFailure
from factcheck.models import (
    Authority,
    Checkability,
    Citation,
    ClaimResult,
    ClaimType,
    Directness,
    DroppedClaim,
    Report,
    Stance,
    SummaryLabel,
    Verdict,
)
from factcheck.report import render_json, render_text
from factcheck.validation import InputError

DISCLAIMER_LINE = "Evidence strength: {n}/100 — experimental rubric; not a probability and not calibrated."


def _result(verdict: Verdict, **kw) -> ClaimResult:
    base = dict(claim_id="C1", index=1, statement="The Eiffel Tower was completed in 1889.",
                source_excerpt="The Eiffel Tower was completed in 1889.", claim_type=ClaimType.FACTUAL,
                checkability=Checkability.CHECKABLE, verdict=verdict)
    base.update(kw)
    return ClaimResult(**base)


def _cite(stance=Stance.SUPPORTS) -> Citation:
    return Citation(evidence_id="C1-E1", url="https://example.gov/eiffel", title="Official history",
                    published_date="2025-01-02", quote="The Eiffel Tower was completed in 1889.", stance=stance,
                    directness=Directness.EXPLICIT, authority=Authority.PRIMARY, cluster_id="C1-K1", weight=1.0)


def _report(*results: ClaimResult, **kw) -> Report:
    return Report(input_chars=40, summary_label=kw.pop("label", SummaryLabel.ALL_CHECKED_CLAIMS_SUPPORTED),
                  summary_text="summary", claims=list(results), **kw)


def flat(text: str) -> str:
    """Rendering wraps at 88 columns; compare content, not line breaks."""
    return " ".join(text.split())


def _cite2(eid="C1-E2", url="https://news.example.com/story", cluster="C1-K2", stance=Stance.SUPPORTS,
           quote="Construction finished in 1889, the paper reported.", title="News story", weight=0.75, **kw) -> Citation:
    return Citation(evidence_id=eid, url=url, title=title, quote=quote, stance=stance, directness=Directness.EXPLICIT,
                    authority=Authority.SECONDARY, cluster_id=cluster, weight=weight, **kw)


def test_normal_report_is_clean_and_complete():
    out = render_text(_report(_result(Verdict.SUPPORTED, score=70, supporting=[_cite()],
                                      score_caps=["capped at 70: only one independent source cluster"],
                                      notes=["model proposed REFUTED; deterministic rules selected SUPPORTED"],
                                      rejected_citations=2)))
    f = flat(out)
    assert out.startswith("FACT-CHECK REPORT")
    assert "OVERALL VERDICT: ALL CHECKED CLAIMS SUPPORTED" in out
    assert "Claim: The Eiffel Tower was completed in 1889." in f and "Verdict: SUPPORTED" in out
    assert "Explanation: The retrieved evidence supports this claim (1 independent source)." in f
    assert "Supporting evidence:" in out and "1. Official history (example.gov, 2025-01-02)" in f
    assert '"The Eiffel Tower was completed in 1889."' in f and "https://example.gov/eiffel" in out
    assert DISCLAIMER_LINE.format(n=70) in f
    assert "Limited by: only one independent source." in f
    # Implementation detail stays out of the normal view.
    for internal in ("[C1]", "C1-E1", "C1-K1", "cluster", "weight", "deterministic rules", "rejected"):
        assert internal not in out


def test_excerpt_shown_only_when_it_differs_from_the_claim():
    same = render_text(_report(_result(Verdict.SUPPORTED)))
    assert "From your text" not in same
    diff = render_text(_report(_result(Verdict.SUPPORTED, source_excerpt="It was finished in 1889")))
    assert 'From your text: "It was finished in 1889"' in flat(diff)


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_type_renders_with_its_name(verdict):
    out = render_text(_report(_result(verdict, failure_reason="every web retrieval attempt failed: boom"),
                              label=SummaryLabel.PARTIALLY_RESOLVED))
    assert f"Verdict: {verdict.value.replace('_', ' ')}" in out
    assert "Explanation:" in out


@pytest.mark.parametrize("verdict", [Verdict.RETRIEVAL_FAILED, Verdict.PROCESSING_FAILED, Verdict.NOT_CHECKABLE])
def test_failures_and_not_checkable_have_no_score(verdict):
    out = render_text(_report(_result(verdict, score=None), label=SummaryLabel.PARTIALLY_RESOLVED))
    assert "Evidence strength" not in out


def test_retrieval_failure_is_not_presented_as_a_judgement():
    out = flat(render_text(_report(_result(Verdict.RETRIEVAL_FAILED, failure_reason="every web retrieval attempt failed"),
                                   label=SummaryLabel.PARTIALLY_RESOLVED)))
    assert "technical problem, not a judgement about whether the claim is true" in out


@pytest.mark.parametrize("reason,plain", [
    ("search-credit budget exhausted before this claim was searched", "budget was used up"),
    ("run time limit reached before retrieval finished", "run time limit was reached"),
    ("judgment remained invalid after a repair attempt: x", "failed validation twice"),
    ("judge model unavailable: Gemini sent no response within 50s (ReadTimeout)", "did not respond in time"),
    ("unexpected error: KeyError: 'x'", "internal processing error"),
])
def test_processing_failure_reason_is_plain_in_normal_view_and_full_in_verbose(reason, plain):
    rep = _report(_result(Verdict.PROCESSING_FAILED, failure_reason=reason), label=SummaryLabel.PARTIALLY_RESOLVED)
    assert plain in flat(render_text(rep))
    assert reason in flat(render_text(rep, verbose=True))


def test_conflicting_shows_both_sides_without_a_winner():
    r = _result(Verdict.CONFLICTING, support_score=70, refute_score=55, supporting=[_cite()],
                refuting=[_cite2(stance=Stance.REFUTES)])
    f = flat(render_text(_report(r, label=SummaryLabel.PARTIALLY_RESOLVED)))
    assert "support 70/100, against 55/100" in f and "No single verdict is given" in f
    assert "1 independent source supports it and 1 contradicts it" in f
    assert "Supporting evidence:" in f and "Contradicting evidence:" in f


def test_insufficient_evidence_is_explained_honestly():
    some = flat(render_text(_report(_result(Verdict.INSUFFICIENT_EVIDENCE, score=40, supporting=[_cite()]),
                                    label=SummaryLabel.PARTIALLY_RESOLVED)))
    assert "not enough reliable evidence" in some and "not whether the claim is true" in some
    none = flat(render_text(_report(_result(Verdict.INSUFFICIENT_EVIDENCE, score=0), label=SummaryLabel.PARTIALLY_RESOLVED)))
    assert "No relevant evidence was found" in none


def test_correction_and_unavailable_correction_rendered():
    out = flat(render_text(_report(_result(Verdict.REFUTED, refuting=[_cite(Stance.REFUTES)],
                                           correction="It was completed in 1889.", correction_note="Grounded in C1-E1."))))
    assert "Correction: It was completed in 1889." in out and "Grounded in C1-E1" not in out
    out2 = flat(render_text(_report(_result(Verdict.REFUTED, correction_note="No correction could be established."))))
    assert "Correction: none could be established from the validated evidence." in out2
    assert "Correction" not in flat(render_text(_report(_result(Verdict.SUPPORTED))))


def test_same_source_page_listed_once_and_extra_sources_summarised():
    cites = [_cite(), _cite2(eid="C1-E5", url="https://example.gov/eiffel", cluster="C1-K1", title="Official history",
                             quote="A second passage from the same page about 1889.", weight=0.6)]
    cites += [_cite2(eid=f"C1-E{i}", url=f"https://site{i}.example/x", cluster=f"C1-K{i}", title=f"Site {i}") for i in (6, 7, 8)]
    r = _result(Verdict.SUPPORTED, supporting=cites)
    out = render_text(_report(r))
    assert out.count("https://example.gov/eiffel") == 1
    assert "A second passage from the same page" not in out  # one quote per source in the normal view
    assert "(+1 more source; see --verbose or --json-out)" in out
    verbose = render_text(_report(r), verbose=True)
    assert verbose.count("https://example.gov/eiffel") == 1 and "A second passage from the same page" in verbose
    assert "Site 8" in verbose and "C1-E5" in verbose


def test_long_quotes_shortened_in_normal_view_only():
    long_quote = "word " * 200 + "END"
    r = _result(Verdict.SUPPORTED, supporting=[_cite2(quote=long_quote.strip())])
    normal = flat(render_text(_report(r)))
    assert "END" not in normal and "…" in normal
    assert "END" in flat(render_text(_report(r), verbose=True))


def test_trust_flags_are_visible_to_users():
    r = _result(Verdict.SUPPORTED, supporting=[_cite2(snippet_only=True, injection_flagged=True, fact_check_site=True)])
    f = flat(render_text(_report(r)))
    assert "search snippet only" in f and "fact-checking site" in f and "instruction-like text" in f


def test_dropped_and_over_limit_disclosed():
    rep = _report(_result(Verdict.SUPPORTED), dropped_claims=[DroppedClaim(statement="X", reason="invented")],
                  over_limit_claims=[DroppedClaim(statement="Y", reason="over the limit")],
                  warnings=["1 extracted claim(s) dropped by validation", "run time limit of 180s was reached"])
    out = flat(render_text(rep))
    assert "Not checked: 1 claim over the per-run limit" in out and "- Y" in out
    assert "Skipped: 1 extracted statement failed validation and was not checked" in out
    assert "Warning: run time limit of 180s was reached." in out
    assert "invented" not in out  # reason only in verbose
    assert "dropped: X (invented)" in flat(render_text(rep, verbose=True))


def test_footer_reports_sources_and_time():
    rep = _report(_result(Verdict.SUPPORTED, sources_considered=4), _result(Verdict.SUPPORTED, claim_id="C2", index=2,
                                                                          sources_considered=5))
    rep.stats.elapsed_s = 12.34
    out = render_text(rep)
    assert "Sources retrieved: 9   Processing time: 12.3 s" in out
    assert "Claim 1 of 2:" in out and "Claim 2 of 2:" in out
    assert "2 claims: 2 supported" in out


def test_rendering_does_not_modify_the_report():
    rep = _report(_result(Verdict.CONFLICTING, support_score=70, refute_score=55, supporting=[_cite()],
                          refuting=[_cite2(stance=Stance.REFUTES)], score_caps=["capped at 70: only one independent source cluster"]))
    before = rep.model_dump(mode="json")
    render_text(rep)
    render_text(rep, verbose=True)
    assert rep.model_dump(mode="json") == before


def test_json_round_trip_is_complete_and_independent_of_rendering():
    rep = _report(_result(Verdict.SUPPORTED, score=70, supporting=[_cite()], notes=["n"], rejected_citations=1,
                          score_caps=["capped at 70: only one independent source cluster"]))
    data = json.loads(render_json(rep))
    assert Report.model_validate(data) == rep
    assert data["score_disclaimer"].startswith("experimental rubric")
    claim = data["claims"][0]
    for key in ("verdict", "supporting", "refuting", "context", "correction", "score", "score_caps", "strengths",
                "notes", "rejected_citations", "evidence", "model_proposed_verdict", "failure_reason"):
        assert key in claim
    assert claim["supporting"][0]["evidence_id"] == "C1-E1" and claim["supporting"][0]["cluster_id"] == "C1-K1"
    render_text(rep, verbose=True)
    assert json.loads(render_json(rep)) == data


# --------------------------------------------------------------------------- CLI


@pytest.mark.parametrize(
    "verdicts,over,code",
    [
        ([Verdict.SUPPORTED, Verdict.REFUTED, Verdict.NOT_CHECKABLE], False, 0),
        ([Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE], False, 1),
        ([Verdict.CONFLICTING], False, 1),
        ([Verdict.PROCESSING_FAILED], False, 1),
        ([Verdict.SUPPORTED], True, 1),
        ([], False, 0),
    ],
)
def test_exit_code_for(verdicts, over, code):
    rep = _report(*[_result(v, claim_id=f"C{i}") for i, v in enumerate(verdicts)],
                  over_limit_claims=[DroppedClaim(statement="Y", reason="r")] if over else [])
    assert cli.exit_code_for(rep) == code


def test_cli_missing_credentials_exit_2(capsys):
    assert cli.main(["The Eiffel Tower was completed in 1889."]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_cli_no_input_exit_2(capsys):
    assert cli.main([]) == 2


def test_cli_bad_config_exit_2(capsys):
    assert cli.main(["--max-claims", "99", "text here"]) == 2


def test_cli_conflicting_sources_rejected_by_argparse():
    with pytest.raises(SystemExit) as exc:
        cli.main(["some text", "--file", "x.txt"])
    assert exc.value.code == 2


def test_cli_check_config_never_prints_secret_values(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-should-not-appear")
    assert cli.main(["--check-config"]) == 0
    out = capsys.readouterr().out
    assert "secret-value-should-not-appear" not in out and "GEMINI_API_KEY: set" in out


def test_cli_verbose_flag_and_json_out(monkeypatch, tmp_path, capsys):
    rep = _report(_result(Verdict.SUPPORTED, score=70, supporting=[_cite()]))
    monkeypatch.setattr(cli, "verify_text", lambda text, config: rep)
    out_file = tmp_path / "report.json"
    assert cli.main(["The Eiffel Tower was completed in 1889.", "-v", "--json-out", str(out_file)]) == 0
    printed = capsys.readouterr().out
    assert "DIAGNOSTICS (verbose)" in printed and "C1-E1" in printed
    assert Report.model_validate(json.loads(out_file.read_text(encoding="utf-8"))) == rep
    assert cli.main(["The Eiffel Tower was completed in 1889."]) == 0
    assert "DIAGNOSTICS" not in capsys.readouterr().out


def test_cli_success_writes_json_and_returns_partial_code(monkeypatch, tmp_path, capsys):
    rep = _report(_result(Verdict.INSUFFICIENT_EVIDENCE, score=10), label=SummaryLabel.PARTIALLY_RESOLVED)
    monkeypatch.setattr(cli, "verify_text", lambda text, config: rep)
    out_file = tmp_path / "sub" / "report.json"
    assert cli.main(["The Eiffel Tower was completed in 1889.", "--json-out", str(out_file)]) == 1
    assert json.loads(out_file.read_text(encoding="utf-8"))["summary_label"] == "PARTIALLY_RESOLVED"
    assert "FACT-CHECK REPORT" in capsys.readouterr().out


def test_cli_reads_file(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "verify_text", lambda text, config: seen.setdefault("text", text) and _report())
    f = tmp_path / "in.txt"
    f.write_text("The Eiffel Tower was completed in 1889.", encoding="utf-8")
    assert cli.main(["--file", str(f), "-q"]) == 0
    assert seen["text"].startswith("The Eiffel Tower")


def test_cli_missing_file_exit_2():
    assert cli.main(["--file", "definitely/not/here.txt"]) == 2


@pytest.mark.parametrize("exc,code", [(InputError("bad"), 2), (ConfigError("bad"), 2), (RunFailure("boom"), 3), (RuntimeError("x"), 3)])
def test_cli_error_mapping(monkeypatch, exc, code):
    def boom(text, config):
        raise exc

    monkeypatch.setattr(cli, "verify_text", boom)
    assert cli.main(["The Eiffel Tower was completed in 1889."]) == code


@pytest.mark.parametrize("raw,shown", [("Mon, 09 Aug 2021 14:00:00 GMT", "2021-08-09"), ("2025-01-02", "2025-01-02"),
                                       ("2025-01-02T10:00:00Z", "2025-01-02")])
def test_publication_dates_are_displayed_as_iso_dates(raw, shown):
    c = _cite().model_copy(update={"published_date": raw})
    out = flat(render_text(_report(_result(Verdict.SUPPORTED, supporting=[c]))))
    assert f"(example.gov, {shown})" in out and "Mon, 09 Au" not in out


def test_unparseable_publication_date_is_omitted():
    c = _cite().model_copy(update={"published_date": "sometime last spring"})
    assert "(example.gov)" in flat(render_text(_report(_result(Verdict.SUPPORTED, supporting=[c]))))
