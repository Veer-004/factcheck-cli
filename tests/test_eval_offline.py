"""Runs every deterministic offline robustness case through the real graph."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = run_eval
_spec.loader.exec_module(run_eval)

from factcheck import Config, verify_text  # noqa: E402

CASES = run_eval.load_cases("offline")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_offline_case(case):
    llm, search = run_eval.offline_providers(case)
    report = verify_text(case["text"], Config(as_of=run_eval.OFFLINE_AS_OF), llm=llm, search=search)
    row = run_eval.evaluate_case(case, report)
    assert row["strict_match"], row
    assert not row["extra_checks_failed"], row
    assert row["citations_valid"] == row["citations_total"]
    assert row["corrections_grounded"] == row["corrections_provided"]


def test_dev_and_heldout_are_disjoint():
    dev = {c["text"] for c in run_eval.load_cases("dev")}
    held = {c["text"] for c in run_eval.load_cases("heldout")}
    assert dev and held and not (dev & held)
    for c in run_eval.load_cases("dev") + run_eval.load_cases("heldout"):
        assert c["expected"] and c["category"]


def test_max_concurrency_bounds_parallel_claim_workers():
    from fakes import FakeLLM, FakeSearch, claim_dict, extraction, rule_judge

    active = peak = 0
    lock = threading.Lock()

    class SlowSearch(FakeSearch):
        def search(self, query, max_results):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return super().search(query, max_results)

    claims = [claim_dict(f"Item {i} weighs {i + 10} grams.", queries=[f"item {i} weight"]) for i in range(6)]
    text = " ".join(c["statement"] for c in claims)
    cfg = Config(as_of=run_eval.OFFLINE_AS_OF, max_concurrency=2)
    verify_text(text, cfg, llm=FakeLLM(extraction(*claims), rule_judge({})), search=SlowSearch())
    assert 1 <= peak <= 2
