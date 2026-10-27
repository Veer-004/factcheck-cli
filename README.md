# factcheck: an evidence-grounded fact-checking CLI

`factcheck` is a college-project command-line tool. You give it a paragraph and it:

1. extracts the atomic, checkable claims (one Gemini call),
2. retrieves real web evidence for each claim (Tavily search and page extraction),
3. has Gemini assess each retrieved passage (stance, directness, verbatim quote),
4. validates everything the model said **in code** and picks the verdict with **deterministic rules**,
5. prints a report with verdicts, validated quotations, source URLs, grounded corrections, an
   experimental evidence-strength score and a paragraph-level summary.

Orchestration uses LangGraph, with per-claim parallel fan-out via `Send`. Models use Pydantic v2.
There's no vector DB, no LangChain model wrappers, no persistent cache and no dashboard.

> **The score is an experimental evidence-strength rubric. It is not a probability and it is not
> calibrated.** Verdicts are produced automatically from web sources and can be wrong. Read the
> cited sources.

---

## Install

Requires Python 3.10+ (developed on 3.11.9, Windows 11).

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python` instead. The install provides the `factcheck` command
(`.venv\Scripts\factcheck.exe`). `python -m factcheck` also works.

Dependencies: `langgraph`, `pydantic`, `google-genai`, `tavily-python` (+ `pytest` for dev).

## Configure

Credentials come from environment variables, or from a `.env` file in the project folder. The
`.env` file is read by the CLI, the diagnostics and live evaluation runs, never by the
`verify_text()` library function or the test suite. Copy `.env.example` to `.env` and fill it in.
`.env` is git-ignored. A real environment variable takes precedence unless it's blank or contains
non-printable characters. Values are never printed. Because this folder syncs to OneDrive, revoke
the keys before you share or release the project.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio key for Gemini (required for real runs) |
| `TAVILY_API_KEY` | Tavily key (required for real runs; the SDK's keyless mode is deliberately not used) |
| `FACTCHECK_MODEL` | Gemini model ID (default `gemini-3.8-flash`) |
| `FACTCHECK_THINKING_LEVEL` | Gemini 3 thinking level: `low` (default), `medium` or `high`. No temperature is sent, so the model default applies, as Google recommends for Gemini 3. |
| `FACTCHECK_SEARCH_BUDGET` | Tavily credits per run, 0–40 (default 40) |
| `FACTCHECK_MAX_CLAIMS` | checkable claims per run, 1–12 (default 6) |
| `FACTCHECK_RUN_TIMEOUT` | run time limit in seconds (default 180) |
| `FACTCHECK_MAX_CONCURRENCY` | parallel claim workers, 1–6 (default 3) |

**Providers: Gemini primary, Ollama local fallback.**

| Variable | Purpose |
|---|---|
| `FACTCHECK_PROVIDER` | `gemini` (default) or `ollama` (local only; no Gemini key needed) |
| `FACTCHECK_OLLAMA_FALLBACK` | `true` (default): when a Gemini call fails for an operational reason, send the same request to Ollama |
| `FACTCHECK_OLLAMA_URL` | default `http://localhost:11434` |
| `FACTCHECK_OLLAMA_MODEL` | default `qwen3:4b` |
| `FACTCHECK_OLLAMA_TIMEOUT` | seconds per local request (default 90, max 300), always clipped to the run deadline |

How the fallback works:
- **When it happens:** only after a recoverable Gemini failure, meaning a rate limit (429), a 5xx,
  a timeout or a network error. The same request then goes to Ollama, and Gemini is skipped for the
  rest of that run.
- **When it doesn't:** if Gemini succeeds, Ollama is never called. A rejected key, missing model
  access or other 4xx stops the run with exit code 3, and a missing `GEMINI_API_KEY` is a
  configuration error (exit 2); neither is hidden behind the local model. Malformed output uses the
  existing single repair attempt.
- **Limits:** each fallback attempt counts against the same LLM call budget and run deadline.
- **What the local model does:** it judges only the claims and the passages Tavily retrieved. It
  never searches, so if retrieval fails the claim is still `RETRIEVAL_FAILED`. Its output goes
  through the same schema, quote, citation and verdict validation as Gemini's.
- **Reporting:** when the fallback is used, the report says so ("…handled N model call(s)") and the
  JSON `stats` record `fallback_model` and `fallback_calls`.
- **How it calls Ollama:** through the HTTP API (`/api/chat`), with `format` set to the JSON schema
  and `think: false`. Any thinking text is discarded and never shown. The tool never starts Ollama
  or downloads models. If needed, open the Ollama app (or run `ollama serve`) and run
  `ollama pull qwen3:4b`.

Check the local model on its own with `python -m factcheck.diagnostics ollama`.
Other providers, such as an OpenAI-compatible API, can be added later as another `LLMClient`
implementation in `graph.build_llm`, without changing the workflow.

PowerShell (current session only):

```powershell
$env:GEMINI_API_KEY = "<your key>"; $env:TAVILY_API_KEY = "<your key>"
```

Check what the tool sees. This prints whether each key is set, never its value:

```bash
factcheck --check-config
```

## Use

```bash
factcheck "The Eiffel Tower was completed in 1889. It is 330 metres tall."
```
```bash
factcheck --file article.txt --json-out report.json
```
```bash
echo "Water boils at 100 degrees Celsius at sea level." | factcheck -
```

Options: `--model`, `--max-claims N`, `--search-budget N`, `--timeout S`, `-q/--quiet`
(prints the summary line only), `-v/--verbose` (adds every citation, evidence IDs, score caps,
rejected model citations and run diagnostics), `--json-out PATH` (the full report, including
every passage the judge saw).

The terminal report shows the overall verdict, then for each claim: its verdict, a plain
explanation, supporting, contradicting and context evidence (each source page listed once, at most
3 sources per side, short validated quotes), any correction, and the evidence strength with what
limited it. The JSON output is a separate interface: it always contains the complete,
unabridged report, whatever the terminal flags.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Completed. Every checkable claim was resolved (SUPPORTED / REFUTED / MISLEADING), or there was nothing to check. |
| 1 | Completed with partial results: some claim is CONFLICTING, INSUFFICIENT_EVIDENCE, RETRIEVAL_FAILED or PROCESSING_FAILED, or was over the per-run limit. |
| 2 | Invalid input or configuration, including missing credentials. |
| 3 | Run-level failure: authentication rejected, claim extraction failed, step limit hit, or an unexpected error. |

### Example output (from the offline test fixtures, so the sources are synthetic)

```
[C1] REFUTED - sufficient evidence contradicts the claim (factual disagreement)
    Claim: The Eiffel Tower was completed in 1899.
    From the text: "The Eiffel Tower was completed in 1899."
    Evidence strength: 100/100 — experimental rubric; not a probability and not calibrated.
    Refuting evidence:
      [C1-E1] The Eiffel Tower - official history (primary)
        https://www.tourism.example.gov/eiffel
        "The Eiffel Tower was completed in 1889 as the entrance arch to the World's Fair held in Paris."
    Correction: The Eiffel Tower was completed in 1889, not 1899.
      Grounded in C1-E1.
```

---

## How it works

```
validate_input -> extract_claims -> validate_claims --Send x N--> claim_worker -> summarize -> render_report

claim_worker runs a per-claim subgraph:
  retrieve -> process_evidence -> sufficiency_gate --(weak, at most once)--> retrieve
                                                   --> judge -> validate_judgment --(invalid, at most once)--> judge
                                                   --> finalize
```

| Module | Responsibility |
|---|---|
| `config.py` | limits and defaults (`Config`), credential presence check |
| `models.py` | Pydantic models: untrusted `LLM*` schemas and app-produced results |
| `graph.py` | LangGraph graphs, `verify_text()` public API |
| `llm.py` | `LLMClient` protocol, call-budget wrapper, Gemini client (structured JSON, bounded retries), `FallbackLLM` policy |
| `ollama.py` | local Ollama client (`/api/chat`, JSON-schema `format`, `think: false`) used as the fallback |
| `retrieval.py` | `SearchProvider` protocol, Tavily adapter, budgeted retrieval rounds, in-run cache |
| `evidence.py` | URL dedupe, authority labels, sanitising, chunking, lexical ranking, independence clusters, evidence IDs |
| `validation.py` | input, claim anchoring/dedupe, quote/stance/numeric validation, correction grounding |
| `judging.py` | deterministic verdict rules and summary labels |
| `scoring.py` | evidence-strength rubric |
| `security.py` | sanitising, per-run delimiters, injection heuristics, secret redaction |
| `prompts.py` | system prompts (kept separate from untrusted content) |
| `report.py`, `cli.py` | terminal/JSON rendering, CLI and exit codes |
| `textnorm.py`, `limits.py` | normalisation and number parsing; thread-safe deadline and budgets |

### Claims

One structured call extracts the claims. Code then drops any claim that fails these checks, and the
report lists each dropped claim with its reason:

- its `source_excerpt` doesn't occur in the input (after normalisation),
- it contains figures that aren't in the input,
- its wording isn't grounded in the input (less than 60% of its content words appear there),
- it's a near-duplicate of an earlier claim,
- it's an instruction to the system ("ignore previous instructions…").

Opinions, predictions and vague statements become `NOT_CHECKABLE`. At most 6 checkable claims are
processed. Any extra ones are listed as over the limit.

**Normalisation** (used for excerpts and quotes): NFKC; zero-width characters and soft hyphens
removed; typographic quotes, dashes and the minus sign mapped to ASCII; whitespace collapsed;
case-folded. Words, digits and their order must otherwise match exactly.

### Evidence

- Up to 2 primary queries, 5 results each, and at most 6 unique URLs per claim. URLs are
  canonicalised: `http`/`https`, `www.`, mobile/AMP hosts (`en.m.wikipedia.org`), host case,
  trailing slashes, fragments and tracking parameters all map to one page.
- Search results request publication dates (`include_published_date`); without it Tavily returns
  no dates for general searches.
- Full page text comes from Tavily extract (top 5 URLs, which is 1 credit). If extraction fails,
  the search snippet is used and its weight is halved.
- Text is sanitised (control and bidi characters, markup and fake delimiters removed), chunked to
  1,200 characters or fewer, and ranked by deterministic lexical relevance × authority. At most 2
  passages per source and 8 passages per claim are kept.
- **Authority** depends on the claim. Government, intergovernmental, academic and journal domains
  are *primary*; established news/reference outlets and fact-check sites are *secondary*. An
  organisation's own site is primary for statements about itself, but *other* for comparative
  claims about itself (self-interest).
- **Independence clusters** group sources from the same organisation (`bbc.com` = `bbc.co.uk`) and
  near-duplicate (syndicated) text: 5-shingle Jaccard ≥ 0.5, or ≥ 80% of the shorter text's
  shingles found in the longer one (so a search snippet of a syndicated article doesn't count as
  a second independent source). Repeated passages from one source never add strength.
- Displayed quotations are the source's exact characters, located through the same normalisation
  that validated them.
- If the evidence looks weak, one fallback round runs with differently worded queries. If every
  search fails technically, the result is `RETRIEVAL_FAILED`, never `INSUFFICIENT_EVIDENCE`.

### Judging and validation

The judge sees only the delimited claim and passages. It has no tools and no search. For every
assessment, code checks that:

- the evidence ID exists, and the URL is the one stored for it;
- the quote is a verbatim substring of that passage (after normalisation) and at least 4 words long;
- the passage is about the claim's subject and the quote shares terms or figures with the claim;
- a SUPPORTS quote isn't a debunking ("myth", "false", "no evidence"…) of a non-negated claim;
- numeric SUPPORTS quotes contain the claimed figure, within rounding and with "about / more than /
  under" qualifiers respected;
- numeric REFUTES assessments carry an explicit comparison whose entity, metric, unit, scope and
  period all match, whose evidence figure appears in the quote, and whose figures actually
  disagree. A different number for a different measurement is never treated as a refutation.

If most citations are fabricated (unknown IDs or non-verbatim quotes), or the output is malformed,
the judge gets **one** repair attempt. If the output is still invalid, the claim becomes
`PROCESSING_FAILED`.

### Verdict rules (deterministic, `judging.py`)

Validated citations are split into support, refutation and material-context sides. Each side's
strength *S*, *R*, *M* is a noisy-OR over independent clusters. Threshold *T* = 0.5:

| Condition | Verdict |
|---|---|
| S ≥ T and R ≥ T | CONFLICTING |
| R ≥ T | REFUTED |
| S ≥ T and M ≥ T | MISLEADING |
| S ≥ T | SUPPORTED |
| otherwise | INSUFFICIENT_EVIDENCE |

The model's proposed verdict is recorded. When it disagrees with the rules, the report says so.

**Corrections** are produced only for REFUTED/MISLEADING claims. A correction must cite validated
refuting or context evidence, every figure in it must appear in the cited passages (or the claim),
and at least 60% of its content words must be grounded. Otherwise the report says that no
correction could be established.

### Evidence-strength score (`scoring.py`)

Passage weight = authority (primary 1.0 / secondary 0.75 / other 0.4) × directness (explicit 1.0 /
partial 0.6 / inferential 0.3) × recency for time-sensitive claims (≤ 1 year 1.0 / undated 0.8 /
older 0.6) × 0.5 if snippet-only × 0.5 if injection-flagged. The two penalties are applied
independently, which is the conservative reading.

Each cluster counts once (its strongest passage), and clusters combine by noisy-OR,
`S = 1 − ∏(1 − w)`.

| Verdict | Base score (×100) |
|---|---|
| SUPPORTED | S·(1−R) |
| REFUTED | R·(1−S) |
| MISLEADING | √(S·M)·(1−R) |
| INSUFFICIENT_EVIDENCE | max(S, R, M), the strength of what was retrieved (not truth) |
| CONFLICTING | no single score; support and refutation are shown separately |

Caps: one independent cluster → 70; no primary source → 85; only fact-check sites → 60; any
injection-flagged contributing passage → 60; time-sensitive with no passage from the last year → 60;
MISLEADING → 75. The result is rounded to the nearest 5. There's no score for NOT_CHECKABLE or for
technical failures.

### Summary label

Any REFUTED → `CONTAINS_FALSE_CLAIMS`. Otherwise, any MISLEADING → `CONTAINS_MISLEADING_CLAIMS`.
If every checkable claim is SUPPORTED → `ALL_CHECKED_CLAIMS_SUPPORTED`. If there's nothing
checkable → `NOTHING_TO_CHECK`. Anything else → `PARTIALLY_RESOLVED`.

---

## Security model

All user text and web content is treated as untrusted data:

- System instructions go through Gemini's separate `system_instruction` channel. Claims and
  passages are wrapped in delimiters with a random per-run token, and delimiter look-alikes are
  stripped from content.
- The judge has no tools and can't trigger searches or network requests. Nothing from claims,
  sources or model output is ever executed.
- Evidence IDs are assigned by code. Every displayed quote is re-verified against stored text. The
  verdict and score are computed in code, so an instruction in a page can't change the rules.
- Pages containing instruction-like text are flagged. Their passages are down-weighted (×0.5) and
  cap the score at 60.
- API keys are never printed. Error messages pass through redaction, and `--check-config` shows
  presence only.

The adversarial tests show that the specific attacks they define fail: direct and indirect
injection, fake citations, malformed output and forced verdicts. **They do not prove the system is
immune to prompt injection.** The injection heuristics are regex-based and can be evaded. The main
protection is that a verdict needs verbatim, on-topic, validated evidence.

## Cost and usage controls

| Limit | Default |
|---|---|
| input length | 4,000 characters |
| checkable claims | 6 |
| queries per retrieval round / results per query | 2 / 5 |
| unique URLs per claim / extracted URLs | 6 / 5 |
| passages per judge call / characters per passage | 8 / 1,200 |
| fallback rounds per claim | 1 |
| Tavily credit budget per run | 40, reserved *before* each request; failed requests are not refunded |
| LLM calls per run | 14 (1 extraction + 1 retry + 6 judges + 6 repairs) |
| HTTP 429 / 5xx / connection-error retries | 2, honouring `retryDelay` up to 30 s; a retry sleep starts only if the retry itself still fits in the run |
| response timeouts (`ReadTimeout`) | not retried: another attempt would spend the same budget waiting again |
| per-request timeout | a **total** per request: connect ≤ min(10 s, ¼ of the budget), the rest for the response. Both httpx and requests would otherwise apply a scalar timeout to each phase separately. SDK retries are disabled. |
| run time limit | 180 s; each Gemini request timeout is clipped to the remaining time, and no request starts with < 5 s left (Tavily: < `search_timeout_s`) |
| LangGraph recursion limit | 25 (main graph and per-claim subgraph) |
| parallel claim workers | 3 (`max_concurrency`) |

Tavily pricing (checked 2026-10-26 at docs.tavily.com): a basic search is 1 credit, and a basic
extract is 1 credit per 5 successful URLs. The free plan has 1,000 credits/month. Worst case per
claim is 2 searches + 1 extract per round × 2 rounds = 6 credits, so 6 claims ≤ 36 credits.

Gemini (ai.google.dev, checked 2026-10-26): `gemini-3.8-flash`, `gemini-3.5-flash-lite` and
`gemini-3.1-flash-lite` list a free tier. On the free tier, Google states that content may be used
to improve its products. Free-tier rate limits are shown only in AI Studio, so they were **not
verified** here. If you hit quota errors, try `--model gemini-3.5-flash-lite`.

When a budget or the deadline is reached, unfinished claims are marked `PROCESSING_FAILED` with the
reason, and the run continues to a report.

---

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

The suite is fully offline and deterministic. It uses fake providers in `tests/fakes.py`, and an
autouse fixture removes API keys for every non-live test. It covers security, evidence processing,
validation, scoring and verdict rules, the LangGraph flows (all verdict paths, fallback, repair,
budgets, deadline, recursion limit, parallel isolation, concurrency bound), the provider adapters
(mocked SDKs: 429/retry-after, fail-fast auth, redaction), report/CLI exit codes, and all offline
evaluation cases.

The live smoke test (1 claim, ≤ 6 credits, ≤ 3 model calls) runs only when you opt in:

```powershell
$env:FACTCHECK_LIVE_TESTS = "1"; .venv\Scripts\python.exe -m pytest tests/test_live.py -q
```

## Evaluation

```bash
.venv\Scripts\python.exe eval/run_eval.py --split offline
```
```bash
.venv\Scripts\python.exe eval/run_eval.py --split dev --max-cases 5
```

- `eval/offline_cases.jsonl`: 20 scripted robustness cases (fake model and fake search). They test
  the pipeline and the deterministic rules. **They don't measure model accuracy.**
- `eval/dev.jsonl` (14) and `eval/heldout.jsonl` (12): live cases covering supported, refuted,
  misleading, conflicting, insufficient, time-sensitive, numeric, compound, non-checkable and
  prompt-injection claims. They're disjoint by test. Some cases list acceptable alternative
  verdicts where reasonable checkers differ. Tune on dev only, and run held-out once at the end.
- Reported metrics: strict and lenient accuracy, per-class precision/recall/F1, macro-F1, citation
  validity (independently re-verified against the judged passages), correction grounding, mean
  score by verdict, latency, search credits and LLM calls. Live runs enforce a total credit budget
  across cases (default 120). Results are written to `eval/results/`.

## Troubleshooting

Check each provider on its own before a full run. Each diagnostic makes exactly one request with no
retries. A hard 15 s wall-clock deadline covers DNS, client creation, connect/TLS and the response
wait: the request runs in a daemon thread that is aborted at the deadline. Each prints elapsed time,
per-stage timings, exception type and HTTP status, and never prints keys or headers:

```bash
.venv\Scripts\python.exe -m factcheck.diagnostics gemini-model
```
```bash
.venv\Scripts\python.exe -m factcheck.diagnostics gemini
```

`gemini-model` makes a metadata-only `models.get` call: is the model ID valid for your key, and
does it support `generateContent`? `gemini` makes one small generation call with the app's
settings. Add `--thinking-level high` to reproduce the old default.
```bash
.venv\Scripts\python.exe -m factcheck.diagnostics tavily
```
```bash
.venv\Scripts\python.exe -m factcheck.diagnostics ollama
```

`ollama` checks that the local server is up and the model is installed (two local metadata calls),
then makes one small structured chat with a 60 s deadline (a cold start loads the model first). It
also reports whether any thinking output came back despite `think: false`.

| Symptom | Fix |
|---|---|
| `Ollama is not reachable` / `Is Ollama running?` | Open the Ollama app or run `ollama serve`, and check `FACTCHECK_OLLAMA_URL`. |
| `model 'qwen3:4b' is not installed` | Run `ollama pull qwen3:4b` (the tool never downloads models). |

| Symptom | Fix |
|---|---|
| `LLMNetworkError: could not connect …` / `TLS certificate verification failed` | Network, proxy or firewall is blocking `generativelanguage.googleapis.com`. For an HTTPS-intercepting proxy, set `SSL_CERT_FILE` to its CA bundle. |
| `LLMTimeoutError: Gemini sent no response within Ns` | Connected, but no answer in time. Run the Gemini diagnostic: if it is fast, the full request is too slow for `llm_timeout_s`; if it is slow too, it's the network or the API. |
| `LLMRateLimitError` (HTTP 429) | Free-tier quota or rate limit. Wait, or check the limits in AI Studio. |
| `error: GEMINI_API_KEY is not set` (exit 2) | Set the variable in the same shell, then check with `factcheck --check-config`. |
| `run failed: authentication/access error` (exit 3) | The key was rejected, or the model isn't available to it. Check the key in AI Studio/Tavily, or try `--model gemini-3.5-flash-lite`. |
| Claims `PROCESSING_FAILED` with "budget" / "time limit" | Raise `--search-budget` (max 40) or `--timeout`, or check fewer claims. |
| `RETRIEVAL_FAILED` | Tavily/network problem. This is a technical failure, not a statement about the claim. |
| Many `INSUFFICIENT_EVIDENCE` | Expected for obscure or very recent claims. The tool won't fall back on model memory. |
| Garbled characters on Windows | The CLI switches stdout to UTF-8. If needed, set `PYTHONIOENCODING=utf-8`. |

## Known limitations

- **English only.** Tokenisation, relevance and the injection/negation heuristics assume English
  text in Latin script. Other scripts produce no tokens, so their claims are dropped.
- Relevance and the topic/scope checks are lexical, not semantic. A passage about a different
  entity with the same name can pass them, and synonyms can cause valid evidence to be rejected.
- Authority is decided by domain lists and simple rules. It isn't real source-credibility
  assessment.
- The judge's stance labels can't be fully verified by code. Validation catches fabricated quotes,
  off-topic passages, debunking-as-support and numeric mismatches, but not every misreading.
  Misleading and conflicting cases rely most heavily on the model.
- The checkability decision (opinion vs fact) is taken from the model.
- Displayed quotes are the model's copy, verified equal to the source after normalisation, so the
  capitalisation may differ from the source.
- The score weights and the 0.5 threshold are hand-chosen and uncalibrated.
- Results depend on what Tavily returns at query time. Runs aren't reproducible across days.
