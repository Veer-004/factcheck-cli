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
(`.venv\Scripts\factcheck.exe`). `python -m factcheck` also works. The other three commands are
`factcheck-ocr` (images), `factcheck-video` and `factcheck-audio`. Each has its own section below.

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
- the quote is a verbatim substring of that passage (after normalisation) and at least 4 words long,
  or a shorter table/infobox cell that is a labelled figure sharing a word with the claim
  ("Payload : 1,650 kg"). Only a quote that is not in the passage counts as fabricated; if half or
  more of a judgment's citations are fabricated, the judgment is rejected and repaired once;
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

The OCR tests (`tests/test_ocr_*.py`) use synthetic boxes, generated images and a fake engine,
so they need no OCR models. A real-engine OCR smoke test (RapidOCR, no network) is opt-in:

```powershell
$env:FACTCHECK_OCR_LIVE_TESTS = "1"; .venv\Scripts\python.exe -m pytest tests/test_ocr_live.py -q
```

The video tests (`tests/test_video_*.py`) decode real, tiny videos that they generate, but
replace Whisper, the OCR engine, Gemini and Tavily with fakes. A real Whisper + RapidOCR
extraction test is opt-in (Windows only, model must already be cached, no network). See
[Video fact-checking](#video-fact-checking-factcheck-video).

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

## Image to text (`factcheck-ocr`)

A separate command that extracts text from images: screenshots, memes, posters, scanned
documents, multi-column pages. Plain extraction shares no code with the fact-checker: without
`--fact-check`, the `factcheck_ocr` package never imports `factcheck`, LangGraph, Gemini or
Tavily (a test checks this). It never interprets the image itself. To fact-check the text it
reads, see [Image fact-checking](#image-fact-checking-factcheck-ocr---fact-check).

```bash
.venv\Scripts\python.exe -m pip install -e ".[ocr]"
```
```bash
.venv\Scripts\factcheck-ocr.exe screenshot.png poster.jpg --json-out ocr.json
```

Options: `--engine rapid|paddle|paddle-structure`, `--layout auto|geometric|none`,
`--min-score` (default 0.5), `--model-size small|medium`, `--textline-orientation`,
`--allow-download`, `--max-side`, `--check`, `-q`, `-v` (each line's confidence and pixel box).
Exit codes: 0 text found, 1 an image had no text, 2 invalid input/setup, 3 engine failure.

**How it works.** The image is decoded with Pillow under limits: 25 MB file, 50 MP (checked
before decoding), and an allow-list of formats. EXIF rotation is applied and transparency is
flattened. Images with a side over 4096 px are downscaled, and boxes are mapped back to original
pixels. The engine returns text boxes. `layout.py` then rebuilds the reading order from geometry,
with no model:

1. recursive XY-cut on the boxes: split along empty horizontal bands and vertical gutters. A
   vertical cut is taken only when the two sides really run side by side, so newspaper columns
   stay together and alternating chat bubbles read top to bottom;
2. consecutive boxes on one row, a word gap apart, become a line;
3. lines become blocks, broken at paragraph gaps and column changes (blank line between blocks).

Recognised text is untrusted: control and bidi-override characters are stripped before printing.

**Engines.**

| Engine | Models | Install | Downloads | Notes |
|---|---|---|---|---|
| `rapid` (default) | PP-OCRv6 small (det + rec), ONNX Runtime | `.[ocr]`, about 240 MB installed | none: models ship in the `rapidocr` wheel | `--model-size medium` fetches PP-OCRv6 medium ONNX (about 130 MB, once, with `--allow-download`) |
| `paddle` | PP-OCRv6 medium + PP-LCNet text-line orientation | `.[ocr-paddle]` in a **separate venv**, about 1.2 GB | about 140 MB from Hugging Face into `~/.paddlex`, once, with `--allow-download` | geometric layout as above |
| `paddle-structure` | PP-StructureV3: PP-DocLayout_plus-L + PP-DocBlockLayout + PP-OCRv5 server | same venv | about 400 MB, once, with `--allow-download` | the layout model's reading order and block labels (title, text, header…) |

No engine downloads anything unless you pass `--allow-download`. `factcheck-ocr --check` shows
what's installed and which models are present. Set up the Paddle venv like this:

```bash
python -m venv C:\Users\you\venvs\factcheck-paddle
```
```bash
C:\Users\you\venvs\factcheck-paddle\Scripts\python.exe -m pip install -e ".[ocr-paddle]"
```

Why a separate venv: `paddlex` pins `opencv-contrib-python==4.10.0.84`, which clashes with
RapidOCR's `opencv-python-headless`. `paddlepaddle` is pinned to 3.2.2 because 3.3.0 and 3.3.1
crash CPU inference with oneDNN (`ConvertPirAttribute2RuntimeAttribute not support`,
[Paddle#77340](https://github.com/PaddlePaddle/Paddle/issues/77340); not fixed in a release as
of October 2026). Keep the venv outside OneDrive: it's about 1.2 GB.

**Evaluation.** `eval/make_ocr_samples.py` renders six synthetic images with known text and
reading order. They are a two-column article, a three-column report, an Impact meme with outlined
text on a busy background, a poster, a chat screenshot and a low-contrast Comic Sans page.
`eval/run_ocr_eval.py` measures the following:

- CER over the whole text, which counts reading-order mistakes as errors;
- order-free word recall;
- the share of expected line pairs in the right order.

```bash
.venv\Scripts\python.exe eval\make_ocr_samples.py
```
```bash
.venv\Scripts\python.exe eval\run_ocr_eval.py --engine rapid --show
```

Measured on 2026-10-27 (i5-12500H laptop, CPU only, warm runs):

| Engine | mean CER | word recall | line order | seconds per image |
|---|---|---|---|---|
| `rapid` (small, default) | 0.039 | 1.000 | 0.97 | 1.6 |
| `rapid --model-size medium` | 0.039 | 1.000 | 0.97 | 12.6 |
| `paddle` | 0.039 | 1.000 | 0.97 | 14.6 |
| `paddle-structure` | 0.002 | 0.982 | 1.00 | 50.5 |

In every geometric-layout row, the only error is the poster's 2×2 detail grid (date | place over
time | price). It is read column by column, while the expected text reads it row by row; both
readings are defensible. PP-StructureV3's layout model read that grid row by row. It also rejoins
wrapped lines into paragraphs, so line breaks are not preserved. Its one recognition slip was
"Saturday12" with a missing space, and it is about 30× slower than the default. Findings that
shaped the defaults:

- RapidOCR's text-direction classifier (PP-OCR v2.0) flipped the meme's outlined top line by 180°
  and produced **confident garbage**: "WHSSN TN TASS" at 0.59 with small models, "SHEASS" at 0.85
  with medium. Confidence did not flag it. The classifier is therefore off by default
  (`--textline-orientation` turns it back on for genuinely upside-down text).
- Medium models gave no accuracy gain on these samples and were 8× slower.
- Paddle's C++ code prints diagnostics such as `ReduceMeanCheckIfOneDNNSupport` straight to
  stdout. The adapter redirects them to stderr, so the extracted text on stdout stays clean.
- On the first PaddleOCR run, Windows Smart App Control blocked a SciPy DLL (`_linalg_pythran`)
  once. The next run worked unchanged. If you see `An Application Control policy has blocked
  this file`, that's Windows security policy, not a bug in this tool.

These are six synthetic images. They show that layout handling and stylised fonts work, but they
are no benchmark for real photos, JPEG artefacts, curved or rotated text, or non-Latin scripts.
Try your own images.

**English clean-up (images and video frames).** The OCR is tuned for English text. After
recognition, three deterministic steps run in `factcheck_ocr/pipeline.py`. They are shared by
`factcheck-ocr` and `factcheck-video`, and each is reported as a note, never silently:

1. **A noise filter** removes fragments that are almost never real text:
   - text with no Latin letter or digit (`难`, `三`, `★`, `??`);
   - a lone letter (`X`, `S`, `Q`);
   - two letters that are not a common word or abbreviation (`um`, `TE`; but `Hi`, `OK`, `US` and
     `BE` are kept);
   - repeated-letter garbage (`IIII`, `mwwwmm`).

   Text that is only digits is always kept. Use `--keep-noise` (images) or `--keep-ocr-noise`
   (video) to turn the filter off.
2. **Symbol fixes:** `℃`/`℉` become `°C`/`°F`, full-width characters become normal ASCII, and a
   standalone `Il` becomes the Roman numeral `II` ("Pokhran Il" → "Pokhran II"). Full Unicode
   NFKC is deliberately *not* used, because it would turn `10³` into `103`.
3. **Tiny figures are marked uncertain.** A text box that contains digits but is under 11 px tall
   (in the pixels the recogniser saw) has its confidence capped at 0.60. The text is kept as read,
   but it is shown with `?`, and the fact-check gates set aside claims that depend on it. The
   reason: in a 640×360 video, a 10-px-tall "$1.5 billion" was read as "**$1.8** billion" with 0.92
   confidence. This rule rests on one measured case and is a deliberately cautious heuristic.

`eval/run_real_ocr_eval.py` measures OCR on 19 frames from three real English videos
(screenshots of infographics, documents, maps and burned-in subtitles). The ground truth was
written by hand. The frames come from personal videos, so `eval/ocr_real/` is git-ignored. Measured
on 2026-11-17:

| | text read | figures read exactly | junk in output | blank frame with output |
|---|---|---|---|---|
| Before the clean-up | 95.7% | 34/37 | 19 tokens (6.5%) | 1 (`难`) |
| **After** | **96.4%** | 34/37 (the misread one is now flagged uncertain) | **13 tokens (4.5%)** | 0 |

Alternatives tested on the same frames and **rejected**:

| Option | Result | Reason rejected |
|---|---|---|
| 2× upscaling of small frames | recall 0.931, junk 8.3% | Worse on both counts |
| PP-OCRv6 medium models | recall 0.960, junk 4.4% | 10× slower |
| RapidOCR English-only recogniser (PP-OCRv5) | recall 0.946 | Fixed "$1.5" but introduced other errors ("e8ay", "Poli") |
| English PP-OCRv4 recogniser | recall 0.855 | Much worse |
| Re-reading figures with the English model and flagging disagreements | n/a | Missed the "$1.8" misread and raised false alarms ("5,000" → "5,00") |

Remaining errors are misreads inside real text: a table's "4,000" read as "4.000",
"e-scooter limit" read as "e-scooter-limit", sideways "Stage" labels, and a logo partly hidden
by an animation ("SFACEX").

```bash
.venv\Scripts\python.exe eval\run_real_ocr_eval.py --show
```

**OCR limitations.**
- Reading order assumes left-to-right, top-to-bottom text that is roughly horizontal. It can't
  split columns that the detector itself merged into one box. Text inside figures is ordered by
  position like any other text.
- A confidence score is not a correctness guarantee (see the meme finding above).
- Only the first frame of a multi-frame GIF/TIFF is read. PDFs are not supported.
- Model licences: RapidOCR and PaddleOCR/PaddleX are Apache-2.0; ONNX Runtime is MIT. The
  PP-OCR model weights are published by PaddlePaddle under Apache-2.0.

## Image fact-checking (`factcheck-ocr --fact-check`)

Fact-checks the **text recognised in one image**. The image itself is never interpreted:
no object recognition, visual reasoning or judging by appearance.

```bash
.venv\Scripts\factcheck-ocr.exe screenshot.png --fact-check
```
```bash
.venv\Scripts\factcheck-ocr.exe meme.jpg --fact-check -v --json-out image_report.json
```

It needs the `.[ocr]` install plus the same keys and settings as `factcheck` (`.env`,
`FACTCHECK_*`). Options: `--ocr-threshold` (default 0.80), `--model`, `--max-claims`,
`--search-budget` and `--timeout`. The OCR options (`--engine`, `--model-size`, `--min-score`,
`--layout`, `--allow-download`) also apply. Exit codes are the text CLI's: 0 resolved,
1 partial or unresolved (or no usable text), 2 invalid input or config, 3 run failure.

**Pipeline.**

```
image -> OCR (factcheck_ocr) -> extraction-quality check -> claim extraction (OCR-aware note)
      -> per-claim OCR gate -> retrieval -> evidence -> judge -> validation -> deterministic verdict
      -> image report (extracted text + OCR status + the normal fact-check report + per-claim OCR trace)
```

Everything after OCR is the text workflow's code (`graph.run_pipeline`, which `verify_text` also
calls), unchanged except for two opt-in hooks that the text workflow never sets:

- an OCR note in the claim-extraction *user* prompt. It tells the model that the text may contain
  recognition errors and that line breaks follow the layout. It also says not to correct or
  complete words, to treat fragments, slogans and incomplete headlines as VAGUE, and that the
  text is the material being checked, not evidence;
- a claim gate between claim validation and retrieval (see below).

Validation still forces every claim's `source_excerpt` to occur verbatim in the OCR text. A model
that "repairs" an OCR misreading (for example `18B9` into `1889`) has its claim dropped instead
of silently checking text the image doesn't contain. Prompt-injection handling is unchanged: image
text is wrapped in random per-run delimiters and instruction-like "claims" are dropped. Retrieved
pages carrying instructions are down-weighted and capped.

**OCR quality and uncertainty.** `factcheck_ocr/quality.py` grades the extraction:

| Status | Meaning | What happens |
|---|---|---|
| `OK` | every line's OCR confidence ≥ threshold | checked normally |
| `LOW_CONFIDENCE` | some lines below threshold, or low-confidence fragments were discarded | checked; OCR warnings added to the report |
| `UNREADABLE` | every line below threshold, or no words | **not** fact-checked; no model or search call |
| `NO_TEXT` | nothing recognised | **not** fact-checked; no model or search call |

Each claim is traced back to the OCR lines its excerpt comes from (line numbers, block, lowest
OCR confidence). If any of those lines is below `--ocr-threshold`, the claim is **withheld**: it
is not searched or judged, costs no search credits, and is reported as `PROCESSING_FAILED`
("the image text it comes from could not be read reliably (try a clearer image)"). That status is
a technical outcome, never a statement about truth, so a misread can't become a confident
REFUTED. The cost is that a correctly read but low-confidence line goes unchecked; lower the
threshold to accept that risk.

Three separate quantities, never mixed:

- **OCR confidence** is how reliably characters were read. It only gates which claims are checked.
- The **evidence strength** score is the existing experimental rubric over web evidence.
- The **verdict** comes from the deterministic rules over validated citations.

**Tests and evaluation.**

- `tests/test_image_factcheck.py` and `tests/test_ocr_quality.py` cover 30 offline cases. They
  run a scripted OCR engine and the fake LLM and search providers over real image files and the
  real pipeline. Cases include:
  - supported, refuted with a correction, multiple claims, misleading, conflicting and
    insufficient evidence;
  - no checkable claims, and empty or unreadable images;
  - a low-confidence claim being withheld, the threshold setting, and a model "correcting" OCR
    text;
  - search failure and timeout, malformed and invented judge output, and injection in image
    text and in a retrieved page;
  - missing and unsupported files, a missing engine, JSON output and CLI exit codes;
  - checks that the text workflow's prompt and CLI are unchanged.
- `eval/run_image_eval.py` is a **live** evaluation (real RapidOCR + Gemini + Tavily; uses API
  credits). It renders three images with hand-checked expected outcomes. On 2026-10-27, with
  `gemini-3.5-flash-lite`, it scored strict **4/4**: the Great Wall meme was REFUTED; the tweet
  screenshot had Eiffel 1889 SUPPORTED and "water boils at 50 °C" REFUTED, with a grounded
  correction; the opinion poster gave NOTHING_TO_CHECK. That is three images, not a benchmark.

```bash
.venv\Scripts\python.exe eval\run_image_eval.py --show
```

**Image fact-checking limitations (measured or unverified).**
- RapidOCR confidence is a weak warning signal. On a progressively blurred screenshot, line
  confidence stayed at or above 0.97 until text disappeared entirely, and at one blur level a
  whole line vanished with no low-confidence signal. Earlier, a misread meme line scored 0.85.
  So the OCR gate catches some misreads, not all, and missing text is not detected.
- A claim is traced through its excerpt. Text the OCR engine never detected can't be flagged, so
  a claim may be checked without context that was in the image (a "not" in a missed line, a date
  in a footer).
- One image per run, sharing the text workflow's limits: 4,000 characters of extracted text,
  6 checkable claims, 40 search credits and 180 s.
- Only the PaddleOCR engines' venv lacks the fact-checking dependencies by default. Install
  `-e ".[ocr-paddle]"` with dependencies there, or use the default rapid engine.
- The live path through the Ollama fallback has not been tested for images; for text it is
  known not to produce verdicts with `qwen3:4b`.

## Video fact-checking (`factcheck-video`)

A third, separate command fact-checks the claims **spoken** in a video and shown as **on-screen
text**. It is CLI only. It does not interpret pictures: objects, scenes, gestures and events are
never used as evidence or as claims. Only speech and visible text are checked.

```bash
.venv\Scripts\python.exe -m pip install -e ".[video]"
```
```bash
.venv\Scripts\factcheck-video.exe --check
```
```bash
.venv\Scripts\factcheck-video.exe clip.mp4 --allow-download
```

The first run with `--allow-download` downloads the speech model (base, about 145 MB) into
`~/.cache/huggingface/hub`. Later runs reuse it and never download anything. Without that flag, a
missing model is reported as an error (exit 2) and nothing is downloaded. Other useful forms:

```bash
.venv\Scripts\factcheck-video.exe clip.mp4 --extract-only -v
```
```bash
.venv\Scripts\factcheck-video.exe clip.mp4 --json-out report.json --whisper-model small --frame-interval 1
```

`--extract-only` shows the transcript and on-screen text and needs no API keys. The fact-check uses
the same `.env` keys as `factcheck`. Exit codes match `factcheck`: 0 resolved, 1 partial or nothing to
check, 2 invalid input, missing model or configuration problem, 3 run failure. `python -m
factcheck_video` also works.

**Prerequisites.** You don't need to install FFmpeg separately. faster-whisper depends on PyAV, and
PyAV's wheels bundle the FFmpeg libraries (FFmpeg 8, libavcodec 62 on the test machine). They read
MP4, MOV, MKV, WebM, AVI and other common formats. The extension allow-list is in
`factcheck_video/media.py`. The `video` extra adds faster-whisper, CTranslate2, PyAV, tokenizers and
huggingface-hub, plus the OCR extra's packages. It upgrades nothing that is already installed.

### How it works

```
video -> probe (PyAV/FFmpeg: streams, duration, size limits)
      -> audio track --stream--> temp 16 kHz mono WAV -> faster-whisper (CPU, int8, VAD) -> sentence segments
      -> frames sampled every 2 s -> unchanged-frame check -> factcheck_ocr (RapidOCR) -> text blocks
      -> merge identical text across frames and across speech/screen (stage 1)
      -> labelled, time-ordered text -> factcheck.graph.run_pipeline:
           claim extraction (video note) -> validation with the strict duplicate rule (stage 2)
           -> extraction-reliability gate -> retrieval -> judge -> validation -> deterministic verdict
      -> claim provenance (speech/screen, timestamps, segment and text-block ids) -> report
```

- **Speech and on-screen text are both extracted first, then compared.** Deduplication saves
  verification work. Extraction itself still processes both.
- **Audio.** The first audio stream is decoded packet by packet into a temporary WAV in a private
  temp folder. The folder is deleted afterwards, also on failure (tested). The video is never fully
  loaded into memory. If decoding fails part-way, the audio up to the error is still used and a note
  is added.
- **Speech-to-text.** faster-whisper runs on CPU with int8 quantisation, voice-activity filtering
  and word timestamps. The model is multilingual and detects the language unless `--language` is
  set. Whisper segments are split into sentences at sentence-final punctuation (not after "U.S."
  or "Dr."). This happens because Whisper can put sentences spoken seconds apart into one segment.
  A segment counts as reliable when Whisper's own confidence, exp(average log-probability), is at
  least `--speech-threshold` (default 0.40). It also must not be repetitive (compression ratio above
  2.4) or look like silence.
- **Frames.** One frame is sampled every `--frame-interval` seconds (default 2, at most
  `--max-frames` 150). The interval widens automatically for long videos. Frames are found by
  seeking and short forward decoding, and downscaled to 1920 px. A frame whose 160x90 RGB thumbnail
  is unchanged reuses the previous frame's OCR result. It is RGB because a grayscale check missed a
  colour change of similar brightness. Each frame goes through the OCR module's own box filtering
  and reading-order code (no second OCR model). Each OCR block (caption, title, label) is tracked
  across frames.
- **Subtitles.** Burned-in subtitles are recognised and attached to the spoken sentence they
  transcribe, so a subtitled sentence is checked once, in full (see the deduplication rules
  below).
- **The fact-check input.** It lists the unique text pieces in time order, under `[speech]`,
  `[on-screen text]` or `[speech, also shown on screen]` labels. Timestamps are left out, so no time
  figures can end up in claims. The extraction prompt tells the model the text is an automatic
  transcript or OCR, not to correct or complete it, and that greetings, calls to subscribe and
  opinions are not claims. It also tells the model to extract a claim once for each place it appears,
  with identical wording when the claim is the same, so that code can merge it. Everything after that
  is the shared pipeline, with all its validation and prompt-injection defences unchanged.
- **Reliability gate.** A claim whose every source (speech segment or text block) is below its
  confidence threshold is withheld as PROCESSING_FAILED with an `extraction:` reason. That is a
  technical outcome, not a verdict, and no search credits are spent on it. If every extracted
  piece is unreliable, no model or search call is made at all.

### Deduplication rules

| Stage | What is merged | What is never merged |
|---|---|---|
| Across frames | The same block text (case, punctuation and spacing ignored), anywhere in the video: one observation with every frame time. Near-identical OCR variants (≥ 0.92 character similarity) in the **next** sampled frame. | Variants whose numbers, letter-digit codes ("A380", "boi1s"), negation, units, months, comparison words or 1-2-letter words differ |
| Speech vs screen, before extraction | A speech sentence and a text block with **identical** wording after removing case, punctuation and spacing: one unit carrying both sources | Anything less than identical, except subtitles (next row) |
| Subtitles, before extraction | On-screen text seen while a sentence is spoken whose words occur **in order** in that sentence (or across the boundary into the next one): it is attached to the spoken sentence, checked once in full context, and kept in the provenance | Captions of 1-2 characters ("1", "2"); captions whose spoken sentence has a negation or denial word outside the caption ("It is a **myth** that the Great Wall is visible …" keeps the caption "The Great Wall is visible …" separate) |
| Claims, after extraction | Two claims whose statements are identical after normalisation, or have the same content words **in the same order** and the same numbers, negation, units, months, codes, comparison/quantity words, modal/tense/logic words and names, **and** whose verbatim source excerpts agree on numbers, negation, units, months and codes | Synonyms, paraphrases, reordering, an added scope word ("adult human body"). These are checked twice: extra cost, nothing lost |

Claim-level merging uses the existing `near-duplicate` step in `validation.validate_claims`, with the
stricter video rule passed in (`factcheck_video/textmatch.py`). The text workflow's own rule ignores
negation. That is fine for one paragraph, but a video can say "is not visible" while the screen says
"is visible". A merged claim is reported once, with every place it came from. When equivalence is
uncertain, both claims are kept. Conflicting speech and screen claims are both checked
independently; neither source is treated as correct.

### Reports, timestamps and provenance

The terminal report shows:

- the media details;
- each stage's status (OK, LOW CONFIDENCE, UNRELIABLE, nothing found, no track, skipped, FAILED);
- the timed transcript, with `?` marking unreliable segments;
- the on-screen text with the sampled-frame times it was seen, so a gap means the text disappeared
  and came back;
- the shared fact-check report, where each claim also gets lines such as
  `Spoken ~00:00.3-00:04.0 (word timing), segment 1, speech confidence 0.82` and
  `On screen: seen in sampled frames 00:02.0-00:06.0, text block 1, OCR confidence 0.99`.

`--json-out` writes the whole structure: media, both stages with every segment, word and
observation, the exact checked text, the shared `Report`, `claim_sources`, the merged duplicates and
timings. It is UTF-8 and not ASCII-escaped, and it round-trips through the Pydantic model (tested).

Precision of the times:

- **Speech times** are Whisper's estimates, accurate to roughly half a second. They are narrowed to
  the claim's own words when word timestamps exist.
- **On-screen times** are sampled-frame times. Text may have appeared up to one interval earlier
  and stayed up to one interval later.
- The **extraction confidence** values (speech and OCR) say how reliably words were *recognised*.
  They are separate from evidence strength and from the verdict.

### CPU, RAM and model choice

These figures were measured on this laptop (i5-12500H, 16 GB, CPU only, int8) with
`eval/run_video_eval.py`, over 11 synthetic clips (95 s of audio). RTF is transcription time divided
by audio length (lower is faster). RAM is the peak working set of the whole eval process, with
RapidOCR loaded:

| Model (multilingual) | Download | Load | RTF | Peak RAM | WER, clean TTS speech | WER, speech under heavy noise |
|---|---|---|---|---|---|---|
| tiny | ~75 MB | 0.5 s | 0.06 | not measured | 0.00 | not measured |
| **base (default)** | ~145 MB | 0.7 s | 0.11-0.26 (varied between runs) | ~545 MB | 0.00 | 1.40 (nonsense, flagged UNRELIABLE) |
| small | ~484 MB | 1.2-1.5 s | 0.33 | ~826 MB | 0.00 | 0.70 (still wrong, flagged UNRELIABLE) |

Base is the default because it is about 3x faster than small with less RAM, and synthetic clean
speech didn't separate the two. Use `--whisper-model small` (or `medium`, ~1.5 GB) for accents,
noise or non-English speech, where larger models are known to be much better. That claim is
**not measured here**: there is no non-English or real-world clip in the set. Frame sampling plus
OCR was the slower part on these short clips. It took 54.8 s in total for the 11 clips, about 3-8 s
per clip, even when only one frame needed OCR. Speed was not optimised further; expect several
minutes for a 10-minute video.

### Evaluation

```bash
.venv\Scripts\python.exe eval\make_video_samples.py
```
```bash
.venv\Scripts\python.exe eval\run_video_eval.py --models tiny,base,small
```
```bash
.venv\Scripts\python.exe eval\run_video_eval.py --live
```

`make_video_samples.py` generates the 11 clips. The speech comes from offline Windows SAPI voices and
the frames are drawn with Pillow, so there is no third-party media. The `--live` flag runs the real
fact-check on 5 claim cases (at most 12 Tavily credits each). Results with base, measured
2026-10-27 to 2026-10-29:

- **Extraction:** WER 0.00 on all clean speech. On-screen word recall 1.00, including a 9 px
  caption at 256x144.
- **Deduplication:** distinct on-screen observations 10/10 and speech/screen exact merges 11/11
  as expected. The claim rule on 40 labelled pairs had **precision 1.00** (0 wrong merges) and
  recall 0.58. The 5 missed merges were synonyms, paraphrases, reordering and added words, all
  checked twice by design.
- **Provenance:** 18/18 utterances or captions placed within 0.75 s / one interval of their true
  start.
- **Robustness:** 2/2 corrupt or non-video files rejected (exit 2). Silence and noise produced no
  speech in 5/5 cases. Heavy noise made Whisper hallucinate fluent text; it was flagged UNRELIABLE
  and would be withheld.
- **Live fact-check:** these are the per-case outcomes (Gemini 3.5 Flash-Lite + Tavily):

  | Case | Result |
  |---|---|
  | Contradicting speech and screen claims | Both checked: "not visible" SUPPORTED, "is visible" REFUTED |
  | Text with no audio | REFUTED |
  | Same claim spoken and subtitled | One unit, checked once, SUPPORTED on the rerun (first run INSUFFICIENT_EVIDENCE) |
  | Speech-only, 2 claims | Eiffel SUPPORTED; Everest SUPPORTED on the rerun (first run PROCESSING_FAILED) |
  | Different years | Speech "1889" SUPPORTED; screen "1887" INSUFFICIENT_EVIDENCE, not REFUTED |

  The "1887" claim was not mismatched by the video code: the existing validation rejected the
  refuting citations, and the text command gives the same INSUFFICIENT_EVIDENCE for that sentence.

None of these numbers says anything about real-world reliability. The clips are synthetic and
short, and the speech is clean text-to-speech.

**Real English videos (2026-11-17).** Three videos were each run more than once: an Elon Musk
biography (animated, 2 min 43 s), an Agni-V explainer (mostly Hindi speech, English infographics)
and a Usain Bolt short (English speech with burned-in subtitles). Observations:

- **Burned-in subtitles.** In the Bolt video, 28 subtitle fragments were attached to the speech.
  This removed a duplicate check of the 9.58 s claim and the excerpts the model had stitched
  together from several fragments.
- **The "$1.8 billion" on screen** (really "$1.5 billion" in tiny print) is now marked uncertain
  instead of being treated as a contradiction of the narration.
- **Run-to-run variation.** Verdicts for the same claim can differ between runs (for example,
  SUPPORTED vs INSUFFICIENT_EVIDENCE for Zip2's founding). So can which claims Gemini extracts:
  2 vs 5 from the same Agni video. Both come from the model and web search, not from extraction,
  which gave the same text each time.
- **Repairs the model shouldn't make.** Once, the extraction model wrote "Compaq" for the
  transcript's misheard "Compact", despite being told not to repair text.
- **Table figures (fixed 2026-11-17).** Claims taken from infographic tables (Agni-V payload
  1,360.78 kg, height 17 m) repeatedly ended as PROCESSING_FAILED. The judge cited genuine short
  table cells ("Length : 17.5 m", "Payload : 1,650 kg"). Validation counted them as fabricated
  quotes and discarded the whole judgment. Now only quotes that are not in the source count as
  fabricated. Short verbatim cells are accepted when they are labelled figures sharing a term with
  the claim, and otherwise rejected only as weak citations. Live re-check: height 17 m became
  SUPPORTED, and payload 1,360.78 kg became INSUFFICIENT_EVIDENCE (the judge's "1,500 kg"
  support was correctly refused).

### Limitations

- **Pictures are not interpreted.** Charts, maps and scenes are ignored, and so are claims made
  only through visuals.
- **Recognition confidence is a weak signal.** Whisper confidence and OCR confidence can be high
  for a wrong reading. Words that were never recognised (missed captions, muffled speech) cannot
  be flagged.
- **Frame sampling** at 2 s can miss text shown for less than 2 s. Use `--frame-interval 0.5` for
  fast captions.
- **Text limit.** Only the first `--max-text-chars` (default 6,000, at most 20,000) characters of
  extracted text are checked. Later segments are listed as not checked, with their start time. One
  video per run, at most 10 minutes and 500 MB by default (`--max-duration`, `--max-file-mb`). The
  shared limits apply: 6 checkable claims, 40 search credits, 180 s for the fact-check, plus 900 s
  for extraction (`--extract-timeout`).
- **Deduplication edge case.** If a claim from an unreliable source is merged with the same claim
  from a reliable one, the gate looks only at the claim that was kept. Merging is deliberately
  strict, so paraphrased duplicates cost extra searches.
- **Speaker identity** is not tracked, so a quoted or rebutted statement is checked like any other.
- **Not tested here:** non-English speech, real camera footage, rotated phone videos (rotation
  metadata is applied but untested), the PaddleOCR engines for video, and the Ollama fallback.
- **Licences.** faster-whisper, CTranslate2, Whisper and Systran's converted models are MIT;
  PyAV is BSD-3. PyAV's Windows wheel also bundles FFmpeg with GPL components (x264, x265). This
  project only decodes and doesn't redistribute them, but check the FFmpeg licence before
  redistributing a bundled build.

### Troubleshooting (video)

| Problem | Fix |
|---|---|
| `speech model 'base' is not downloaded` | Re-run once with `--allow-download`, or pass a local CTranslate2 model directory to `--whisper-model` |
| `PyAV is not installed` / `faster-whisper is not installed` | `pip install -e ".[video]"` |
| `not a readable video/audio file` | The file is corrupt, truncated or not a video. Try re-encoding it |
| `unsupported file extension` | Rename only if it really is a video container, or convert it to MP4 |
| `video is N s long; the limit is …` | Use `--max-duration`, or cut the video |
| Speech `nothing found` | Silence or music only. Try `--no-vad` if quiet speech was filtered out |
| Speech `UNRELIABLE` | Noisy audio. Try `--whisper-model small` and set `--language` |
| On-screen text missed | Use a smaller `--frame-interval`, or `--ocr-model-size medium --allow-download` |
| Hugging Face symlink warning on Windows | Harmless. The cache works without symlinks but uses more disk |
| `An Application Control policy has blocked this file` for a `.exe` launcher | Windows Smart App Control. Run `python -m factcheck_video` (or `python -m factcheck`) instead |

## Audio fact-checking (`factcheck-audio`)

The fourth command fact-checks what is said in an **English** audio file: a podcast, interview,
lecture or voice note. Speech is transcribed locally with faster-whisper, and the transcript goes
through the same shared fact-checking pipeline as `factcheck` (claim extraction, validation,
retrieval, judge, citation validation, deterministic verdicts). Only the transcription is new.

```bash
.venv\Scripts\python.exe -m pip install -e ".[audio]"
```
```bash
python -m factcheck_audio podcast.mp3
```
```bash
python -m factcheck_audio recording.m4a --extract-only
```
```bash
python -m factcheck_audio interview.wav --json-out report.json
```

`factcheck-audio …` (the installed launcher) does the same thing. `--extract-only` (alias
`--transcribe-only`) needs no API keys. The audio extra is a subset of the video extra (faster-whisper,
PyAV, numpy), so nothing new is installed if video already works. No new API or key is used.

### How it works

```
audio file -> validation (extension, size, readable, has audio, duration)
           -> PyAV decode -> 16 kHz mono WAV (temporary, always deleted)
           -> faster-whisper: language identification -> English only -> timed segments (+ word times)
           -> transcript as plain prose (no timestamps in the text)
           -> shared run_pipeline (audio note, conservative duplicate rule, transcription-reliability gate)
           -> each claim traced back to its segment(s), start/end time and transcription confidence
```

* **Formats.** MP3, WAV, M4A/AAC, FLAC, OGG and Opus are tested; WMA, AIFF, AMR, MKA, CAF, AC3 and
  MP2 are accepted too (anything the bundled FFmpeg decodes). Video files are refused with a pointer
  to `factcheck-video`. Embedded cover art is ignored.
* **English only.** Whisper identifies the language over up to ~90 s of speech before decoding. If
  it is not English, nothing is decoded and the report says so (exit 1). Non-English speech is never
  translated or force-decoded into English words that weren't said. `--assume-english` skips
  identification, for accented English that Whisper misidentifies. Any segment that still comes out
  in a non-Latin script is marked unreliable and never checked. A `*.en` model (e.g. `base.en`)
  skips identification.
* **Code reuse, not duplication.** Model loading (local cache only unless `--allow-download`),
  segment confidence, sentence splitting, the WAV decoder and the duplicate rule come from
  `factcheck_video` unchanged. The video module doesn't import anything from the audio module, and
  the audio module never loads OCR or the video pipeline.
* **Claims.** The audio note tells the extractor to skip greetings, fillers, questions, opinions,
  predictions and promotional calls to action. Fragments and garbled passages are classed VAGUE. It
  must copy words, names and numbers exactly as transcribed. The shared validation still drops
  invented excerpts and changed figures. A claim repeated in the recording is checked once, and the
  report lists every place it was said. Claims that differ in a number, date or negation are never
  merged.
* **Transcription confidence ≠ factual confidence.** Confidence is Whisper's own estimate of how
  well it heard the words. A claim whose every source segment is below `--speech-threshold`
  (default 0.40) is withheld (PROCESSING_FAILED, "may have been misheard") and never searched.
  Confidence never enters the verdict or the evidence-strength score.

### Traceability

Every claim in the terminal report ends with a line like
`Spoken ~00:17.8-00:22.6 (word timing), segment 5, transcription confidence 0.94`. The JSON report
has `traced_claims`, one entry per claim holding:
- the statement, the transcript excerpt, start/end times and segment ids;
- the full segment transcript and its transcription confidence;
- how many times it was said;
- the verdict and evidence strength;
- the supporting and contradicting sources with their validated quotes, and how many citations
  validation rejected.

It also has the audio metadata, the language decision, every segment (with word timings), the exact
text sent to claim extraction, segments left out by `--max-text-chars`, timings and the full shared
fact-check report.

### CPU defaults (i5-12500H, 16 GB RAM, CPU only)

`base` int8 is the default. On the real 163 s English recording it ran at 0.06× real time (about
9 s) and used under 1 GB RAM. `small` ran at 0.15× and agreed with `base` to within 4% WER. Thread
counts were measured: the library default (`--threads 0`) was fastest (9.0 s), ahead of 4 threads
(9.3 s), 8 (10.5 s) and 12 (11.4 s). Default limits are 60 min (`--max-duration`), 500 MB
(`--max-file-mb`) and 30 min for decoding plus transcription (`--extract-timeout`). The fact-check
reads the first 6,000 characters of transcript (about 6–7 minutes of speech; `--max-text-chars`, up
to 20,000). Later segments are listed as not checked.

### Evaluation (`eval/run_audio_eval.py`, measured 2026-11-18)

| Set | Result (base) |
|---|---|
| 6 synthetic English cases (Windows voices; mp3/wav/flac/m4a/ogg/opus; numbers, dates, negation, fast speech, 10 dB noise, 20 s pause) | WER 0.000 after formatting normalisation (raw 0.094: "8,849"/"8849", "meters"/"metres", "$1.5 billion"/"1.5 billion dollars"); numbers 9/9; English identified 6/6; segment start within 0.56 s (mean 0.37 s) |
| Silence, white noise, tones | 0 hallucinated segments (3/3) |
| Real English (two of the user's videos, 163 s + 60 s) | English p ≥ 0.998; 100% of segments reliable; base vs small agreement WER 0.042 and 0.010 (a consistency proxy, since no human transcript exists) |
| Real Hindi (Agni video) | Rejected as not English in 0.8 s, nothing decoded |
| Live fact-check of the 163 s English recording | 6 claims checked with times, segments and sources: 3 SUPPORTED, 2 INSUFFICIENT EVIDENCE, 1 CONFLICTING (Queen's University 1989 vs Wikipedia's 1990); 3 more over the per-run claim limit |

Offline tests: `tests/test_audio_*.py` (formats, corrupt/empty/truncated files, limits, silence and
noise, the language policy, timestamps and word timing, claim safeguards, repeats, numbers/dates/
negations, the reliability gate, API failures, JSON, CLI exit codes, module separation, regressions
for the other three commands). Opt-in real-Whisper tests:
`set FACTCHECK_AUDIO_LIVE_TESTS=1` then `python -m pytest tests/test_audio_live.py`.

### Limitations

- Mixed-language recordings: the language is identified once, from the opening speech. An English
  recording with a non-English stretch is transcribed as English throughout, apart from the
  non-Latin-script guard.
- No speaker identification, so a quoted or rebutted statement is checked like any other.
- Long recordings are only partly fact-checked (the character limit). The whole transcript is still
  shown and saved.
- Synthetic voices are cleaner than real recordings. Real-audio accuracy was measured only as
  agreement between two Whisper sizes, not against a human transcript.

| Problem | Fix |
|---|---|
| `the detected language is …; only English is supported` | The recording isn't English, or use `--assume-english` for misidentified accented English |
| `looks like a video file` | Use `factcheck-video` |
| `not a readable audio file or it is corrupted` | Re-encode the file (e.g. to MP3 or WAV) |
| `contains no decodable audio samples` | The file is empty |
| Transcription `no speech found` | Silence or music only; try `--no-vad` if quiet speech was filtered out |
| Many `?` segments | Noisy audio: try `--whisper-model small` |
| `.exe` blocked by Application Control | Use `python -m factcheck_audio` |

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
