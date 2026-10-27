# factcheck: agent workflow

How a paragraph becomes a fact-check report. The two LangGraph diagrams (sections 2 and 3) were
exported from the compiled graphs in `src/factcheck/graph.py` with
`graph.get_graph().draw_mermaid()`. Only the edge labels were added by hand. The diagrams render on
GitHub, in VS Code with a Mermaid extension, or in `docs/workflow.html` (open it in a browser).

## 1. End-to-end overview

```mermaid
flowchart TD
    U["User text<br/>(CLI argument, --file or stdin)"] --> V["validate_input<br/>length & content limits"]
    V --> X["extract_claims<br/>1 LLM call: atomic claims + search queries"]
    X --> C["validate_claims<br/>excerpt anchored in input, no invented figures,<br/>dedupe, opinions -> NOT_CHECKABLE, max 6 claims"]
    C -->|"one Send() per checkable claim<br/>(parallel, max_concurrency = 3)"| W["claim_worker<br/>per-claim subgraph (section 3)"]
    C -->|"no checkable claims"| S
    W --> S["summarize<br/>merge results in claim order"]
    S --> R["render_report<br/>deterministic overall label"]
    R --> T["Terminal report<br/>(normal / -v verbose)"]
    R --> J["--json-out report.json<br/>(complete structured data)"]

    subgraph EXT ["External services"]
        G["Gemini (primary LLM)"]
        O["Ollama qwen3:4b (local fallback LLM)"]
        TV["Tavily (web search + page extraction)"]
    end
    X -.-> G
    W -.-> G
    W -.-> TV
    G -.->|"recoverable failure:<br/>429 / 5xx / timeout / network"| O
```

## 2. Main graph (LangGraph `build_graph`)

```mermaid
flowchart TD
    start([start]) --> validate_input
    validate_input --> extract_claims
    extract_claims --> validate_claims
    validate_claims -.->|"Send per checkable claim"| claim_worker
    validate_claims -.->|"nothing to check"| summarize
    claim_worker --> summarize
    summarize --> render_report
    render_report --> finish([end])
```

## 3. Per-claim subgraph (LangGraph `build_claim_graph`)

```mermaid
flowchart TD
    start([start]) --> retrieve
    retrieve["retrieve<br/>Tavily: up to 2 queries x 5 results,<br/>extract top 5 pages (credit budget)"] --> process_evidence
    process_evidence["process_evidence<br/>sanitise, chunk <= 1200 chars, rank,<br/>dedupe URLs, cluster sources, assign evidence IDs"] --> sufficiency_gate
    sufficiency_gate{"sufficiency_gate"}
    sufficiency_gate -.->|"weak evidence and fallback round unused<br/>(at most 1 extra round)"| retrieve
    sufficiency_gate -.->|"enough passages"| judge
    sufficiency_gate -.->|"all searches failed -> RETRIEVAL_FAILED<br/>no relevant passages -> INSUFFICIENT_EVIDENCE<br/>budget / deadline -> PROCESSING_FAILED"| finalize
    judge["judge<br/>1 LLM call: stance + verbatim quote per passage<br/>(untrusted content delimited, no tools)"] --> validate_judgment
    validate_judgment{"validate_judgment<br/>IDs exist, quotes verbatim, on-topic,<br/>figures present, numeric comparisons"}
    validate_judgment -.->|"malformed / fabricated citations<br/>(at most 1 repair)"| judge
    validate_judgment -.->|"valid, or still invalid -> PROCESSING_FAILED"| finalize
    finalize["finalize<br/>deterministic verdict + correction check + evidence-strength score"] --> finish([end])
```

### Deterministic verdict rules (inside `finalize`)

```mermaid
flowchart TD
    A["Validated citations only<br/>S = support, R = refutation, M = missing context<br/>(noisy-OR over independent source clusters)"] --> B{"S >= 0.5 and R >= 0.5?"}
    B -->|yes| CF[CONFLICTING]
    B -->|no| C{"R >= 0.5?"}
    C -->|yes| RF[REFUTED]
    C -->|no| D{"S >= 0.5 and M >= 0.5?"}
    D -->|yes| ML[MISLEADING]
    D -->|no| E{"S >= 0.5?"}
    E -->|yes| SP[SUPPORTED]
    E -->|no| IE[INSUFFICIENT_EVIDENCE]
```

The model's own proposed verdict is recorded, but it never decides the outcome.

## 4. LLM provider chain and fallback (`llm.FallbackLLM`)

```mermaid
flowchart TD
    Q["LLM request<br/>(extract_claims or judge)"] --> BUD{"Run deadline left and<br/>LLM call budget available?"}
    BUD -->|no| FAIL1["LLMUnavailable -> claim PROCESSING_FAILED<br/>(extraction: run failure, exit 3)"]
    BUD -->|yes| DOWN{"Gemini already marked<br/>unavailable this run?"}
    DOWN -->|no| GEM["Gemini<br/>JSON schema, bounded retries (429/5xx/connect)"]
    DOWN -->|yes| OLL
    GEM -->|success| OK["Structured output -> Pydantic + validation"]
    GEM -->|"auth / model access / other 4xx"| STOP["Stop run (exit 3)<br/>no fallback for bad credentials"]
    GEM -->|"malformed output"| REP["Graph repair path (max 1)"]
    GEM -->|"recoverable: 429, 5xx, timeout, network"| MARK["Mark Gemini unavailable for this run"]
    MARK --> OLL["Ollama /api/chat<br/>format = JSON schema, think = false,<br/>one request at a time, deadline-clipped"]
    OLL -->|success| OK
    OLL -->|"server down / model missing /<br/>timeout / invalid output"| FAIL2["Combined error naming both providers"]
```

## Step reference

| Step | Kind | What it does | Limits |
|---|---|---|---|
| `validate_input` | code | Cleans control and bidi characters; rejects empty or too-long input | 4,000 characters |
| `extract_claims` | LLM | Splits text into atomic claims with exact excerpts and search queries | 1 call (+1 retry if malformed) |
| `validate_claims` | code | Drops claims with invented excerpts or figures, instructions and duplicates; marks opinions and predictions NOT_CHECKABLE | max 6 checkable claims |
| `claim_worker` | graph | Runs the per-claim subgraph for each claim in parallel | concurrency 3, recursion limit 25 |
| `retrieve` | Tavily | Searches (and one fallback round with other queries if evidence is weak); extracts full pages | 40 credits per run |
| `process_evidence` | code | Sanitises and chunks; ranks; canonical URL dedupe; independence clusters; evidence IDs | 8 passages per claim |
| `sufficiency_gate` | code | Chooses fallback round, judge or finish; separates technical failure from no evidence | — |
| `judge` | LLM | Assesses each passage: stance, directness, verbatim quote, numeric comparison | 1 call (+1 repair) |
| `validate_judgment` | code | Rejects unknown IDs, non-verbatim or off-topic quotes, missing figures, mismatched measurements | — |
| `finalize` | code | Verdict rules, grounded correction, 0–100 evidence-strength score with caps | — |
| `summarize` / `render_report` | code | Overall label, terminal report, JSON | — |

Every run is bounded by a 180 s deadline, a 14-call LLM budget, a 40-credit search budget and a
single fallback/repair per claim. Each limit stops work cleanly and marks unfinished claims
`PROCESSING_FAILED`.
