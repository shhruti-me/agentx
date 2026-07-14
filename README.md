# AGENTX

**An autonomous browser agent with LLM-based planning, real browser execution, multi-method verification, and self-correction — backed by a SQLite memory system that improves planning over time.**

AGENTX accepts a natural language goal, decomposes it into an executable plan, drives a real browser to carry it out, verifies each step actually succeeded, and recovers from failures by classifying what went wrong and selecting a correction strategy. Every action — success or failure — is persisted, and future plans are biased toward what has already worked.

```
Goal (natural language)
        │
        ▼
   Planner (LLM) ──reads──> Memory (SQLite)
        │
        ▼
ExecutionDAG (ordered steps)
        │
        ▼
  Execution Engine ──> Browser Controller (Playwright)
        │
        ▼
 Verification Engine (DOM → Text → LLM, cheapest first)
        │
   ┌────┴────┐
  PASS      FAIL
   │          │
   │          ▼
   │   Self-Correction Engine
   │   (retry / selector_fix / replan / abort)
   │          │
   └────┬─────┘
        ▼
  Memory writes (successes + failures)
        │
        ▼
   Final Result
```

---

## How it works

Give it a goal:

```bash
python main.py "Go to en.wikipedia.org/wiki/Python_(programming_language) and return the year Python was first released"
```

The **Planner** converts that into a JSON step list via an LLM call, first querying memory for similar past tasks and tools that have worked for this goal type and injecting that as few-shot context. No embeddings — plain SQL keyword matching (`tasks.goal LIKE '%keyword%'`, `action_successes.goal_type = ?`) is enough to bias planning toward known-working approaches.

The **Execution Engine** walks the resulting `ExecutionDAG` step by step, resolves each step's tool from a static registry, and dispatches it to the **Browser Controller** — a Playwright facade that exposes `navigate`, `click`, `click_text`, `type`, `scroll`, `extract`, `extract_page`, `get_links`, `get_dom_snapshot`, and `screenshot`. Nothing outside `browser/` ever imports Playwright directly, and every action returns an `ActionResult` rather than raising across the boundary.

Each step result is checked by the **Verification Engine**, a cost-ordered Chain of Responsibility:

1. **DOM Verifier** — structural checks (URL changed, title matches, element appeared/disappeared). Zero token cost.
2. **Text Verifier** — keyword and failure-signal matching against extracted page text. Zero token cost.
3. **LLM Verifier** — only invoked when the first two return `UNCERTAIN`. Sends the expected outcome and page content to the LLM and asks it to judge pass/fail with a confidence score.

`UNCERTAIN` from all three is treated as `FAIL` — the system never silently assumes success.

On failure, the **Self-Correction Engine** classifies the failure type (`stale_selector`, `timeout`, `auth_wall`, `wrong_page`, `extraction_empty`, `plan_error`, `unknown`) and walks a per-failure-type priority list of strategies, skipping anything already tried in this task session:

- **`RETRY`** — re-run the exact step. Transient failures only.
- **`SELECTOR_FIX`** — ask the LLM for an alternative CSS selector using live DOM/HTML context, with an explicit constraint that it cannot return the same broken selector.
- **`REPLAN`** — call `Planner.replan()` with the current page state to regenerate the plan from the failure point forward.
- **`ABORT`** — mark the task failed with a clear reason. Used for auth walls, CAPTCHAs, and exhausted retries.

Every success and failure is written to SQLite, closing the loop back into the Planner and Correction Engine for the next run.

---

## Architecture

### Core components

| Component | Responsibility |
|---|---|
| **Orchestrator** (`core/orchestrator.py`) | Top-level pipeline coordinator: intake → plan → execute → report. Owns task lifecycle only. |
| **Planner** (`core/planner.py`) | Goal → `ExecutionDAG`. Retrieval-augmented prompting, structured JSON parsing, replan-on-failure. |
| **Execution Engine** (`core/execution_engine.py`) | DAG traversal, step dispatch, retry state, step timing. |
| **Browser Controller** (`browser/controller.py`, `browser/actions.py`) | Playwright session lifecycle and action dispatch behind a clean facade. |
| **Tool Registry** (`tools/registry.py`) | Static name → handler mapping. The Planner reads tool descriptions to construct valid steps. |
| **Verification Engine** (`verification/`) | Three-method chain producing `PASS` / `FAIL` / `UNCERTAIN` with confidence and reasoning. |
| **Self-Correction Engine** (`correction/`) | Failure classification, strategy selection, recovery execution. |
| **Memory** (`memory/`) | SQLite-backed task, action-success, and action-failure stores plus a composite retrieval layer. |
| **LLM Layer** (`llm/`) | Provider-agnostic interface (`LLMProvider` ABC) with pluggable backends. |
| **Logger** (`log/logger.py`) | Structured JSON logging to stdout and rotating file, stdlib only. |
| **API** (`api/`) | FastAPI surface: `/v1/run`, `/v1/status/{id}`, `/v1/results/{id}`, `/v1/tasks`, `/health`. |

### Design patterns

| Pattern | Where applied |
|---|---|
| **Strategy** | Correction strategies (`retry`, `selector_fix`, `replan`, `abort`) are independently swappable functions |
| **Chain of Responsibility** | DOM → Text → LLM verifiers, ordered cheapest-first |
| **Repository** | Every memory module exposes only named functions — no raw SQL leaks outside `memory/` |
| **Facade** | `BrowserController` hides the full Playwright API behind a minimal interface |
| **Command** | Every browser action is a discrete, named, independently-testable operation |
| **Factory** | `tools/registry.py` resolves tool name → handler; `llm/factory.py` resolves LLM provider |

### Data model

Core dataclasses in `core/models.py`: `Task`, `Step`, `ExecutionDAG`, `ActionResult`, `VerificationResult`, `CorrectionResult`, plus the `TaskStatus`, `StepStatus`, `GoalType`, `VerificationStatus`, `FailureType`, and `CorrectionStrategy` enums. Every module in the system imports its types from here — no module defines its own task or step shape.

---

## Database

One SQLite file, five tables, created via `CREATE TABLE IF NOT EXISTS` on startup:

- **`tasks`** — one row per goal run: status, plan JSON, result, token usage, step/correction counts.
- **`steps`** — one row per executed step: tool, input/output JSON, status, retry count.
- **`action_successes`** — successful tool invocations keyed by goal type and site domain, read by the Planner before generating a plan.
- **`action_failures`** — every failure and correction attempt with outcome, read by the Self-Correction Engine to avoid retrying a strategy that already failed in this session.
- **`benchmark_results`** — per-task benchmark output: status, steps, corrections, tokens, timing.

Retrieval is plain SQL wrapped in named Python functions — no ORM, no query builder.

---

## LLM layer

Every LLM call goes through `LLMProvider.complete()` — no component outside `llm/` imports an HTTP client or a provider SDK directly. `llm/factory.py` resolves the concrete provider from configuration at startup, so switching providers is a config change, not a code change.

Providers: **Ollama** (local, default), **Anthropic**, **OpenAI**, **Groq**. All implementations conform to the same `LLMResponse` shape (`content`, `input_tokens`, `output_tokens`, `model`, `provider`, `latency_ms`), which is what makes cost tracking and benchmark reporting provider-agnostic.

---

## Getting started

### Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally
- Playwright's Chromium binary

### Install

```bash
pip install -r requirements.txt
playwright install chromium

ollama serve &
ollama pull qwen3:latest
```

Copy `.env.example` to `.env` and adjust as needed.

### Run a task

```bash
python main.py "Go to news.ycombinator.com and return the title of the top post"
```

### Check system health

```bash
python main.py --health
```

### Smoke-test the browser layer

```bash
python main.py --browse "https://en.wikipedia.org/wiki/Python_(programming_language)"
```

### Start the API

```bash
python main.py --serve
curl -H "X-API-Key: dev-key-change-in-production" http://127.0.0.1:8000/health
```

---

## Configuration

All configuration is centralized in `config/settings.py` (Pydantic `BaseSettings`, reads `.env`). No module calls `os.environ` directly.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `anthropic` \| `openai` \| `groq` |
| `LLM_MODEL` | `qwen3:latest` | Model string passed to the active provider |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `BROWSER_HEADLESS` | `true` | Set `false` to watch the agent drive the browser |
| `MAX_RETRIES_PER_STEP` | `3` | Retry ceiling before escalating to REPLAN/ABORT |
| `MAX_CORRECTIONS_PER_TASK` | `6` | Hard correction ceiling per task |
| `DB_PATH` | `db/agentx.db` | SQLite file location |
| `LOG_LEVEL` | `INFO` | Structured JSON log verbosity |

---

## Evaluation

`evaluation/benchmarks/dataset.json` defines 25 tasks across 5 categories, each with a goal, expected-output checks, and step/correction/timeout budgets:

- **Navigation** — basic browser control and DOM extraction.
- **Data Extraction** — selector accuracy against pages with verifiable ground truth (e.g. release years, prices).
- **Form Interaction** — multi-action sequencing: filling inputs, clicking through, verifying result pages.
- **Multi-Step** — 5+ sequential actions with intermediate state (comparing prices across a category, pagination).
- **Error Recovery** — tasks with intentional instability (dynamic layouts, sort interactions, pagination edges) that require the Self-Correction Engine to actually fire.

A benchmark run produces a report keyed on task success rate, recovery rate, average steps/corrections per task, and a breakdown of failure types and correction effectiveness — the empirical basis for any claim about how well the agent performs.

```
Task Success Rate        72%   (18/25)
Recovery Rate            81%   (13/16 corrections)
Avg Steps / Task          6.4
Avg Corrections / Task    0.64
```

---

## Project layout

```
agentx/
├── main.py                  # CLI: run a goal, --serve, --health, --browse
├── api/                     # FastAPI app + routes
├── core/                    # models, orchestrator, planner, execution engine
├── browser/                 # Playwright controller, actions, text/DOM extractor
├── tools/                   # static tool registry + browser tool handlers
├── verification/            # DOM / text / LLM verifiers + orchestrator
├── correction/               # failure classifier, engine, 4 recovery strategies
├── memory/                  # SQLite connection + task/action/failure stores + retrieval
├── llm/                     # provider-agnostic LLM interface
├── config/                  # Pydantic settings, single source of truth
├── log/                     # structured JSON logger, stdlib only
├── evaluation/
│   └── benchmarks/dataset.json   # 25-task benchmark suite
└── db/                       # agentx.db
```
