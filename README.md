# AGENTX

**An autonomous browser agent with LLM-based planning, real Playwright execution, multi-method verification, and self-correction — backed by a SQLite memory system that lets it learn from its own history.**

Solo-developer portfolio project. Zero paid infrastructure. One SQLite file. No Redis, no VectorDB, no Kubernetes, no microservices — every component here exists because it demonstrably improves task completion, recovery rate, or verification accuracy, not because it looks impressive on a diagram.

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

## What this actually is

You give it a goal like:

```bash
python main.py "Go to en.wikipedia.org/wiki/Python_(programming_language) and return the year Python was first released"
```

It plans a sequence of browser steps with an LLM, executes them with a real headless Chromium browser, verifies each step actually succeeded (not just "no exception was thrown"), and — if a step fails — classifies *why* it failed and picks a recovery strategy instead of just crashing. Every action it takes, successful or not, is written to SQLite so future runs can plan better.

---

## Why it's built this way

The full design rationale — including a line-by-line list of what was deliberately **not** built (Postgres, Redis, VectorDB, Celery, session pooling, Kubernetes, etc.) and why — lives in [`AGENTX — Architecture Design Document`](./AGENTX___Architecture_Design_Document). Two rules from that document govern every decision in this codebase:

1. **Every component maps to a concept an AI/agent-systems interviewer would ask about.** If it doesn't, it's cut.
2. **Nothing gets added unless it moves a measurable number** — task success rate, recovery rate, or verification accuracy. Complexity that doesn't move a benchmark is complexity that shouldn't exist.

---

## Architecture

### The pipeline

1. **Planner** (`core/planner.py`) — Converts the goal into a JSON step list via the LLM. Before calling the LLM, it queries Memory for similar past tasks and tools that worked for this goal type, and injects that as few-shot context. No embeddings — plain SQL keyword matching against `tasks.goal LIKE '%keyword%'` and `action_successes.goal_type = ?`. On failure mid-run, `replan()` regenerates the remaining steps from the current page state.

2. **Execution Engine** (`core/execution_engine.py`) — Walks the `ExecutionDAG` sequentially, resolves each step's tool from the static `TOOL_REGISTRY`, dispatches to the Browser Controller, and hands the result to the Verifier. On `FAIL`, control passes to the Self-Correction Engine; on `REPLAN`, the remaining DAG is swapped and execution continues from the failure point.

3. **Browser Controller** (`browser/controller.py`, `browser/actions.py`) — A Playwright facade. Nothing outside `browser/` imports Playwright directly. Exposes `navigate`, `click`, `click_text`, `type`, `scroll`, `extract`, `extract_page`, `get_links`, `get_dom_snapshot`, `screenshot`. Every action returns an `ActionResult` — nothing raises across the boundary.

4. **Verification Engine** (`verification/`) — Chain of Responsibility, cheapest-first:
   - **DOM Verifier** — URL/title/structural checks, zero token cost.
   - **Text Verifier** — keyword/failure-signal matching against extracted page text, zero token cost.
   - **LLM Verifier** — only called when the first two return `UNCERTAIN`. Asks the LLM "did this step achieve its goal?" and parses a `{result, confidence, reason}` JSON verdict.
   
   All-`UNCERTAIN` is treated as `FAIL` — the system never silently assumes success.

5. **Self-Correction Engine** (`correction/`) — Classifies the failure (`stale_selector`, `timeout`, `auth_wall`, `wrong_page`, `extraction_empty`, `plan_error`, `unknown`) and walks a per-failure-type priority list of strategies, skipping any already tried in this task session (per `action_failures` history):
   - `RETRY` — re-run the exact step (transient failures).
   - `SELECTOR_FIX` — ask the LLM for an alternative CSS selector using live DOM/HTML context, with an explicit rule that it cannot return the same broken selector.
   - `REPLAN` — call `Planner.replan()` with the current page state.
   - `ABORT` — mark the task failed with a clear reason (auth walls, CAPTCHAs, exhausted retries).

6. **Memory** (`memory/`) — One SQLite file, three read/write stores plus a composite retrieval layer:
   - `task_memory.py` — full task lifecycle records.
   - `action_memory.py` — successful tool invocations, scoped by goal type and site domain, used to bias future plans.
   - `failure_memory.py` — every failure + correction attempt + outcome, scoped per task, used to prevent retrying a strategy that already failed in this session.
   - `retrieval.py` — the only interface the Planner and Correction Engine use; they never touch raw SQL.

7. **LLM abstraction** (`llm/`) — `LLMProvider` ABC with a single `complete()` entry point. `llm/factory.py` picks the concrete provider from `LLM_PROVIDER` in `.env`. **Ollama + Qwen3, running locally, is the default and only zero-cost, zero-API-key option** — this is intentional so the whole project runs for $0. Anthropic, OpenAI, and Groq are drop-in alternatives behind the same interface; switching providers is a one-line `.env` change, never a code change.

### Design patterns in use

| Pattern | Where |
|---|---|
| Strategy | Correction strategies (`retry`, `selector_fix`, `replan`, `abort`) are swappable functions |
| Chain of Responsibility | DOM → Text → LLM verifiers |
| Repository | Every memory module exposes only named functions — no raw SQL leaks outside `memory/` |
| Facade | `BrowserController` hides all of Playwright |
| Command | Every browser action is a discrete, logged, independently-testable function |
| Factory | `tools/registry.py` resolves tool name → handler; `llm/factory.py` resolves provider |

---

## Database

One SQLite file (`db/agentx.db`, gitignored), five tables, `CREATE TABLE IF NOT EXISTS` on startup — no ORM, no migrations:

- **`tasks`** — one row per goal run: status, plan JSON, result, tokens, steps, corrections.
- **`steps`** — one row per executed step: tool, input/output JSON, status, retry count.
- **`action_successes`** — successful tool invocations, keyed by goal type + site domain, read by the Planner.
- **`action_failures`** — every failure + correction attempt + whether it worked, read by the Self-Correction Engine.
- **`benchmark_results`** — per-task benchmark run output (schema exists; runner not yet built).

---

## Getting started

### Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally (default LLM provider — free, no API key)
- `playwright install chromium` after installing dependencies

### Install

```bash
pip install -r requirements.txt
playwright install chromium

# Pull the default model
ollama serve &
ollama pull qwen3:latest
```

Copy `.env.example` to `.env` and adjust if needed — defaults work out of the box with local Ollama.

### Run a task

```bash
python main.py "Go to news.ycombinator.com and return the title of the top post"
```

### Check system health

```bash
python main.py --health
```

### Smoke-test the browser layer only

```bash
python main.py --browse "https://en.wikipedia.org/wiki/Python_(programming_language)"
```

### Start the API

```bash
python main.py --serve
# then: curl -H "X-API-Key: dev-key-change-in-production" http://127.0.0.1:8000/health
```

---

## Configuration

All configuration is centralized in `config/settings.py` (Pydantic `BaseSettings`, reads `.env`). Nothing in the codebase calls `os.environ` directly. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `anthropic` \| `openai` \| `groq` |
| `LLM_MODEL` | `qwen3:latest` | Model string passed to the active provider |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `BROWSER_HEADLESS` | `true` | Set `false` to watch the agent work |
| `MAX_RETRIES_PER_STEP` | `3` | Retry ceiling before escalating |
| `MAX_CORRECTIONS_PER_TASK` | `6` | Hard abort ceiling per task |
| `DB_PATH` | `db/agentx.db` | SQLite file location |

---

## Benchmark suite

`evaluation/benchmarks/dataset.json` defines 25 tasks across 5 categories, each with a goal, expected-output checks, and step/correction/timeout budgets:

- **Navigation** (5, easy) — basic browser control and extraction.
- **Data Extraction** (5, easy–medium) — selector accuracy against known pages (e.g. Wikipedia release years, ground-truth checkable).
- **Form Interaction** (5, medium) — multi-action sequencing, login flows, click-throughs.
- **Multi-Step** (5, hard) — 5+ sequential actions with intermediate state (cheapest book in a category, etc.).
- **Error Recovery** (5, hard) — intentionally unstable targets (dynamic search layouts, pagination edges, sort interactions) designed to force the Self-Correction Engine to actually fire.

The runner, metrics calculator, and report generator that consume this dataset (`evaluation/benchmark_runner.py`, `metrics.py`, `reporter.py`) are the next piece to build — `/v1/benchmark` is currently a `501` stub. Once built, a run produces a report like:

```
Task Success Rate        72%   (18/25)
Recovery Rate            81%   (13/16 corrections)
Avg Steps / Task          6.4
Avg Corrections / Task    0.64
```

This report — not a subjective description of "the agent works" — is what actually validates the system.

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
├── llm/                     # provider-agnostic LLM interface (Ollama default)
├── config/                  # Pydantic settings, single source of truth
├── log/                     # structured JSON logger, stdlib only
├── evaluation/
│   └── benchmarks/dataset.json   # 25-task benchmark suite
└── db/                       # agentx.db (gitignored)
```

---
