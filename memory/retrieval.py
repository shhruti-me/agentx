"""
memory/retrieval.py

Composite memory queries for the Planner and Self-Correction Engine.

This is the only memory interface those two components use.
They never import task_memory, action_memory, or failure_memory directly.

Why a separate retrieval module:
  - Planner and Correction Engine call memory with intent
    ("give me context for planning"), not with table knowledge
    ("query action_successes WHERE goal_type = ?").
  - If the schema changes, only this file needs updating.
  - Composite queries that span multiple tables live here,
    not scattered across callers.

Functions
---------
    get_planning_context(goal, goal_type)  → PlanningContext
    get_correction_context(tool, task_id) → CorrectionContext

TOKEN BUDGET
------------
get_planning_context() enforces a hard cap of MEMORY_CONTEXT_TOKEN_BUDGET
tokens on all data it returns. This prevents prompt bloat from growing
unboundedly as the benchmark accumulates task history.

Root cause of the fix:
  - nav_002  failed with Groq 429 (rate limit) — caused by oversized prompts
              consuming the entire per-minute token allowance per call.
  - extract_002 failed with Groq 413 (payload too large) — memory context
              grew to 26,044 tokens by task 7 because output_json blobs
              from prior extractions were injected in full.

Fix: trim each record before it enters PlanningContext. The Planner's
_build_prompt() is unchanged — it already formats PlanningContext correctly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memory.action_memory import get_recent_successes
from memory.failure_memory import get_failure_history
from memory.task_memory import get_similar_tasks


# ── Token budget constants ────────────────────────────────────────────────────

# Hard cap on total tokens the memory context may contribute to any prompt.
# At ~4 chars/token this is ~3,200 characters — safe for all providers.
# Groq free tier, Ollama/Qwen3, OpenAI, Anthropic all handle this comfortably.
MEMORY_CONTEXT_TOKEN_BUDGET: int = 800

# Per-record char limits prevent a single large blob eating the whole budget.
_MAX_CHARS_PER_TASK_RECORD: int = 200
_MAX_CHARS_PER_SUCCESS_RECORD: int = 300


# ── Return types ──────────────────────────────────────────────────────────────


@dataclass
class PlanningContext:
    """
    All memory context the Planner needs before generating a plan.

    The Planner serialises this into its prompt as few-shot examples.

    Fields
    ------
    similar_tasks     : Past completed tasks with similar goals.
                        Tells the Planner what worked end-to-end.
                        Records are pre-trimmed to _MAX_CHARS_PER_TASK_RECORD.

    recent_successes  : Recent successful tool invocations for this
                        goal type. Tells the Planner which selectors
                        and approaches have worked on similar pages.
                        output_json is pre-truncated to 120 chars.

    has_prior_context : True if any memory was found. Lets the Planner
                        decide whether to include a memory section in
                        its prompt or skip it entirely.
    """

    similar_tasks: list[dict] = field(default_factory=list)
    recent_successes: list[dict] = field(default_factory=list)

    @property
    def has_prior_context(self) -> bool:
        return bool(self.similar_tasks or self.recent_successes)


@dataclass
class CorrectionContext:
    """
    All memory context the Self-Correction Engine needs when a step fails.

    Fields
    ------
    failure_history          : All correction attempts for this tool in
                               this task. Used to skip already-tried strategies.
    already_tried            : Set of strategy names already attempted.
                               Convenience accessor over failure_history.
    previous_recovery_worked : True if a prior correction for this tool
                               in this task eventually succeeded. Hints that
                               the site is navigable — keep trying.
    attempt_count            : Total correction attempts so far.
    """

    failure_history: list[dict] = field(default_factory=list)

    @property
    def already_tried(self) -> set[str]:
        return {r["correction_used"] for r in self.failure_history}

    @property
    def previous_recovery_worked(self) -> bool:
        return any(r["was_recovered"] for r in self.failure_history)

    @property
    def attempt_count(self) -> int:
        return len(self.failure_history)


# ── Public interface ──────────────────────────────────────────────────────────


def get_planning_context(goal: str, goal_type: str) -> PlanningContext:
    """
    Retrieve all memory context needed before planning a task.

    Enforces MEMORY_CONTEXT_TOKEN_BUDGET across all returned data.
    Records are trimmed before being placed in PlanningContext so that
    the Planner's prompt stays within provider token limits regardless
    of how many tasks have previously run.

    Called by the Planner at the start of plan() and replan().

    Parameters
    ----------
    goal      : The raw goal string. Keywords are extracted from it
                to search for similar past tasks.
    goal_type : The classified goal category (navigation, extraction, etc.)

    Returns
    -------
    PlanningContext with similar_tasks and recent_successes populated.
    Both lists may be empty if the agent has no relevant history yet.
    All records are pre-trimmed to fit within the token budget.
    """
    keywords = _extract_keywords(goal)
    budget: int = MEMORY_CONTEXT_TOKEN_BUDGET

    # ── 1. Similar past tasks ─────────────────────────────────────────────────
    # Fetch raw, deduplicate, then trim each record to budget.
    raw_similar: list[dict] = []
    for keyword in keywords[:2]:  # top 2 keywords avoids over-querying
        for r in get_similar_tasks(keyword=keyword, limit=3):
            if r not in raw_similar:
                raw_similar.append(r)

    capped_similar: list[dict] = []
    for t in raw_similar[:5]:
        # Build a trimmed representation — only fields the Planner uses
        result_preview = str(t.get("result", ""))[:120]
        line = f"{t.get('goal', '')} → {result_preview}"
        line = _truncate(line, _MAX_CHARS_PER_TASK_RECORD)
        cost = _estimate_tokens(line)
        if cost > budget:
            break
        capped_similar.append(
            {
                "goal": t.get("goal", ""),
                # result is pre-truncated — no full extraction blobs
                "result": result_preview,
                "steps_taken": t.get("steps_taken", "?"),
                "status": t.get("status", ""),
            }
        )
        budget -= cost

    # ── 2. Recent successful actions for this goal type ───────────────────────
    # output_json is the field that bloated to 26k — hard-cap it at 120 chars.
    raw_successes = get_recent_successes(goal_type=goal_type, limit=5)

    capped_successes: list[dict] = []
    for s in raw_successes:
        if budget < 50:
            break
        trimmed = {
            "tool": s.get("tool", ""),
            "context": s.get("context", "")[:80],
            # This is the field that was bloating to 26k — hard cap it.
            "output_json": _truncate(s.get("output_json", ""), 120),
        }
        cost = _estimate_tokens(str(trimmed))
        if cost > budget:
            break
        capped_successes.append(trimmed)
        budget -= cost

    return PlanningContext(
        similar_tasks=capped_similar,
        recent_successes=capped_successes,
    )


def get_correction_context(tool: str, task_id: str) -> CorrectionContext:
    """
    Retrieve failure history for a specific tool within a task.

    Called by the Self-Correction Engine immediately after a step
    fails, before selecting a recovery strategy.

    This does NOT enforce a token budget — correction context is read
    as structured data by Python code, never injected raw into an LLM prompt.

    Parameters
    ----------
    tool    : The tool that just failed.
    task_id : Current task id — scopes history to this session.

    Returns
    -------
    CorrectionContext with failure_history populated.
    already_tried is computed from that history.
    """
    history = get_failure_history(tool=tool, task_id=task_id)
    return CorrectionContext(failure_history=history)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: 1 token ≈ 4 characters (conservative).
    No external dependency. Accurate enough to enforce a budget ceiling.
    """
    return max(1, len(text) // 4)


def _truncate(value: Any, max_chars: int) -> str:
    """Serialize value to string and hard-truncate to max_chars."""
    s = json.dumps(value) if not isinstance(value, str) else value
    return s[:max_chars] + "…" if len(s) > max_chars else s


def _extract_keywords(goal: str) -> list[str]:
    """
    Extract meaningful keywords from a goal string for SQL LIKE queries.

    Strips common filler words and returns up to 3 content words.

        _extract_keywords("Go to wikipedia.org and find the release year of Python")
        # → ["wikipedia", "release", "Python"]

    Intentionally simple — SQL LIKE '%keyword%' is fuzzy enough.
    """
    stopwords = {
        "go", "to", "and", "the", "a", "an", "of", "in", "on",
        "at", "is", "it", "find", "get", "return", "click", "open",
        "navigate", "search", "for", "from", "with", "that", "this",
        "then", "first", "last", "all", "any", "into", "onto",
    }
    words = re.findall(r"[a-zA-Z]{3,}", goal)
    keywords = [w for w in words if w.lower() not in stopwords]
    return keywords[:3]