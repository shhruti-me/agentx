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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory.action_memory import get_recent_successes
from memory.failure_memory import get_failure_history
from memory.task_memory import get_similar_tasks


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
    recent_successes  : Recent successful tool invocations for this
                        goal type. Tells the Planner which selectors
                        and approaches have worked on similar pages.
    has_prior_context : True if any memory was found. Lets the Planner
                        decide whether to include a memory section in
                        its prompt or skip it entirely.
    """
    similar_tasks:    list[dict] = field(default_factory=list)
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
    failure_history       : All correction attempts for this tool in
                            this task. Used to skip already-tried strategies.
    already_tried         : Set of strategy names already attempted.
                            Convenience accessor over failure_history.
    previous_recovery_worked : True if a prior correction for this tool
                            in this task eventually succeeded. Hints that
                            the site is navigable — keep trying.
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
    """
    keywords = _extract_keywords(goal)

    similar: list[dict] = []
    for keyword in keywords[:2]:  # top 2 keywords avoids over-querying
        results = get_similar_tasks(keyword=keyword, limit=3)
        for r in results:
            if r not in similar:
                similar.append(r)
    similar = similar[:5]  # cap at 5 total

    successes = get_recent_successes(goal_type=goal_type, limit=5)

    return PlanningContext(
        similar_tasks=similar,
        recent_successes=successes,
    )


def get_correction_context(tool: str, task_id: str) -> CorrectionContext:
    """
    Retrieve failure history for a specific tool within a task.

    Called by the Self-Correction Engine immediately after a step
    fails, before selecting a recovery strategy.

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_keywords(goal: str) -> list[str]:
    """
    Extract meaningful keywords from a goal string for SQL LIKE queries.

    Strips common filler words and returns up to 3 content words.

        _extract_keywords("Go to wikipedia.org and find the release year of Python")
        # → ["wikipedia", "release", "Python"]

    This is intentionally simple — we don't need NLP here.
    The SQL query uses LIKE '%keyword%' which is fuzzy enough.
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