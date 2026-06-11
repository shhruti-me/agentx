"""
core/models.py

All dataclasses and enums for AGENTX.

This is the type system for the entire project. Every module imports
from here. No module defines its own task, step, or result types.

Rule: no logic lives here. No methods that call other modules.
Only data, validation, and serialisation helpers.

Import pattern used everywhere:
    from core.models import Task, Step, ExecutionDAG, TaskStatus
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Enums ─────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """
    Lifecycle states for a Task.

    str mixin means TaskStatus.PENDING == "pending" is True,
    which makes SQLite storage and JSON serialisation trivial —
    no conversion needed.
    """
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class GoalType(str, Enum):
    """
    Broad category of what the agent is being asked to do.
    Used by the Planner to select prompt templates and by Memory
    to scope retrieval queries.
    """
    NAVIGATION       = "navigation"
    EXTRACTION       = "extraction"
    FORM_INTERACTION = "form_interaction"
    MULTI_STEP       = "multi_step"
    UNKNOWN          = "unknown"


class VerificationStatus(str, Enum):
    PASS      = "pass"
    FAIL      = "fail"
    UNCERTAIN = "uncertain"


class FailureType(str, Enum):
    """
    Taxonomy of step failures used by the Self-Correction Engine
    to select the right recovery strategy.

    Each value maps to one or more correction strategies in
    correction/classifier.py.
    """
    STALE_SELECTOR    = "stale_selector"    # element not found, DOM changed
    TIMEOUT           = "timeout"           # page/action did not complete in time
    AUTH_WALL         = "auth_wall"         # unexpected login or CAPTCHA page
    WRONG_PAGE        = "wrong_page"        # navigated somewhere unexpected
    EXTRACTION_EMPTY  = "extraction_empty"  # extractor returned no content
    PLAN_ERROR        = "plan_error"        # step makes no sense given page state
    UNKNOWN           = "unknown"           # fallback — triggers RETRY


class CorrectionStrategy(str, Enum):
    """
    Recovery strategies available to the Self-Correction Engine.
    Applied in order of increasing cost.
    """
    RETRY         = "retry"         # re-run the exact step
    SELECTOR_FIX  = "selector_fix"  # ask LLM for alternative selectors
    REPLAN        = "replan"        # regenerate DAG from current position
    ABORT         = "abort"         # mark task failed, return clear reason


# ── Step ──────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """
    A single executable unit within an ExecutionDAG.

    Created by the Planner. Executed by the Execution Engine.
    Verified by the Verification Engine.

    Fields
    ------
    tool          : Name of the tool to invoke. Must exist in Tool Registry.
    input         : Parameters passed to the tool. Schema is tool-specific.
    expected      : Natural language description of what success looks like.
                    Used by the Verification Engine as the success condition.
    task_id       : Set by the Execution Engine when execution begins.
    output        : Populated after the tool runs successfully.
    error_message : Populated on failure.
    """
    tool:          str
    input:         dict[str, Any]
    expected:      str

    # Set at execution time
    id:            str        = field(default_factory=_new_id)
    task_id:       str        = field(default="")
    step_index:    int        = field(default=0)
    status:        StepStatus = field(default=StepStatus.PENDING)
    output:        dict[str, Any] | None = field(default=None)
    retry_count:   int        = field(default=0)
    error_message: str | None = field(default=None)
    started_at:    datetime | None = field(default=None)
    completed_at:  datetime | None = field(default=None)

    def mark_running(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = _now_utc()

    def mark_completed(self, output: dict[str, Any]) -> None:
        self.status = StepStatus.COMPLETED
        self.output = output
        self.completed_at = _now_utc()

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error_message = error
        self.completed_at = _now_utc()

    def mark_skipped(self) -> None:
        self.status = StepStatus.SKIPPED
        self.completed_at = _now_utc()

    @property
    def duration_ms(self) -> int | None:
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":            self.id,
            "task_id":       self.task_id,
            "step_index":    self.step_index,
            "tool":          self.tool,
            "input":         self.input,
            "expected":      self.expected,
            "output":        self.output,
            "status":        self.status,
            "retry_count":   self.retry_count,
            "error_message": self.error_message,
            "started_at":    self.started_at.isoformat() if self.started_at else None,
            "completed_at":  self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms":   self.duration_ms,
        }


# ── ExecutionDAG ──────────────────────────────────────────────────────────────


@dataclass
class ExecutionDAG:
    """
    An ordered list of Steps produced by the Planner.

    V1: sequential execution only — steps run in index order.
    V2: parallel branches via asyncio.gather (the dataclass
        structure already supports it via depends_on).

    The Execution Engine calls get_next_runnable() in a loop
    until is_complete() returns True or a step fails.
    """
    steps: list[Step] = field(default_factory=list)

    def get_next_runnable(self) -> Step | None:
        """Return the first PENDING step, or None if all steps are done."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_complete(self, step_id: str) -> None:
        step = self._get(step_id)
        if step:
            step.status = StepStatus.COMPLETED

    def mark_failed(self, step_id: str) -> None:
        step = self._get(step_id)
        if step:
            step.status = StepStatus.FAILED

    def is_complete(self) -> bool:
        """True when every step is COMPLETED or SKIPPED."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def completed_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def failed_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def _get(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self.steps], default=str)

    @classmethod
    def from_json(cls, raw: str) -> ExecutionDAG:
        """Reconstruct an ExecutionDAG from a stored JSON string."""
        items = json.loads(raw)
        steps = []
        for item in items:
            step = Step(
                tool=item["tool"],
                input=item.get("input", {}),
                expected=item.get("expected", ""),
            )
            step.id          = item.get("id", _new_id())
            step.task_id     = item.get("task_id", "")
            step.step_index  = item.get("step_index", 0)
            step.status      = StepStatus(item.get("status", "pending"))
            step.output      = item.get("output")
            step.retry_count = item.get("retry_count", 0)
            step.error_message = item.get("error_message")
            if item.get("started_at"):
                step.started_at = datetime.fromisoformat(item["started_at"])
            if item.get("completed_at"):
                step.completed_at = datetime.fromisoformat(item["completed_at"])
            steps.append(step)
        return cls(steps=steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        counts = {s.status: 0 for s in self.steps}
        for s in self.steps:
            counts[s.status] += 1
        return f"ExecutionDAG(steps={len(self.steps)}, status={dict(counts)})"


# ── Task ──────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    """
    Top-level unit of work. One goal = one Task.

    Created at intake. Written to SQLite immediately.
    Updated throughout execution. Final state is COMPLETED or FAILED.
    """
    goal:      str
    goal_type: GoalType = GoalType.UNKNOWN

    id:           str        = field(default_factory=_new_id)
    status:       TaskStatus = field(default=TaskStatus.PENDING)
    plan:         ExecutionDAG | None = field(default=None)
    result:       str | None = field(default=None)
    tokens_used:  int        = field(default=0)
    steps_taken:  int        = field(default=0)
    corrections:  int        = field(default=0)
    started_at:   datetime   = field(default_factory=_now_utc)
    completed_at: datetime | None = field(default=None)

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = _now_utc()

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.result = reason
        self.completed_at = _now_utc()

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":           self.id,
            "goal":         self.goal,
            "goal_type":    self.goal_type,
            "status":       self.status,
            "result":       self.result,
            "tokens_used":  self.tokens_used,
            "steps_taken":  self.steps_taken,
            "corrections":  self.corrections,
            "started_at":   self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


# ── ActionResult ──────────────────────────────────────────────────────────────


@dataclass
class ActionResult:
    """
    Returned by every Browser Controller action and tool.

    The Execution Engine receives this and passes it to the
    Verification Engine. The Verifier checks actual vs expected.
    """
    success:     bool
    output:      dict[str, Any] = field(default_factory=dict)
    error:       str | None = field(default=None)
    duration_ms: int = field(default=0)

    @classmethod
    def ok(cls, output: dict[str, Any], duration_ms: int = 0) -> ActionResult:
        return cls(success=True, output=output, duration_ms=duration_ms)

    @classmethod
    def fail(cls, error: str, duration_ms: int = 0) -> ActionResult:
        return cls(success=False, error=error, duration_ms=duration_ms)


# ── VerificationResult ────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """
    Returned by the Verification Engine after checking a step outcome.

    method tells us which verifier resolved the check:
      "dom"  — structural page change confirmed
      "text" — expected string found in page content
      "llm"  — LLM judge confirmed semantic success
      "none" — all methods returned UNCERTAIN → treated as FAIL

    confidence is 0.0–1.0. DOM and text verifiers return 1.0 on match.
    LLM verifier returns a model-estimated confidence.
    """
    result:     VerificationStatus
    method:     str
    confidence: float
    reason:     str

    @property
    def passed(self) -> bool:
        return self.result == VerificationStatus.PASS

    @property
    def failed(self) -> bool:
        return self.result == VerificationStatus.FAIL


# ── CorrectionResult ──────────────────────────────────────────────────────────


@dataclass
class CorrectionResult:
    """
    Returned by the Self-Correction Engine after attempting recovery.

    If strategy_used is REPLAN, new_dag contains the revised plan
    from the current position forward. The Execution Engine switches
    to this DAG and continues.
    """
    strategy_used: CorrectionStrategy
    success:       bool
    new_dag:       ExecutionDAG | None = field(default=None)
    reason:        str = field(default="")