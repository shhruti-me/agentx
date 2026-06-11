"""
memory/task_memory.py

Read and write for the tasks table.

No SQL outside this file touches the tasks table.
All other modules call these named functions.

Functions
---------
    write_task(task)              → None
    update_task(task)             → None
    get_task(task_id)             → Task | None
    get_similar_tasks(keyword, limit) → list[dict]
    list_tasks(status, limit)     → list[dict]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from core.models import ExecutionDAG, GoalType, Task, TaskStatus
from memory.db import get_connection

logger = logging.getLogger(__name__)


def write_task(task: Task) -> None:
    """
    Insert a new task record. Called immediately after Task is created.
    Raises on duplicate id (logic error — never create two Tasks with same id).
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                id, goal, goal_type, status, plan_json, result,
                tokens_used, steps_taken, corrections,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.id,
                task.goal,
                task.goal_type,
                task.status,
                task.plan.to_json() if task.plan else None,
                task.result,
                task.tokens_used,
                task.steps_taken,
                task.corrections,
                task.started_at.isoformat(),
                task.completed_at.isoformat() if task.completed_at else None,
            ),
        )
    logger.debug("task_written", task_id=task.id, status=task.status)


def update_task(task: Task) -> None:
    """
    Update all mutable fields of an existing task record.
    Called whenever task status, result, or counters change.
    """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tasks SET
                status       = ?,
                plan_json    = ?,
                result       = ?,
                tokens_used  = ?,
                steps_taken  = ?,
                corrections  = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                task.status,
                task.plan.to_json() if task.plan else None,
                task.result,
                task.tokens_used,
                task.steps_taken,
                task.corrections,
                task.completed_at.isoformat() if task.completed_at else None,
                task.id,
            ),
        )
    logger.debug("task_updated", task_id=task.id, status=task.status)


def get_task(task_id: str) -> Task | None:
    """
    Fetch a Task by id. Returns None if not found.
    Reconstructs the Task dataclass including the ExecutionDAG.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    if row is None:
        return None

    return _row_to_task(row)


def get_similar_tasks(keyword: str, limit: int = 3) -> list[dict]:
    """
    Find completed tasks whose goal contains the keyword.

    Used by the Planner to inject past successful context into
    the planning prompt. Returns plain dicts — the Planner only
    needs goal/result strings, not full Task objects.

    Parameters
    ----------
    keyword : A word or phrase extracted from the current goal.
    limit   : Maximum number of results.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT goal, status, result, goal_type, steps_taken, tokens_used
            FROM tasks
            WHERE goal LIKE ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (f"%{keyword}%", limit),
        ).fetchall()

    return [dict(row) for row in rows]


def list_tasks(
    status: str | None = None,
    goal_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    List tasks with optional filters. Used by GET /v1/tasks.
    Returns plain dicts suitable for JSON serialisation.
    """
    query = "SELECT * FROM tasks"
    params: list = []
    conditions: list[str] = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if goal_type:
        conditions.append("goal_type = ?")
        params.append(goal_type)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _row_to_task(row: object) -> Task:
    """Reconstruct a Task dataclass from a sqlite3.Row."""
    task = Task(
        goal=row["goal"],
        goal_type=GoalType(row["goal_type"]) if row["goal_type"] else GoalType.UNKNOWN,
    )
    task.id          = row["id"]
    task.status      = TaskStatus(row["status"])
    task.result      = row["result"]
    task.tokens_used = row["tokens_used"] or 0
    task.steps_taken = row["steps_taken"] or 0
    task.corrections = row["corrections"] or 0
    task.started_at  = datetime.fromisoformat(row["started_at"])

    if row["completed_at"]:
        task.completed_at = datetime.fromisoformat(row["completed_at"])

    if row["plan_json"]:
        task.plan = ExecutionDAG.from_json(row["plan_json"])

    return task