"""
memory/failure_memory.py

Read and write for the action_failures table.

Records every failure event: which tool failed, what type of failure,
what correction was applied, and whether it worked.

The Self-Correction Engine queries this before selecting a strategy
so it skips approaches that have already failed for this tool in
this session.

Functions
---------
    write_failure(tool, failure_type, input_data, error,
                  correction_used, was_recovered, task_id) → None
    get_failure_history(tool, task_id)                     → list[dict]
    get_failure_stats(tool, limit)                         → list[dict]
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import get_connection

logger = logging.getLogger(__name__)


def write_failure(
    tool: str,
    failure_type: str,
    input_data: dict[str, Any],
    error_message: str,
    correction_used: str,
    was_recovered: bool,
    task_id: str,
) -> None:
    """
    Record a failure event and the correction that was attempted.

    Called by the Self-Correction Engine after each recovery attempt,
    win or lose.

    Parameters
    ----------
    tool            : The tool that failed (e.g. "click", "extract").
    failure_type    : Classified failure type (e.g. "stale_selector").
    input_data      : Parameters that were passed to the tool when it failed.
    error_message   : The raw error string from the action or verifier.
    correction_used : Which strategy was applied (retry/selector_fix/replan/abort).
    was_recovered   : True if the correction succeeded, False if it didn't.
    task_id         : Parent task — used to scope history queries to this session.
    """
    record_id = uuid.uuid4().hex[:12]
    now = datetime.now(tz=timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO action_failures (
                id, task_id, tool, failure_type,
                input_json, error_message,
                correction_used, was_recovered, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                task_id,
                tool,
                failure_type,
                json.dumps(input_data, default=str),
                error_message,
                correction_used,
                1 if was_recovered else 0,
                now,
            ),
        )

    logger.debug(
        "failure_written",
        tool=tool,
        failure_type=failure_type,
        correction_used=correction_used,
        was_recovered=was_recovered,
        task_id=task_id,
    )


def get_failure_history(tool: str, task_id: str) -> list[dict]:
    """
    Return all failure/correction events for this tool within this task.

    Called by the Self-Correction Engine before selecting a strategy.
    If a correction strategy already appears in this history and
    was_recovered=0, the engine skips it and tries the next one.

    Parameters
    ----------
    tool    : The tool that just failed.
    task_id : Current task — scopes the query to this session only.
              We don't want failures from unrelated past tasks to
              block a strategy that might work now.

    Returns
    -------
    List of dicts with keys: correction_used, was_recovered, failure_type,
    error_message, created_at. Ordered oldest-first so the engine can
    see the sequence of attempts.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT correction_used, was_recovered, failure_type,
                   error_message, created_at
            FROM action_failures
            WHERE tool = ? AND task_id = ?
            ORDER BY created_at ASC
            """,
            (tool, task_id),
        ).fetchall()

    return [dict(row) for row in rows]


def get_failure_stats(tool: str | None = None, limit: int = 50) -> list[dict]:
    """
    Aggregate failure and recovery rates across all tasks.

    Used by the benchmark reporter and optionally surfaced in the
    API to show which tools are least reliable.

    Returns
    -------
    List of dicts with: tool, failure_type, total, recovered, recovery_rate.
    Ordered by total failures descending.
    """
    query = """
        SELECT
            tool,
            failure_type,
            COUNT(*)                                    AS total,
            SUM(was_recovered)                          AS recovered,
            ROUND(AVG(was_recovered) * 100, 1)          AS recovery_rate
        FROM action_failures
    """
    params: list = []
    if tool:
        query += " WHERE tool = ?"
        params.append(tool)

    query += " GROUP BY tool, failure_type ORDER BY total DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]