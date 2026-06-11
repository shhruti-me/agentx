"""
memory/action_memory.py

Read and write for the action_successes table.

Records every successful tool invocation with its context and output.
The Planner queries this before generating a plan to bias toward
known-working approaches for a given goal type and site.

Functions
---------
    write_success(tool, context, input, output, goal_type, site_domain) → None
    get_recent_successes(goal_type, tool, limit) → list[dict]
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from memory.db import get_connection

logger = logging.getLogger(__name__)


def write_success(
    tool: str,
    context: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    goal_type: str,
    site_domain: str = "",
) -> None:
    """
    Record a successful tool invocation.

    Called by the Execution Engine after a step completes and
    the Verifier returns PASS.

    Parameters
    ----------
    tool        : Tool name (e.g. "navigate", "click", "extract").
    context     : Short description of the page/situation when the
                  action ran. Used by the Planner to understand when
                  this approach works.
    input_data  : The parameters passed to the tool.
    output_data : What the tool returned.
    goal_type   : Category of the parent task (navigation, extraction, etc.)
    site_domain : Domain extracted from the URL being acted on.
                  Helps the Planner prefer site-specific patterns.
    """
    record_id = uuid.uuid4().hex[:12]
    now = datetime.now(tz=timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO action_successes (
                id, tool, goal_type, context,
                input_json, output_json, site_domain, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                tool,
                goal_type,
                context,
                json.dumps(input_data, default=str),
                json.dumps(output_data, default=str),
                site_domain,
                now,
            ),
        )

    logger.debug(
        "action_success_written",
        tool=tool,
        goal_type=goal_type,
        site_domain=site_domain,
    )


def get_recent_successes(
    goal_type: str,
    tool: str | None = None,
    site_domain: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve recent successful actions for the given goal type.

    Called by the Planner before generating a plan. Results are
    injected into the planning prompt as few-shot examples of
    what has worked before.

    Parameters
    ----------
    goal_type   : Filter to this goal category.
    tool        : Optional — filter to a specific tool.
    site_domain : Optional — prefer results from the same domain.
    limit       : Maximum results to return.

    Returns
    -------
    List of dicts with keys: tool, context, input_json, output_json,
    site_domain, created_at. JSON fields are returned as strings —
    the Planner embeds them directly in the prompt.
    """
    conditions = ["goal_type = ?"]
    params: list = [goal_type]

    if tool:
        conditions.append("tool = ?")
        params.append(tool)
    if site_domain:
        conditions.append("site_domain = ?")
        params.append(site_domain)

    query = f"""
        SELECT tool, context, input_json, output_json, site_domain, created_at
        FROM action_successes
        WHERE {" AND ".join(conditions)}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def domain_from_url(url: str) -> str:
    """
    Extract the domain from a URL for site_domain storage.

    Helper used by the Execution Engine when writing successes.

        domain_from_url("https://news.ycombinator.com/item?id=1")
        # → "news.ycombinator.com"
    """
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""