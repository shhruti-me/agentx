"""
memory/db.py

SQLite connection management and schema initialisation for AGENTX.

One SQLite file. Five tables. No ORM. No migration framework.
All tables are created with CREATE TABLE IF NOT EXISTS on startup.

Every memory module calls get_connection() to get a connection.
No module outside memory/ ever touches SQLite directly.

Usage
-----
    from memory.db import get_connection, init_db

    # At startup (called by api/app.py lifespan and main.py):
    init_db()

    # In memory modules:
    with get_connection() as conn:
        conn.execute("SELECT ...", params)
"""

from __future__ import annotations
from log.logger import get_logger

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config.settings import settings

logger = get_logger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Every task run from goal to result
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    goal_type       TEXT,
    status          TEXT NOT NULL,
    plan_json       TEXT,
    result          TEXT,
    tokens_used     INTEGER DEFAULT 0,
    steps_taken     INTEGER DEFAULT 0,
    corrections     INTEGER DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT
);

-- Every step executed within a task
CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    step_index      INTEGER NOT NULL,
    tool            TEXT NOT NULL,
    input_json      TEXT,
    output_json     TEXT,
    expected        TEXT,
    status          TEXT NOT NULL,
    retry_count     INTEGER DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT,
    error_message   TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- Successful action patterns the agent has learned
CREATE TABLE IF NOT EXISTS action_successes (
    id              TEXT PRIMARY KEY,
    tool            TEXT NOT NULL,
    goal_type       TEXT,
    context         TEXT,
    input_json      TEXT,
    output_json     TEXT,
    site_domain     TEXT,
    created_at      TEXT
);

-- Failure events and what fixed them
CREATE TABLE IF NOT EXISTS action_failures (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    tool            TEXT NOT NULL,
    failure_type    TEXT,
    input_json      TEXT,
    error_message   TEXT,
    correction_used TEXT,
    was_recovered   INTEGER DEFAULT 0,
    created_at      TEXT
);

-- Benchmark run results
CREATE TABLE IF NOT EXISTS benchmark_results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    task_name       TEXT NOT NULL,
    category        TEXT,
    difficulty      TEXT,
    goal            TEXT,
    status          TEXT,
    steps_taken     INTEGER,
    corrections     INTEGER,
    tokens_used     INTEGER,
    time_seconds    REAL,
    output          TEXT,
    notes           TEXT,
    ran_at          TEXT
);
"""


# ── Connection ────────────────────────────────────────────────────────────────


def _get_db_path() -> Path:
    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a SQLite connection with sane defaults.

    Use as a context manager — connection is committed and closed
    automatically. On exception, the transaction is rolled back.

        with get_connection() as conn:
            conn.execute("INSERT INTO tasks ...", params)

    Settings applied:
      - WAL mode: allows concurrent reads during a write
      - Row factory: rows accessible as dicts (row["column_name"])
      - Foreign keys: enforced
      - Timeout: 10s before raising OperationalError on lock
    """
    path = _get_db_path()
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ────────────────────────────────────────────────────────────


def init_db() -> None:
    """
    Create all tables if they don't exist.

    Safe to call multiple times — CREATE TABLE IF NOT EXISTS
    is idempotent. Called at startup from main.py and the
    FastAPI lifespan handler.
    """
    logger.info("db_init", path=str(_get_db_path()))
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
    logger.info("db_ready", tables=5)


def check_db() -> bool:
    """
    Verify the database is reachable and tables exist.
    Returns True on success. Used by the /health endpoint.
    Never raises.
    """
    try:
        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {row["name"] for row in tables}
            required = {"tasks", "steps", "action_successes", "action_failures", "benchmark_results"}
            return required.issubset(names)
    except Exception:
        return False