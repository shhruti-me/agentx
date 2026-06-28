"""
db/add_task_id_column.py

One-time migration: adds task_id column to benchmark_results table.

Run once before the next benchmark:
    python db/add_task_id_column.py

Safe to run multiple times — uses ALTER TABLE IF NOT EXISTS pattern via
exception handling (SQLite doesn't support IF NOT EXISTS on ALTER TABLE).
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "agentx.db"


def migrate() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "ALTER TABLE benchmark_results ADD COLUMN task_id TEXT DEFAULT ''"
        )
        conn.commit()
        print(f"✓ Added task_id column to benchmark_results in {DB_PATH}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("✓ task_id column already exists — nothing to do")
        else:
            raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()