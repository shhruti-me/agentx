"""
evaluation/benchmark_runner.py

Runs the full benchmark suite against the live agent.

Reads task definitions from evaluation/benchmarks/dataset.json.
Runs each task through the Orchestrator.
Writes results to the benchmark_results table.
Prints a live progress line per task.

Usage
-----
    # From CLI:
    python -m evaluation.benchmark_runner

    # Subset:
    python -m evaluation.benchmark_runner --suite easy
    python -m evaluation.benchmark_runner --category navigation
    python -m evaluation.benchmark_runner --ids nav_001,nav_002

    # From API (Week 6+):
    POST /v1/benchmark {"suite": "full"}

CHANGE (2026-06-27):
    BenchmarkResult now carries task_id (the UUID from tasks.id).
    _write_result persists task_id to the benchmark_results table.
    This lets MetricsCalculator scope correction/failure queries to
    the current run without a fragile time-window join.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings
from core.orchestrator import Orchestrator
from memory.db import get_connection
from log.logger import get_logger

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    run_id:       str
    task_name:    str
    category:     str
    difficulty:   str
    goal:         str
    status:       str           # pass | fail | partial
    steps_taken:  int
    corrections:  int
    tokens_used:  int
    time_seconds: float
    output:       str
    notes:        str = ""
    task_id:      str = ""      # UUID from tasks.id — used to scope metrics queries


# ── Runner ────────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """
    Runs benchmark tasks and records results.

    Parameters
    ----------
    suite      : "full" | "easy" | "medium" | "hard"
    category   : Filter to one category (optional)
    task_ids   : Run only these specific task IDs (optional)
    timeout    : Per-task wall-clock timeout in seconds
    """

    def __init__(
        self,
        suite: str = "full",
        category: str | None = None,
        task_ids: list[str] | None = None,
        timeout: int = 120,
    ) -> None:
        self._suite    = suite
        self._category = category
        self._task_ids = task_ids
        self._timeout  = timeout
        self._run_id   = uuid.uuid4().hex[:12]

    async def run(self) -> list[BenchmarkResult]:
        """Execute all selected benchmark tasks and return results."""
        tasks = self._load_tasks()
        if not tasks:
            logger.warning("benchmark_no_tasks", suite=self._suite, category=self._category)
            return []

        logger.info(
            "benchmark_start",
            run_id=self._run_id,
            task_count=len(tasks),
            suite=self._suite,
        )

        print(f"\n  AGENTX Benchmark — run {self._run_id}")
        print(f"  Tasks: {len(tasks)}  |  Suite: {self._suite}")
        print("  " + "─" * 56)

        results: list[BenchmarkResult] = []

        for i, task_def in enumerate(tasks, 1):
            print(
                f"\n  [{i:02d}/{len(tasks):02d}] {task_def['id']:15s} {task_def['category']:20s} ",
                end="",
                flush=True,
            )

            result = await self._run_one(task_def)
            results.append(result)
            self._write_result(result)

            icon = "✓" if result.status == "pass" else ("~" if result.status == "partial" else "✗")
            print(
                f"{icon}  {result.time_seconds:.1f}s  {result.steps_taken}steps  {result.corrections}corr",
                flush=True,
            )

        print("\n  " + "─" * 56)
        print(f"  Run complete: {sum(1 for r in results if r.status == 'pass')}/{len(results)} passed")
        print()

        return results

    async def _run_one(self, task_def: dict) -> BenchmarkResult:
        """
        Run a single benchmark task with timeout.
        Returns a BenchmarkResult regardless of outcome.
        Captures task.id so MetricsCalculator can scope DB queries.
        """
        start = time.monotonic()
        orchestrator = Orchestrator()

        try:
            task = await asyncio.wait_for(
                orchestrator.run(
                    goal=task_def["goal"],
                    goal_type=task_def.get("category", "unknown"),
                ),
                timeout=self._timeout,
            )

            elapsed = time.monotonic() - start
            output  = task.result or ""
            status  = self._evaluate(task_def, output, task.status)

            return BenchmarkResult(
                run_id=self._run_id,
                task_name=task_def["id"],
                category=task_def.get("category", ""),
                difficulty=task_def.get("difficulty", ""),
                goal=task_def["goal"],
                status=status,
                steps_taken=task.steps_taken,
                corrections=task.corrections,
                tokens_used=task.tokens_used,
                time_seconds=round(elapsed, 1),
                output=output[:500],
                notes=task_def.get("note", ""),
                task_id=task.id,            # ← captured here
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return BenchmarkResult(
                run_id=self._run_id,
                task_name=task_def["id"],
                category=task_def.get("category", ""),
                difficulty=task_def.get("difficulty", ""),
                goal=task_def["goal"],
                status="fail",
                steps_taken=0,
                corrections=0,
                tokens_used=0,
                time_seconds=round(elapsed, 1),
                output="",
                notes=f"Wall-clock timeout after {self._timeout}s",
                task_id="",
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("benchmark_task_error", task_id=task_def["id"], error=str(exc)[:200])
            return BenchmarkResult(
                run_id=self._run_id,
                task_name=task_def["id"],
                category=task_def.get("category", ""),
                difficulty=task_def.get("difficulty", ""),
                goal=task_def["goal"],
                status="fail",
                steps_taken=0,
                corrections=0,
                tokens_used=0,
                time_seconds=round(time.monotonic() - start, 1),
                output="",
                notes=f"Exception: {str(exc)[:200]}",
                task_id="",
            )

    def _evaluate(self, task_def: dict, output: str, task_status: object) -> str:
        """
        Determine pass / fail / partial for a completed task.

        pass    — task completed AND all expected output checks pass
        partial — task completed but expected strings not in output
        fail    — task failed (status=failed or timed out)
        """
        from core.models import TaskStatus
        if task_status == TaskStatus.FAILED or str(task_status) == "failed":
            return "fail"

        output_lower = output.lower()

        # Check min output length
        min_len = task_def.get("min_output_length", 0)
        if min_len and len(output.strip()) < min_len:
            return "partial"

        # Check expected strings
        expected = task_def.get("expected_output_contains", [])
        if expected:
            all_found = all(s.lower() in output_lower for s in expected)
            if not all_found:
                return "partial"

        # Check regex pattern if present
        pattern = task_def.get("expected_output_pattern")
        if pattern and not re.search(pattern, output, re.IGNORECASE):
            return "partial"

        return "pass"

    def _load_tasks(self) -> list[dict]:
        """Load and filter benchmark tasks from dataset.json."""
        path = settings.benchmark_dataset_path
        if not path.exists():
            logger.error("benchmark_dataset_missing", path=str(path))
            return []

        raw   = path.read_text(encoding="utf-8")
        clean = re.sub(r"//[^\n]*", "", raw)   # strip JS-style comments
        all_tasks: list[dict] = json.loads(clean)

        # Filter by specific IDs first
        if self._task_ids:
            return [t for t in all_tasks if t["id"] in self._task_ids]

        # Filter by difficulty suite
        if self._suite == "easy":
            all_tasks = [t for t in all_tasks if t.get("difficulty") == "easy"]
        elif self._suite == "medium":
            all_tasks = [t for t in all_tasks if t.get("difficulty") in ("easy", "medium")]
        elif self._suite == "hard":
            all_tasks = [t for t in all_tasks if t.get("difficulty") == "hard"]

        # Filter by category
        if self._category:
            all_tasks = [t for t in all_tasks if t.get("category") == self._category]

        return all_tasks

    def _write_result(self, result: BenchmarkResult) -> None:
        """
        Persist one benchmark result to SQLite.
        Writes task_id so MetricsCalculator can join to action_failures.
        """
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO benchmark_results (
                        id, run_id, task_name, category, difficulty,
                        goal, status, steps_taken, corrections,
                        tokens_used, time_seconds, output, notes, ran_at, task_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        uuid.uuid4().hex[:12],
                        result.run_id,
                        result.task_name,
                        result.category,
                        result.difficulty,
                        result.goal,
                        result.status,
                        result.steps_taken,
                        result.corrections,
                        result.tokens_used,
                        result.time_seconds,
                        result.output,
                        result.notes,
                        now,
                        result.task_id,         # ← written here
                    ),
                )
        except Exception as exc:
            logger.warning("benchmark_write_failed", error=str(exc)[:200])


# ── CLI entry point ───────────────────────────────────────────────────────────


async def main() -> None:
    import argparse
    from log.logger import setup_logging
    from memory.db import init_db
    from evaluation.metrics import MetricsCalculator
    from evaluation.reporter import Reporter

    parser = argparse.ArgumentParser(description="AGENTX Benchmark Runner")
    parser.add_argument("--suite", default="full", choices=["full", "easy", "medium", "hard"])
    parser.add_argument("--category", default=None, help="Filter to one category")
    parser.add_argument("--ids", default=None, help="Comma-separated task IDs to run")
    parser.add_argument("--timeout", type=int, default=120, help="Per-task timeout in seconds")
    args = parser.parse_args()

    setup_logging()
    init_db()

    task_ids = [x.strip() for x in args.ids.split(",")] if args.ids else None

    runner = BenchmarkRunner(
        suite=args.suite,
        category=args.category,
        task_ids=task_ids,
        timeout=args.timeout,
    )

    results = await runner.run()
    if not results:
        print("No results — check dataset.json path and filters.")
        return

    run_id  = runner._run_id
    metrics = MetricsCalculator.compute(results, run_id=run_id)

    reporter = Reporter()
    reporter.print_report(metrics, results)
    reporter.save_report(metrics, results, run_id)


if __name__ == "__main__":
    asyncio.run(main())