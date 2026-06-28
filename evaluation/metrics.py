"""
evaluation/metrics.py

Computes all benchmark metrics from a list of BenchmarkResults.

All metrics defined in Section 10 of the architecture document:
  - task_success_rate
  - partial_success_rate
  - avg_steps_per_task
  - avg_tokens_per_task
  - avg_time_per_task
  - recovery_rate
  - avg_corrections_per_task
  - failure_breakdown by type
  - correction_effectiveness by strategy

BUG FIXED (2026-06-27):
  _compute_recovery_rate, failure_breakdown, and correction_effectiveness
  previously queried action_failures with no WHERE clause, pulling every
  failure from every benchmark run ever recorded. This caused stale
  correction stats to appear in runs that had 0 corrections.

  Fix: BenchmarkResult now carries task_id (the UUID from tasks.id).
  benchmark_runner._write_result persists it to benchmark_results.task_id.
  All three DB methods here scope their queries with:

      WHERE task_id IN (
          SELECT task_id FROM benchmark_results WHERE run_id = ?
      )

  This is an exact join — no time-window guessing, no fragile heuristics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


# ── Return types ──────────────────────────────────────────────────────────────


@dataclass
class CategoryMetrics:
    category:   str
    total:      int
    passed:     int
    partial:    int
    failed:     int
    avg_steps:  float
    avg_tokens: float
    avg_time:   float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class BenchmarkSummary:
    run_id:               str
    total_tasks:          int
    passed:               int
    partial:              int
    failed:               int

    task_success_rate:        float   # passed / total
    partial_success_rate:     float   # partial / total
    recovery_rate:            float   # corrections that succeeded / total corrections

    avg_steps_per_task:       float
    avg_corrections_per_task: float
    avg_tokens_per_task:      float
    avg_time_per_task:        float

    total_tokens: int
    total_time:   float

    by_category:   dict[str, CategoryMetrics] = field(default_factory=dict)
    by_difficulty: dict[str, dict]            = field(default_factory=dict)


# ── Calculator ────────────────────────────────────────────────────────────────


class MetricsCalculator:
    """Computes BenchmarkSummary from a list of BenchmarkResult objects."""

    @staticmethod
    def compute(results: list, run_id: str = "") -> BenchmarkSummary:
        if not results:
            return BenchmarkSummary(
                run_id=run_id, total_tasks=0, passed=0, partial=0, failed=0,
                task_success_rate=0, partial_success_rate=0, recovery_rate=0,
                avg_steps_per_task=0, avg_corrections_per_task=0,
                avg_tokens_per_task=0, avg_time_per_task=0,
                total_tokens=0, total_time=0,
            )

        effective_run_id = run_id or (results[0].run_id if results else "")

        total   = len(results)
        passed  = sum(1 for r in results if r.status == "pass")
        partial = sum(1 for r in results if r.status == "partial")
        failed  = sum(1 for r in results if r.status == "fail")

        total_steps       = sum(r.steps_taken  for r in results)
        total_corrections = sum(r.corrections  for r in results)
        total_tokens      = sum(r.tokens_used  for r in results)
        total_time        = sum(r.time_seconds for r in results)

        # Recovery rate — scoped to this run only
        recovery_rate = MetricsCalculator._compute_recovery_rate(effective_run_id)

        # Per-category
        by_category: dict[str, CategoryMetrics] = {}
        cat_groups: dict[str, list] = defaultdict(list)
        for r in results:
            cat_groups[r.category].append(r)

        for cat, group in cat_groups.items():
            by_category[cat] = CategoryMetrics(
                category=cat,
                total=len(group),
                passed=sum(1 for r in group if r.status == "pass"),
                partial=sum(1 for r in group if r.status == "partial"),
                failed=sum(1 for r in group if r.status == "fail"),
                avg_steps=sum(r.steps_taken  for r in group) / len(group),
                avg_tokens=sum(r.tokens_used  for r in group) / len(group),
                avg_time=sum(r.time_seconds  for r in group) / len(group),
            )

        # Per-difficulty
        by_difficulty: dict[str, dict] = {}
        diff_groups: dict[str, list] = defaultdict(list)
        for r in results:
            diff_groups[r.difficulty].append(r)

        for diff, group in diff_groups.items():
            by_difficulty[diff] = {
                "total":     len(group),
                "passed":    sum(1 for r in group if r.status == "pass"),
                "pass_rate": sum(1 for r in group if r.status == "pass") / len(group),
            }

        return BenchmarkSummary(
            run_id=effective_run_id,
            total_tasks=total,
            passed=passed,
            partial=partial,
            failed=failed,
            task_success_rate=passed / total,
            partial_success_rate=partial / total,
            recovery_rate=recovery_rate,
            avg_steps_per_task=total_steps / total,
            avg_corrections_per_task=total_corrections / total,
            avg_tokens_per_task=total_tokens / total,
            avg_time_per_task=total_time / total,
            total_tokens=total_tokens,
            total_time=total_time,
            by_category=by_category,
            by_difficulty=by_difficulty,
        )

    # ── DB helpers — all scoped to run_id via task_id ─────────────────────────

    @staticmethod
    def _task_ids_for_run(run_id: str) -> list[str]:
        """
        Return the task UUIDs that belong to this benchmark run.
        Reads from benchmark_results.task_id which is written by
        BenchmarkRunner._write_result at the end of each task.
        Returns empty list if run_id is unknown or has no task_ids recorded.
        """
        if not run_id:
            return []
        try:
            from memory.db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT task_id FROM benchmark_results WHERE run_id = ? AND task_id != ''",
                    (run_id,),
                ).fetchall()
            return [r["task_id"] for r in rows]
        except Exception:
            return []

    @staticmethod
    def _compute_recovery_rate(run_id: str) -> float:
        """
        Recovery rate = corrections that succeeded / total corrections attempted.
        Scoped to this run_id only. Returns 0.0 if no corrections recorded.
        """
        if not run_id:
            return 0.0
        try:
            from memory.db import get_connection
            task_ids = MetricsCalculator._task_ids_for_run(run_id)
            if not task_ids:
                return 0.0

            placeholders = ",".join("?" * len(task_ids))
            with get_connection() as conn:
                rows = conn.execute(
                    f"SELECT was_recovered FROM action_failures WHERE task_id IN ({placeholders})",
                    task_ids,
                ).fetchall()

            if not rows:
                return 0.0
            recovered = sum(1 for r in rows if r["was_recovered"])
            return recovered / len(rows)
        except Exception:
            return 0.0

    @staticmethod
    def failure_breakdown(results: list, run_id: str = "") -> dict[str, int]:
        """
        Count failures by type from action_failures — scoped to this run.

        Args:
            results: BenchmarkResult list (used to derive run_id if not passed).
            run_id:  The benchmark run_id. Preferred over deriving from results.

        Returns empty dict if no failures recorded for this run.
        """
        effective_run_id = run_id or (results[0].run_id if results else "")
        if not effective_run_id:
            return {}
        try:
            from memory.db import get_connection
            task_ids = MetricsCalculator._task_ids_for_run(effective_run_id)
            if not task_ids:
                return {}

            placeholders = ",".join("?" * len(task_ids))
            with get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT failure_type, COUNT(*) as cnt
                    FROM action_failures
                    WHERE task_id IN ({placeholders})
                    GROUP BY failure_type
                    """,
                    task_ids,
                ).fetchall()
            return {r["failure_type"]: r["cnt"] for r in rows}
        except Exception:
            return {}

    @staticmethod
    def correction_effectiveness(results: list, run_id: str = "") -> dict[str, dict]:
        """
        Per-strategy correction effectiveness — scoped to this run.

        Returns: {strategy: {attempts, successes, success_rate}}
        Returns empty dict if no corrections recorded for this run.

        Args:
            results: BenchmarkResult list (used to derive run_id if not passed).
            run_id:  The benchmark run_id. Preferred over deriving from results.
        """
        effective_run_id = run_id or (results[0].run_id if results else "")
        if not effective_run_id:
            return {}
        try:
            from memory.db import get_connection
            task_ids = MetricsCalculator._task_ids_for_run(effective_run_id)
            if not task_ids:
                return {}

            placeholders = ",".join("?" * len(task_ids))
            with get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT correction_used,
                           COUNT(*) as attempts,
                           SUM(was_recovered) as successes
                    FROM action_failures
                    WHERE task_id IN ({placeholders})
                    GROUP BY correction_used
                    """,
                    task_ids,
                ).fetchall()

            result: dict[str, dict] = {}
            for r in rows:
                attempts  = r["attempts"]
                successes = r["successes"] or 0
                result[r["correction_used"]] = {
                    "attempts":     attempts,
                    "successes":    successes,
                    "success_rate": successes / attempts if attempts else 0.0,
                }
            return result
        except Exception:
            return {}