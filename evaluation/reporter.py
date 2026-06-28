"""
evaluation/reporter.py

Formats benchmark results into a readable report.

Prints to terminal and saves to evaluation/reports/<run_id>.txt.
The saved report goes in the README as evidence of benchmark performance.

Report format matches Section 10 of the architecture document exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from evaluation.metrics import BenchmarkSummary, MetricsCalculator
from log.logger import get_logger

logger = get_logger(__name__)


class Reporter:
    """Formats and saves benchmark reports."""

    def print_report(
        self,
        summary: BenchmarkSummary,
        results: list,
    ) -> None:
        """Print the full report to stdout."""
        lines = self._build_report(summary, results)
        print("\n".join(lines))

    def save_report(
        self,
        summary: BenchmarkSummary,
        results: list,
        run_id: str,
    ) -> Path:
        """
        Save report to evaluation/reports/<run_id>.txt
        Returns the path written.
        """
        reports_dir = settings.benchmark_reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"{run_id}.txt"
        lines = self._build_report(summary, results)
        report_path.write_text("\n".join(lines), encoding="utf-8")

        # Also save raw results as JSON for programmatic access
        json_path = reports_dir / f"{run_id}.json"
        json_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "summary": {
                        "total_tasks": summary.total_tasks,
                        "passed": summary.passed,
                        "partial": summary.partial,
                        "failed": summary.failed,
                        "task_success_rate": round(summary.task_success_rate * 100, 1),
                        "partial_success_rate": round(summary.partial_success_rate * 100, 1),
                        "recovery_rate": round(summary.recovery_rate * 100, 1),
                        "avg_steps_per_task": round(summary.avg_steps_per_task, 1),
                        "avg_corrections_per_task": round(summary.avg_corrections_per_task, 2),
                        "avg_tokens_per_task": round(summary.avg_tokens_per_task),
                        "avg_time_per_task": round(summary.avg_time_per_task, 1),
                        "total_tokens": summary.total_tokens,
                        "total_time_seconds": round(summary.total_time, 1),
                    },
                    "by_category": {
                        cat: {
                            "total": m.total,
                            "passed": m.passed,
                            "pass_rate": round(m.pass_rate * 100, 1),
                            "avg_steps": round(m.avg_steps, 1),
                            "avg_time": round(m.avg_time, 1),
                        }
                        for cat, m in summary.by_category.items()
                    },
                    "results": [
                        {
                            "id": r.task_name,
                            "category": r.category,
                            "difficulty": r.difficulty,
                            "status": r.status,
                            "steps": r.steps_taken,
                            "corrections": r.corrections,
                            "tokens": r.tokens_used,
                            "time_s": r.time_seconds,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info("report_saved", path=str(report_path))
        print(f"\n  Report saved to: {report_path}")
        return report_path

    def _build_report(
        self,
        summary: BenchmarkSummary,
        results: list,
    ) -> list[str]:
        """Build the report as a list of lines."""
        W = 56  # report width
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines: list[str] = []
        lines.append("")
        lines.append("  " + "═" * W)
        lines.append("  AGENTX Benchmark Report")
        lines.append(f"  Run ID:  {summary.run_id}")
        lines.append(f"  Date:    {now}")
        lines.append(f"  Tasks:   {summary.total_tasks}")
        lines.append("  " + "═" * W)

        # ── Overall ───────────────────────────────────────────────────
        lines.append("")
        lines.append("  OVERALL")
        lines.append("  " + "─" * W)
        lines.append(
            f"  Task Success Rate      "
            f"{summary.task_success_rate * 100:5.1f}%"
            f"   ({summary.passed}/{summary.total_tasks})"
        )
        lines.append(
            f"  Partial Success Rate   "
            f"{summary.partial_success_rate * 100:5.1f}%"
            f"   ({summary.partial}/{summary.total_tasks})"
        )
        lines.append(
            f"  Recovery Rate          "
            f"{summary.recovery_rate * 100:5.1f}%"
            f"   (corrections that worked)"
        )
        lines.append(
            f"  Avg Steps / Task       {summary.avg_steps_per_task:5.1f}"
        )
        lines.append(
            f"  Avg Corrections / Task {summary.avg_corrections_per_task:5.2f}"
        )
        lines.append(
            f"  Avg Tokens / Task      {summary.avg_tokens_per_task:7,.0f}"
        )
        lines.append(
            f"  Avg Time / Task        {summary.avg_time_per_task:5.1f}s"
        )
        lines.append(
            f"  Total Tokens           {summary.total_tokens:7,}"
        )
        lines.append(
            f"  Total Time             {summary.total_time:5.1f}s"
        )

        # ── By category ───────────────────────────────────────────────
        if summary.by_category:
            lines.append("")
            lines.append("  BY CATEGORY")
            lines.append("  " + "─" * W)
            for cat, m in sorted(summary.by_category.items()):
                lines.append(
                    f"  {cat:<22s} {m.passed}/{m.total}"
                    f"  {m.pass_rate * 100:5.1f}%"
                    f"   avg {m.avg_steps:.1f} steps"
                    f"  {m.avg_time:.1f}s"
                )

        # ── By difficulty ─────────────────────────────────────────────
        if summary.by_difficulty:
            lines.append("")
            lines.append("  BY DIFFICULTY")
            lines.append("  " + "─" * W)
            for diff in ("easy", "medium", "hard"):
                if diff in summary.by_difficulty:
                    d = summary.by_difficulty[diff]
                    lines.append(
                        f"  {diff:<10s}  {d['passed']}/{d['total']}"
                        f"  {d['pass_rate'] * 100:5.1f}%"
                    )

        # ── Failure breakdown ─────────────────────────────────────────
        failure_counts = MetricsCalculator.failure_breakdown(results)
        if failure_counts:
            total_failures = sum(failure_counts.values())
            lines.append("")
            lines.append("  FAILURE BREAKDOWN")
            lines.append("  " + "─" * W)
            for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                pct = count / total_failures * 100 if total_failures else 0
                lines.append(f"  {ftype:<22s} {count:3d}   ({pct:.0f}%)")

        # ── Correction effectiveness ───────────────────────────────────
        correction_eff = MetricsCalculator.correction_effectiveness(results)
        if correction_eff:
            lines.append("")
            lines.append("  CORRECTION EFFECTIVENESS")
            lines.append("  " + "─" * W)
            for strategy, stats in sorted(correction_eff.items()):
                lines.append(
                    f"  {strategy:<14s}"
                    f"  {stats['attempts']:3d} attempts"
                    f"  {stats['successes']:3d} success"
                    f"  {stats['success_rate'] * 100:5.1f}%"
                )

        # ── Per-task detail ───────────────────────────────────────────
        lines.append("")
        lines.append("  TASK DETAIL")
        lines.append("  " + "─" * W)
        for r in results:
            icon = "✓" if r.status == "pass" else ("~" if r.status == "partial" else "✗")
            lines.append(
                f"  {icon} {r.task_name:<14s}"
                f"  {r.status:<8s}"
                f"  {r.steps_taken:2d}steps"
                f"  {r.corrections:2d}corr"
                f"  {r.time_seconds:5.1f}s"
            )

        lines.append("")
        lines.append("  " + "═" * W)
        lines.append("")

        return lines