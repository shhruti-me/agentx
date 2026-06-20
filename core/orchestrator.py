"""
core/orchestrator.py

Orchestrator — the single top-level coordinator for AGENTX.

Receives a Task, runs it to completion, returns the result.

Pipeline:
  1. Mark task running, write to DB
  2. Open browser
  3. Call Planner → get ExecutionDAG
  4. Call ExecutionEngine.run(dag)
  5. Mark task completed or failed, write final state to DB
  6. Close browser

The Orchestrator owns task lifecycle and pipeline coordination.
It does NOT own planning logic, browser control, or verification.

Usage (from main.py and api/routes.py):
    orchestrator = Orchestrator()
    task = await orchestrator.run("find the top HN post", goal_type="navigation")
    print(task.result)
"""

from __future__ import annotations

from core.models import GoalType, Task, TaskStatus
from core.planner import Planner, PlannerError
from core.execution_engine import ExecutionEngine
from browser.controller import BrowserController
from memory.task_memory import write_task, update_task
from config.settings import settings
from log.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """
    Runs one task at a time through the full agent pipeline.

    Stateless between tasks — safe to reuse for multiple sequential tasks.
    Not safe for concurrent use (browser sessions are not shared).
    """

    def __init__(self) -> None:
        self._planner = Planner()

    async def run(self, goal: str, goal_type: str = "unknown") -> Task:
        """
        Execute a goal end-to-end and return the completed Task.

        Parameters
        ----------
        goal      : Natural language goal string.
        goal_type : Category hint for the Planner and memory queries.

        Returns
        -------
        Task with status=COMPLETED and result set,
        or status=FAILED and result containing the error reason.
        """
        # Resolve goal type
        try:
            gt = GoalType(goal_type)
        except ValueError:
            gt = GoalType.UNKNOWN

        # Create and persist task
        task = Task(goal=goal, goal_type=gt)
        write_task(task)

        logger.info(
            "orchestrator_start",
            task_id=task.id,
            goal=goal[:80],
            goal_type=gt,
        )

        # Run pipeline inside browser session
        async with BrowserController() as browser:
            try:
                await self._run_pipeline(task=task, browser=browser)
            except Exception as exc:
                # Top-level catch — nothing should escape to the caller
                logger.error(
                    "orchestrator_unhandled_error",
                    task_id=task.id,
                    error=str(exc),
                )
                task.mark_failed(f"Unhandled error: {exc}")
                update_task(task)

        logger.info(
            "orchestrator_done",
            task_id=task.id,
            status=task.status,
            steps_taken=task.steps_taken,
            corrections=task.corrections,
            tokens_used=task.tokens_used,
            result_preview=(task.result or "")[:100],
        )

        return task

    async def _run_pipeline(self, task: Task, browser: BrowserController) -> None:
        """
        Inner pipeline. Exceptions propagate to run() which catches them.
        """
        task.mark_running()
        update_task(task)

        # ── Step 1: Plan ───────────────────────────────────────────────────────
        logger.info("pipeline_planning", task_id=task.id)
        try:
            dag, tokens = await self._planner.plan(
                goal=task.goal,
                goal_type=task.goal_type,
            )
        except PlannerError as exc:
            logger.error("pipeline_plan_failed", task_id=task.id, error=str(exc))
            task.mark_failed(f"Planning failed: {exc}")
            update_task(task)
            return

        task.plan = dag
        task.tokens_used += tokens
        update_task(task)

        logger.info(
            "pipeline_plan_ready",
            task_id=task.id,
            steps=len(dag),
        )

        # ── Step 2: Execute ────────────────────────────────────────────────────
        engine = ExecutionEngine(
            browser=browser,
            task=task,
            max_corrections=settings.max_corrections_per_task,
        )

        success, result = await engine.run(dag)

        task.steps_taken = engine._task.steps_taken
        task.corrections = engine._task.corrections

        # ── Step 3: Finalise ───────────────────────────────────────────────────
        if success:
            task.mark_completed(result or "Task completed successfully (no text output)")
        else:
            task.mark_failed(result or "Task failed (unknown reason)")

        update_task(task)