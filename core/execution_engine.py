"""
core/execution_engine.py

Execution Engine — walks the ExecutionDAG step by step.

For each step:
  1. Resolve tool from registry
  2. Dispatch to handler (browser action)
  3. Pass result to Verifier
  4. On PASS: write to memory, move to next step
  5. On FAIL: hand to Self-Correction Engine

V1: sequential execution only.
V2: parallel branches via asyncio.gather — DAG structure already supports it.

The Engine owns: DAG traversal, step dispatch, retry state, step timing.
The Engine does NOT own: planning, browser control, verification logic.
"""

from __future__ import annotations

import asyncio
import time

from core.models import (
    ActionResult,
    CorrectionStrategy,
    ExecutionDAG,
    FailureType,
    Step,
    StepStatus,
    Task,
    VerificationStatus,
)
from browser.controller import BrowserController
from tools.registry import get_tool
from memory.action_memory import write_success, domain_from_url
from memory.failure_memory import write_failure
from memory.task_memory import write_task
from log.logger import get_logger

logger = get_logger(__name__)

# Imported lazily to avoid circular imports at module load time
# verification and correction are imported inside methods


class ExecutionEngine:
    """
    Walks an ExecutionDAG and executes each step against a live browser.

    One instance per task run. Holds references to the browser and task
    for the duration of execution.

    Parameters
    ----------
    browser : Active BrowserController (already started).
    task    : The Task this engine is executing.
    max_corrections : Abort after this many self-correction events.
    """

    def __init__(
        self,
        browser: BrowserController,
        task: Task,
        max_corrections: int = 6,
    ) -> None:
        self._browser = browser
        self._task = task
        self._max_corrections = max_corrections
        self._correction_count = 0

    async def run(self, dag: ExecutionDAG) -> tuple[bool, str]:
        """
        Execute all steps in the DAG sequentially.

        Returns
        -------
        (success: bool, result: str)
          success=True  → all steps completed, result is the final extracted value
          success=False → task failed, result is the error reason
        """
        logger.info(
            "engine_start",
            task_id=self._task.id,
            steps=len(dag),
        )

        final_result = ""

        for step_index, step in enumerate(dag.steps):
            step.task_id = self._task.id
            step.step_index = step_index

            logger.info(
                "step_start",
                task_id=self._task.id,
                step_index=step_index,
                tool=step.tool,
                input=step.input,
            )

            # Check correction budget
            if self._correction_count >= self._max_corrections:
                reason = f"Correction budget exhausted ({self._max_corrections} corrections used)"
                logger.warning("engine_abort_budget", task_id=self._task.id)
                return False, reason

            # Execute with retry loop
            success, result_text = await self._execute_step(step, dag)

            if not success:
                return False, result_text

            # Track the last meaningful text output as the candidate result
            if result_text:
                final_result = result_text

            # Update task counters
            self._task.steps_taken = step_index + 1
            self._task.corrections = self._correction_count

        logger.info(
            "engine_complete",
            task_id=self._task.id,
            steps_taken=self._task.steps_taken,
            corrections=self._correction_count,
        )

        return True, final_result

    async def _execute_step(
        self, step: Step, dag: ExecutionDAG
    ) -> tuple[bool, str]:
        """
        Execute one step, verify, and handle failures.

        Returns (success, result_text).
        result_text is the extracted text on success, error reason on failure.
        """
        from verification.verifier import Verifier
        from correction.engine import SelfCorrectionEngine

        verifier = Verifier(browser=self._browser)
        corrector = SelfCorrectionEngine(browser=self._browser, task=self._task)

        step.mark_running()

        # Resolve tool
        tool_def = get_tool(step.tool)
        if tool_def is None:
            step.mark_failed(f"Unknown tool: {step.tool!r}")
            logger.error("step_unknown_tool", tool=step.tool, task_id=self._task.id)
            return False, f"Unknown tool: {step.tool!r}"

        # Dispatch action
        action_result = await _dispatch(tool_def, self._browser, step)

        # Verify outcome
        verification = await verifier.verify(step=step, actual=action_result)

        if verification.passed:
            step.mark_completed(action_result.output or {})
            result_text = _extract_text(action_result)
            self._write_success(step, action_result)

            logger.info(
                "step_completed",
                task_id=self._task.id,
                step_index=step.step_index,
                tool=step.tool,
                method=verification.method,
                duration_ms=action_result.duration_ms,
            )
            return True, result_text

        # Verification failed — attempt correction
        logger.warning(
            "step_failed",
            task_id=self._task.id,
            step_index=step.step_index,
            tool=step.tool,
            reason=verification.reason,
            action_error=action_result.error,
        )

        correction = await corrector.correct(
            step=step,
            verification=verification,
            action_result=action_result,
            dag=dag,
        )
        self._correction_count += 1
        self._task.corrections = self._correction_count

        self._write_failure(step, action_result, verification, correction.strategy_used)

        if correction.strategy_used == CorrectionStrategy.ABORT:
            step.mark_failed(f"Aborted: {correction.reason}")
            return False, f"Step aborted: {correction.reason}"

        if correction.strategy_used == CorrectionStrategy.REPLAN and correction.new_dag:
            # Replace remaining steps with the new plan and re-index them
            current_idx = step.step_index
            new_steps = correction.new_dag.steps
            for i, s in enumerate(new_steps):
                s.step_index = current_idx + i
                s.task_id = self._task.id
            dag.steps = dag.steps[: current_idx] + new_steps
            step.mark_failed(f"Replanned: {correction.reason}")
            # Execute the new steps immediately
            replan_result = ""
            for new_step in new_steps:
                success, result = await self._execute_step(new_step, dag)
                if not success:
                    return False, result
                if result:
                    replan_result = result
            return True, replan_result

        if correction.success:
            # RETRY or SELECTOR_FIX succeeded — re-run this step
            return await self._execute_step(step, dag)

        step.mark_failed(correction.reason)
        return False, f"Step failed after correction: {correction.reason}"

    def _write_success(self, step: Step, result: ActionResult) -> None:
        """Write successful action to memory for future planning context."""
        try:
            url = self._browser._page.url if self._browser._page else ""
            write_success(
                tool=step.tool,
                context=f"Step {step.step_index}: {step.expected[:100]}",
                input_data=step.input,
                output_data=result.output or {},
                goal_type=self._task.goal_type,
                site_domain=domain_from_url(url),
            )
        except Exception as exc:
            logger.warning("memory_write_success_failed", error=str(exc))

    def _write_failure(
        self,
        step: Step,
        result: ActionResult,
        verification,
        strategy_used,
    ) -> None:
        """Write failure event to memory so future corrections skip tried strategies."""
        try:
            write_failure(
                tool=step.tool,
                failure_type=_classify_failure_type(result, verification),
                input_data=step.input,
                error_message=result.error or verification.reason,
                correction_used=strategy_used,
                was_recovered=False,  # updated by corrector if recovery works
                task_id=self._task.id,
            )
        except Exception as exc:
            logger.warning("memory_write_failure_failed", error=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _dispatch(tool_def, browser: BrowserController, step: Step) -> ActionResult:
    """Call the tool handler with kwargs unpacked from step.input."""
    try:
        return await tool_def.handler(browser, **step.input)
    except TypeError as exc:
        # Wrong kwargs — plan generated bad input schema
        return ActionResult.fail(
            error=f"Tool {step.tool!r} received unexpected input {step.input}: {exc}"
        )
    except Exception as exc:
        return ActionResult.fail(error=f"Tool {step.tool!r} raised: {exc}")


def _extract_text(result: ActionResult) -> str:
    """Pull the most useful text string from an ActionResult output."""
    if not result.output:
        return ""
    # Prefer explicit 'text' key, then 'title', then first string value
    if "text" in result.output:
        return str(result.output["text"])
    if "title" in result.output:
        return str(result.output["title"])
    for v in result.output.values():
        if isinstance(v, str) and v:
            return v
    return ""


def _classify_failure_type(result: ActionResult, verification) -> str:
    """
    Quick failure classification without the full Correction Engine.
    Used when writing to failure memory before correction runs.
    """
    error = (result.error or "").lower()
    if "not found" in error or "selector" in error:
        return FailureType.STALE_SELECTOR
    if "timeout" in error:
        return FailureType.TIMEOUT
    if "login" in error or "captcha" in error:
        return FailureType.AUTH_WALL
    if "empty" in error or result.output == {}:
        return FailureType.EXTRACTION_EMPTY
    return FailureType.UNKNOWN