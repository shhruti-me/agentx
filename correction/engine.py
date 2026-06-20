"""
correction/engine.py

Self-Correction Engine — classifies failures and selects recovery strategies.

Week 3 stub: retries once, then aborts.
Week 4: full failure taxonomy + all four strategies (retry, selector_fix, replan, abort).
"""

from __future__ import annotations

from core.models import (
    ActionResult,
    CorrectionResult,
    CorrectionStrategy,
    ExecutionDAG,
    Step,
    VerificationResult,
)
from browser.controller import BrowserController
from memory.retrieval import get_correction_context
from log.logger import get_logger

logger = get_logger(__name__)

# Maximum retries before aborting (Week 3 stub behaviour)
_MAX_RETRIES = 2


class SelfCorrectionEngine:
    """
    Receives a failed step and attempts to recover.

    Week 3 stub behaviour:
      - Retry up to _MAX_RETRIES times
      - Abort if retries exhausted

    Week 4 full behaviour:
      - Classify failure type (stale_selector, timeout, auth_wall, etc.)
      - Select strategy based on history (skip already-tried strategies)
      - Execute: retry / selector_fix / replan / abort
    """

    def __init__(self, browser: BrowserController, task) -> None:
        self._browser = browser
        self._task = task

    async def correct(
        self,
        step: Step,
        verification: VerificationResult,
        action_result: ActionResult,
        dag: ExecutionDAG,
    ) -> CorrectionResult:
        """
        Attempt to recover from a failed step.

        Parameters
        ----------
        step         : The step that failed.
        verification : The verification result (FAIL).
        action_result: The raw action output.
        dag          : Current execution DAG (for replan context).

        Returns
        -------
        CorrectionResult describing what was tried and whether it worked.
        """
        # Check correction history for this tool in this task
        ctx = get_correction_context(tool=step.tool, task_id=self._task.id)

        logger.info(
            "correction_start",
            tool=step.tool,
            attempt=ctx.attempt_count + 1,
            already_tried=list(ctx.already_tried),
            task_id=self._task.id,
        )

        # Week 3 stub: retry up to _MAX_RETRIES, then abort
        if step.retry_count < _MAX_RETRIES and CorrectionStrategy.RETRY not in ctx.already_tried:
            step.retry_count += 1
            logger.info(
                "correction_retry",
                tool=step.tool,
                retry_count=step.retry_count,
                task_id=self._task.id,
            )
            # Reset step status so engine re-dispatches it
            from core.models import StepStatus
            step.status = StepStatus.PENDING
            step.error_message = None

            return CorrectionResult(
                strategy_used=CorrectionStrategy.RETRY,
                success=True,
                reason=f"Retrying step (attempt {step.retry_count})",
            )

        # Retries exhausted → abort
        reason = (
            action_result.error
            or verification.reason
            or f"Step failed after {step.retry_count} retries"
        )
        logger.warning(
            "correction_abort",
            tool=step.tool,
            reason=reason[:200],
            task_id=self._task.id,
        )

        return CorrectionResult(
            strategy_used=CorrectionStrategy.ABORT,
            success=False,
            reason=reason,
        )