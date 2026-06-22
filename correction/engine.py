"""
correction/engine.py

Self-Correction Engine — classifies failures and dispatches recovery strategies.

Flow for each failure:
  1. Get correction history for this tool in this task (from memory)
  2. Classify the failure type
  3. Select the best untried strategy for this failure type
  4. Execute the strategy
  5. Return CorrectionResult to the Execution Engine

Strategy selection table:
  stale_selector   → selector_fix → retry → replan → abort
  timeout          → retry → retry → replan → abort
  auth_wall        → abort (immediate, no recovery possible)
  wrong_page       → replan → abort
  extraction_empty → selector_fix → replan → abort
  plan_error       → replan → abort
  unknown          → retry → selector_fix → replan → abort
"""

from __future__ import annotations

from core.models import (
    ActionResult,
    CorrectionResult,
    CorrectionStrategy,
    ExecutionDAG,
    FailureType,
    Step,
    VerificationResult,
)
from browser.controller import BrowserController
from correction.classifier import classify
from correction.strategies import (
    strategy_retry,
    strategy_selector_fix,
    strategy_replan,
    strategy_abort,
)
from memory.retrieval import get_correction_context
from config.settings import settings
from log.logger import get_logger

logger = get_logger(__name__)

# Strategy priority order per failure type.
# The engine walks this list and picks the first untried strategy.
_STRATEGY_MAP: dict[str, list[CorrectionStrategy]] = {
    FailureType.STALE_SELECTOR:   [CorrectionStrategy.SELECTOR_FIX,
                                   CorrectionStrategy.RETRY,
                                   CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
    FailureType.TIMEOUT:          [CorrectionStrategy.RETRY,
                                   CorrectionStrategy.SELECTOR_FIX,  # timeout on extract = likely stale selector
                                   CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
    FailureType.AUTH_WALL:        [CorrectionStrategy.ABORT],
    FailureType.WRONG_PAGE:       [CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
    FailureType.EXTRACTION_EMPTY: [CorrectionStrategy.SELECTOR_FIX,
                                   CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
    FailureType.PLAN_ERROR:       [CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
    FailureType.UNKNOWN:          [CorrectionStrategy.RETRY,
                                   CorrectionStrategy.SELECTOR_FIX,
                                   CorrectionStrategy.REPLAN,
                                   CorrectionStrategy.ABORT],
}


class SelfCorrectionEngine:
    """
    Selects and executes the best recovery strategy for a failed step.

    One instance per task. Holds browser and task references.
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

        Returns CorrectionResult describing what was tried.
        Never raises — returns ABORT on any unexpected error.
        """
        # Get current page text for classification
        page_text = ""
        if self._browser and self._browser.is_running:
            try:
                page_result = await self._browser.extract_page()
                if page_result.success:
                    page_text = page_result.output.get("text", "")
            except Exception:
                pass

        # Classify the failure
        failure_type = classify(
            action_result=action_result,
            verification=verification,
            step_tool=step.tool,
            page_text=page_text,
        )

        # Get history of what's already been tried for this tool/task
        ctx = get_correction_context(tool=step.tool, task_id=self._task.id)
        already_tried = ctx.already_tried

        logger.info(
            "correction_start",
            tool=step.tool,
            failure_type=failure_type,
            attempt=ctx.attempt_count + 1,
            already_tried=list(already_tried),
            task_id=self._task.id,
        )

        # Select strategy
        strategy = _select_strategy(failure_type, already_tried, step)

        logger.info(
            "correction_strategy_selected",
            strategy=strategy,
            failure_type=failure_type,
            task_id=self._task.id,
        )

        # Dispatch strategy
        return await self._dispatch(
            strategy=strategy,
            step=step,
            action_result=action_result,
            verification=verification,
            dag=dag,
        )

    async def _dispatch(
        self,
        strategy: CorrectionStrategy,
        step: Step,
        action_result: ActionResult,
        verification: VerificationResult,
        dag: ExecutionDAG,
    ) -> CorrectionResult:
        """Execute the selected strategy and return result."""
        try:
            if strategy == CorrectionStrategy.RETRY:
                return await strategy_retry(
                    step=step,
                    action_result=action_result,
                    verification=verification,
                    browser=self._browser,
                    task=self._task,
                )

            elif strategy == CorrectionStrategy.SELECTOR_FIX:
                return await strategy_selector_fix(
                    step=step,
                    action_result=action_result,
                    verification=verification,
                    browser=self._browser,
                    task=self._task,
                )

            elif strategy == CorrectionStrategy.REPLAN:
                return await strategy_replan(
                    step=step,
                    action_result=action_result,
                    verification=verification,
                    browser=self._browser,
                    task=self._task,
                    dag=dag,
                )

            else:  # ABORT
                reason = (
                    action_result.error
                    or verification.reason
                    or "No viable recovery strategy remaining"
                )
                return await strategy_abort(
                    step=step,
                    action_result=action_result,
                    verification=verification,
                    reason=reason,
                    task=self._task,
                )

        except Exception as exc:
            logger.error(
                "correction_dispatch_error",
                strategy=strategy,
                error=str(exc),
                task_id=self._task.id,
            )
            return CorrectionResult(
                strategy_used=CorrectionStrategy.ABORT,
                success=False,
                reason=f"Correction engine error: {exc}",
            )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _select_strategy(
    failure_type: str,
    already_tried: set[str],
    step: Step,
) -> CorrectionStrategy:
    """
    Pick the first untried strategy from the failure type's priority list.

    Falls back to ABORT if all strategies have been tried.
    Skips SELECTOR_FIX if the step tool doesn't use a selector.
    """
    priority_list = _STRATEGY_MAP.get(failure_type, _STRATEGY_MAP[FailureType.UNKNOWN])

    for strategy in priority_list:
        # Skip selector_fix if this tool doesn't have a selector input
        if (strategy == CorrectionStrategy.SELECTOR_FIX
                and "selector" not in step.input):
            continue

        # Skip if already tried in this task session
        if strategy in already_tried and strategy != CorrectionStrategy.RETRY:
            continue

        # For RETRY: allow up to max_retries_per_step attempts
        # But if already_tried contains RETRY, it means we already retried
        # at least once this correction cycle - move to next strategy
        if strategy == CorrectionStrategy.RETRY:
            if CorrectionStrategy.RETRY in already_tried:
                continue  # already retried - try next strategy
            if step.retry_count < settings.max_retries_per_step:
                return strategy
            continue

        return strategy

    return CorrectionStrategy.ABORT