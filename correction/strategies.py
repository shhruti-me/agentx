"""
correction/strategies.py

The four correction strategies as standalone async functions.

Each strategy receives the failed step and browser context,
attempts recovery, and returns a CorrectionResult.

Strategy selection logic lives in correction/engine.py.
Implementation logic lives here.

Adding a new strategy:
  1. Add a function here following the same signature
  2. Add one case in engine.py strategy selection
  Zero changes anywhere else.
"""

from __future__ import annotations

import json

from core.models import (
    ActionResult,
    CorrectionResult,
    CorrectionStrategy,
    ExecutionDAG,
    FailureType,
    Step,
    StepStatus,
    VerificationResult,
)
from log.logger import get_logger

logger = get_logger(__name__)


# ── RETRY ─────────────────────────────────────────────────────────────────────


async def strategy_retry(
    step: Step,
    action_result: ActionResult,
    verification: VerificationResult,
    browser,
    task,
) -> CorrectionResult:
    """
    Re-run the exact same step with no changes.

    Best for: transient failures - network blips, race conditions,
    slow page loads that just needed more time.
    """
    step.retry_count += 1
    step.status = StepStatus.PENDING
    step.error_message = None

    logger.info(
        "strategy_retry",
        tool=step.tool,
        retry_count=step.retry_count,
        task_id=task.id,
    )

    return CorrectionResult(
        strategy_used=CorrectionStrategy.RETRY,
        success=True,
        reason=f"Retrying step (attempt {step.retry_count})",
    )


# ── SELECTOR_FIX ──────────────────────────────────────────────────────────────


async def strategy_selector_fix(
    step: Step,
    action_result: ActionResult,
    verification: VerificationResult,
    browser,
    task,
) -> CorrectionResult:
    """
    Ask the LLM to suggest alternative selectors using live page context.

    Best for: stale_selector and timeout failures where the element exists
    but the CSS path has changed (HN .storylink -> .titleline > a, etc.)

    Uses page HTML source as context since it is more reliable than the
    DOM snapshot for table-heavy or dynamically structured sites.
    """
    from llm.factory import get_llm_client

    logger.info(
        "strategy_selector_fix",
        tool=step.tool,
        original_selector=step.input.get("selector", ""),
        task_id=task.id,
    )

    # Get page context - try DOM snapshot first, fall back to HTML source
    dom_snapshot = ""
    page_source = ""
    page_text = ""

    if browser and browser.is_running:
        snap = await browser.get_dom_snapshot()
        if snap.success and len(snap.output.get("snapshot", "")) > 100:
            dom_snapshot = snap.output.get("snapshot", "")

        try:
            page_source = await browser._page.content()
        except Exception:
            pass

        page_result = await browser.extract_page()
        if page_result.success:
            page_text = page_result.output.get("text", "")

    context = dom_snapshot or page_source[:4000] or page_text[:2000]
    if not context:
        logger.warning("selector_fix_no_dom", task_id=task.id)
        return CorrectionResult(
            strategy_used=CorrectionStrategy.SELECTOR_FIX,
            success=False,
            reason="Could not get page context for selector fix",
        )

    original_selector = step.input.get("selector", "")
    context_label = "DOM structure" if dom_snapshot else "page HTML source"

    prompt_lines = [
        "A CSS selector failed and needs to be replaced with a working alternative.",
        "",
        f"BROKEN selector (do NOT return this): {original_selector!r}",
        f"Goal: {step.expected}",
        f"Error: {action_result.error or verification.reason}",
        "",
        f"Current page {context_label}:",
        context[:4000],
        "",
        "Find a DIFFERENT CSS selector that will extract the requested content.",
        "CRITICAL RULES:",
        f"1. You MUST return a selector different from {original_selector!r}",
        "2. Target the <a> tag that contains the link text, not a wrapper element",
        "3. Look for class names describing the content type in the page above",
        "4. For news sites: look for classes like titleline, title, story, post-title, entry-title",
        "5. Verify mentally: document.querySelectorAll(your_selector) should return link text",
        "",
        '{"selector": "DIFFERENT-selector-here", "reason": "one sentence why this works"}',
        "Return ONLY the JSON. The selector field MUST differ from the broken one.",
    ]
    prompt = "\n".join(prompt_lines)

    try:
        client = get_llm_client()
        response = await client.complete(
            prompt=prompt,
            system="You are a browser automation expert. Respond only with the JSON object requested.",
            max_tokens=150,
            temperature=0.0,
        )
        data = _parse_json(response.content)
        new_selector = data.get("selector", "").strip()
        fix_reason = data.get("reason", "LLM suggested alternative")

        if not new_selector:
            raise ValueError("LLM returned empty selector")

        if new_selector == original_selector:
            raise ValueError(
                f"LLM returned the same broken selector: {new_selector!r}. "
                "This selector is already confirmed not to work."
            )

    except Exception as exc:
        logger.warning("selector_fix_llm_failed", error=str(exc)[:200], task_id=task.id)
        return CorrectionResult(
            strategy_used=CorrectionStrategy.SELECTOR_FIX,
            success=False,
            reason=f"LLM could not suggest alternative selector: {exc}",
        )

    logger.info(
        "selector_fix_applied",
        original=original_selector,
        new=new_selector,
        reason=fix_reason[:100],
        task_id=task.id,
    )

    step.input["selector"] = new_selector
    step.status = StepStatus.PENDING
    step.error_message = None

    return CorrectionResult(
        strategy_used=CorrectionStrategy.SELECTOR_FIX,
        success=True,
        reason=f"Selector updated: {original_selector!r} -> {new_selector!r}",
    )


# ── REPLAN ────────────────────────────────────────────────────────────────────


async def strategy_replan(
    step: Step,
    action_result: ActionResult,
    verification: VerificationResult,
    browser,
    task,
    dag: ExecutionDAG,
) -> CorrectionResult:
    """
    Call the Planner to regenerate the plan from the current position.

    Best for: wrong_page, plan_error, or any failure where the current
    page state has diverged significantly from what the plan assumed.
    """
    from core.planner import Planner, PlannerError

    logger.info(
        "strategy_replan",
        failed_tool=step.tool,
        task_id=task.id,
    )

    page_text = ""
    if browser and browser.is_running:
        page_result = await browser.extract_page()
        if page_result.success:
            page_text = page_result.output.get("text", "")

    completed = [s for s in dag.steps if s.status == StepStatus.COMPLETED]

    planner = Planner()
    try:
        new_dag, tokens = await planner.replan(
            goal=task.goal,
            goal_type=str(task.goal_type),
            completed_steps=completed,
            failed_step=step,
            page_text=page_text,
        )
        task.tokens_used = getattr(task, "tokens_used", 0) + tokens
    except PlannerError as exc:
        logger.warning("replan_failed", error=str(exc)[:200], task_id=task.id)
        return CorrectionResult(
            strategy_used=CorrectionStrategy.REPLAN,
            success=False,
            reason=f"Replan failed: {exc}",
        )

    logger.info(
        "replan_success",
        new_steps=len(new_dag),
        task_id=task.id,
    )

    return CorrectionResult(
        strategy_used=CorrectionStrategy.REPLAN,
        success=True,
        new_dag=new_dag,
        reason=f"Replanned with {len(new_dag)} new steps",
    )


# ── ABORT ─────────────────────────────────────────────────────────────────────


async def strategy_abort(
    step: Step,
    action_result: ActionResult,
    verification: VerificationResult,
    reason: str,
    task,
) -> CorrectionResult:
    """
    Mark the task as unrecoverable and stop execution.

    Used when:
      - Auth wall / CAPTCHA detected (cannot recover without credentials)
      - Max retries exhausted
      - All other strategies have already been tried and failed
    """
    logger.warning(
        "strategy_abort",
        tool=step.tool,
        reason=reason[:200],
        task_id=task.id,
    )

    return CorrectionResult(
        strategy_used=CorrectionStrategy.ABORT,
        success=False,
        reason=reason,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_json(content: str) -> dict:
    """Parse JSON from LLM response, tolerant of markdown fences."""
    import re
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}