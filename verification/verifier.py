"""
verification/verifier.py

Verification Engine — orchestrates three verifiers in cost order.

Chain of Responsibility pattern:
  1. DOM Verifier   — structural checks, zero cost, runs first
  2. Text Verifier  — string matching, zero cost, runs if DOM = UNCERTAIN
  3. LLM Verifier   — semantic judgment, token cost, runs if Text = UNCERTAIN

Stops at the first PASS or FAIL. Only reaches LLM if truly ambiguous.

This design means the vast majority of steps are verified at zero
token cost. LLM verification is reserved for genuinely ambiguous cases
like "did the search return relevant results?" or "is this the login page?"
"""

from __future__ import annotations

from core.models import ActionResult, Step, VerificationResult, VerificationStatus
from verification.dom_verifier import dom_verify
from verification.text_verifier import text_verify
from verification.llm_verifier import llm_verify
from log.logger import get_logger

logger = get_logger(__name__)


class Verifier:
    """
    Orchestrates the three-method verification chain.

    Requires a live BrowserController to get current page state
    for DOM and text checks. Injected at instantiation.
    """

    def __init__(self, browser=None) -> None:
        # browser is optional — if None, DOM/text checks use ActionResult data only
        self._browser = browser

    async def verify(self, step: Step, actual: ActionResult) -> VerificationResult:
        """
        Run the verification chain for a completed step.

        Parameters
        ----------
        step   : The step that was executed.
        actual : What the tool returned.

        Returns
        -------
        VerificationResult. Never raises.
        """
        # Gather current page state for verification context
        current_url = ""
        current_title = ""
        dom_snapshot = ""
        page_text = ""

        if self._browser and self._browser.is_running:
            try:
                current_url = await self._browser.current_url()
                current_title = await self._browser.current_title()
                snap_result = await self._browser.get_dom_snapshot()
                dom_snapshot = snap_result.output.get("snapshot", "") if snap_result.success else ""
                page_result = await self._browser.extract_page()
                page_text = page_result.output.get("text", "") if page_result.success else ""
            except Exception as exc:
                logger.warning("verifier_page_state_error", error=str(exc)[:200])
        else:
            # No browser — use what's in the ActionResult
            page_text = actual.output.get("text", "") if actual.output else ""
            current_url = actual.output.get("url", "") if actual.output else ""
            current_title = actual.output.get("title", "") if actual.output else ""

        # ── Method 1: DOM Verifier ─────────────────────────────────────
        dom_result = await dom_verify(
            step=step,
            actual=actual,
            current_url=current_url,
            current_title=current_title,
            dom_snapshot=dom_snapshot,
        )

        logger.debug(
            "verify_dom",
            tool=step.tool,
            result=dom_result.result,
            confidence=dom_result.confidence,
            reason=dom_result.reason[:100],
        )

        if dom_result.result != VerificationStatus.UNCERTAIN:
            return dom_result

        # ── Method 2: Text Verifier ────────────────────────────────────
        text_result = await text_verify(
            step=step,
            actual=actual,
            page_text=page_text,
        )

        logger.debug(
            "verify_text",
            tool=step.tool,
            result=text_result.result,
            confidence=text_result.confidence,
            reason=text_result.reason[:100],
        )

        if text_result.result != VerificationStatus.UNCERTAIN:
            return text_result

        # ── Method 3: LLM Verifier (last resort) ──────────────────────
        logger.info(
            "verify_llm_called",
            tool=step.tool,
            step_index=step.step_index,
            reason="both dom and text verifiers returned uncertain",
        )

        llm_result = await llm_verify(
            step=step,
            actual=actual,
            page_text=page_text,
        )

        logger.debug(
            "verify_llm",
            tool=step.tool,
            result=llm_result.result,
            confidence=llm_result.confidence,
            reason=llm_result.reason[:100],
        )

        # UNCERTAIN from all three → treat as FAIL
        if llm_result.result == VerificationStatus.UNCERTAIN:
            return VerificationResult(
                result=VerificationStatus.FAIL,
                method="none",
                confidence=0.5,
                reason="All three verifiers returned UNCERTAIN — treating as FAIL",
            )

        return llm_result