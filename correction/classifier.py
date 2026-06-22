"""
correction/classifier.py

Failure Classifier — maps error signals to FailureType.

Priority order matters — checks are applied top-to-bottom and
return on first match. Timeout must come before selector signals
because timeout errors contain "locator" and "waiting for".
Auth wall must require multiple signals to avoid false positives
from pages that simply have a "login" link in their navbar.
"""

from __future__ import annotations

from core.models import ActionResult, FailureType, VerificationResult
from log.logger import get_logger

logger = get_logger(__name__)


def classify(
    action_result: ActionResult,
    verification: VerificationResult,
    step_tool: str,
    page_text: str = "",
) -> FailureType:
    """
    Classify failure type from available signals.

    Parameters
    ----------
    action_result : ActionResult (may contain error string).
    verification  : VerificationResult (contains reason string).
    step_tool     : The tool that failed.
    page_text     : Current page text for context-based detection.

    Returns
    -------
    FailureType enum value.
    """
    error = (action_result.error or "").lower()
    reason = (verification.reason or "").lower()
    combined = error + " " + reason
    page_lower = page_text.lower()

    # ── 1. Timeout — check first, before any DOM/content signals ──────
    # Timeout errors often contain "locator", "waiting for", "not found"
    # which would otherwise trigger stale_selector incorrectly.
    if "timeout" in combined or "timed out" in combined or "time out" in combined:
        logger.debug("classifier_timeout")
        return FailureType.TIMEOUT

    # ── 2. Auth wall — requires MULTIPLE signals to avoid false positives
    # Many normal pages have "login" in their navbar (HN, Reddit, etc.)
    # Require at least 2 auth signals OR a strong single signal.
    strong_auth = ["please sign in", "please log in", "sign in to continue",
                   "log in to continue", "captcha", "verify you are human",
                   "unusual traffic", "403 forbidden", "access denied"]
    weak_auth = ["login", "sign in", "create account", "register", "password"]

    strong_hits = [s for s in strong_auth if s in page_lower]
    weak_hits = [s for s in weak_auth if s in page_lower]

    if strong_hits or len(weak_hits) >= 3:
        logger.debug("classifier_auth_wall", strong=strong_hits, weak=weak_hits)
        return FailureType.AUTH_WALL

    # ── 3. Stale selector ─────────────────────────────────────────────
    selector_signals = [
        "not found", "no element", "locator", "selector",
        "waiting for", "element not visible", "element is not attached",
        "strict mode violation",
    ]
    if any(s in combined for s in selector_signals):
        logger.debug("classifier_stale_selector")
        return FailureType.STALE_SELECTOR

    # ── 4. Empty extraction ───────────────────────────────────────────
    empty_signals = ["empty", "no content", "no text", "no results",
                     "0 chars", "too short"]
    if any(s in combined for s in empty_signals):
        logger.debug("classifier_extraction_empty")
        return FailureType.EXTRACTION_EMPTY

    # ── 5. Wrong page ─────────────────────────────────────────────────
    wrong_page_signals = ["wrong page", "unexpected page", "page not found",
                          "404", "navigated somewhere", "redirect"]
    if any(s in combined for s in wrong_page_signals):
        logger.debug("classifier_wrong_page")
        return FailureType.WRONG_PAGE

    # ── 6. Plan error ─────────────────────────────────────────────────
    plan_signals = ["unexpected input", "unknown tool", "bad input",
                    "invalid parameter", "makes no sense"]
    if any(s in combined for s in plan_signals):
        logger.debug("classifier_plan_error")
        return FailureType.PLAN_ERROR

    # ── 7. Default ────────────────────────────────────────────────────
    logger.debug("classifier_unknown", error=error[:100], reason=reason[:100])
    return FailureType.UNKNOWN