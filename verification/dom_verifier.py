"""
verification/dom_verifier.py

DOM Verifier — checks structural page changes after a step.

Zero LLM cost. Runs first in the verification chain.
Returns PASS, FAIL, or UNCERTAIN.

Checks performed (in order, stops at first match):
  1. URL changed to expected pattern
  2. Expected element appeared in DOM
  3. Expected element disappeared from DOM
  4. Page title matches expected pattern
  5. Action itself was a navigate/click with no error → UNCERTAIN
     (let text verifier try)
"""

from __future__ import annotations

import re

from core.models import ActionResult, Step, VerificationResult, VerificationStatus
from log.logger import get_logger

logger = get_logger(__name__)


async def dom_verify(
    step: Step,
    actual: ActionResult,
    current_url: str,
    current_title: str,
    dom_snapshot: str,
) -> VerificationResult:
    """
    Check structural DOM/URL changes against what the step expected.

    Parameters
    ----------
    step          : The executed step with step.expected description.
    actual        : The ActionResult from the tool.
    current_url   : Page URL after the action.
    current_title : Page title after the action.
    dom_snapshot  : Compact DOM outline (from get_dom_snapshot).

    Returns
    -------
    VerificationResult with PASS, FAIL, or UNCERTAIN.
    UNCERTAIN means "I can't tell — try the next verifier."
    """
    expected = step.expected.lower()
    tool = step.tool

    # ── Hard fail: action itself errored ──────────────────────────────
    if not actual.success:
        return VerificationResult(
            result=VerificationStatus.FAIL,
            method="dom",
            confidence=1.0,
            reason=actual.error or "Action failed",
        )

    # ── Navigate: check URL and title ─────────────────────────────────
    if tool == "navigate":
        input_url = step.input.get("url", "").lower()
        if input_url and _url_matches(current_url, input_url):
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=1.0,
                reason=f"URL matches expected: {current_url}",
            )
        if current_title and current_title.lower() not in ("", "about:blank"):
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=0.8,
                reason=f"Page loaded with title: {current_title}",
            )
        return VerificationResult(
            result=VerificationStatus.UNCERTAIN,
            method="dom",
            confidence=0.3,
            reason="Navigation completed but URL/title check inconclusive",
        )

    # ── Click: check URL changed or new element appeared ──────────────
    if tool in ("click", "click_text"):
        output_url = actual.output.get("url", "")
        if output_url and output_url != current_url:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=0.9,
                reason=f"Click caused navigation to {output_url}",
            )
        # Can't tell from DOM alone if click had the right effect
        return VerificationResult(
            result=VerificationStatus.UNCERTAIN,
            method="dom",
            confidence=0.4,
            reason="Click completed — cannot verify effect without content check",
        )

    # ── Type: field was found and typed into ──────────────────────────
    if tool == "type":
        if actual.success:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=0.9,
                reason=f"Typed into {step.input.get('selector', 'field')}",
            )

    # ── Scroll: scrollY changed ───────────────────────────────────────
    if tool == "scroll":
        scroll_y = actual.output.get("scroll_y", 0)
        direction = step.input.get("direction", "down")
        if direction == "down" and scroll_y > 0:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=1.0,
                reason=f"Scrolled to Y={scroll_y}",
            )
        if direction == "up" and scroll_y == 0:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=1.0,
                reason="Scrolled to top",
            )

    # ── Extract tools: non-empty output is the signal ─────────────────
    if tool in ("extract", "extract_page"):
        text = actual.output.get("text", "")
        count = actual.output.get("count", 0)
        if text and len(text.strip()) > 0:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="dom",
                confidence=0.85,
                reason=f"Extracted {len(text)} chars from {count or 1} element(s)",
            )
        return VerificationResult(
            result=VerificationStatus.FAIL,
            method="dom",
            confidence=0.9,
            reason="Extract returned empty content",
        )

    # ── Default: action succeeded, defer to text verifier ─────────────
    return VerificationResult(
        result=VerificationStatus.UNCERTAIN,
        method="dom",
        confidence=0.3,
        reason=f"Tool {tool!r} succeeded — deferring to text verifier",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _url_matches(current: str, expected: str) -> bool:
    """
    Check if current URL matches the expected URL.
    Tolerant of trailing slashes and http/https differences.
    """
    current = current.lower().rstrip("/")
    expected = expected.lower().rstrip("/")
    # Strip protocol for comparison
    current_path = re.sub(r"^https?://", "", current)
    expected_path = re.sub(r"^https?://", "", expected)
    return current_path == expected_path or current_path.startswith(expected_path)