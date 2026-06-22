"""
verification/text_verifier.py

Text Verifier — checks whether expected content appears in the page.

Zero LLM cost. Runs second in the verification chain, after DOM verifier
returns UNCERTAIN.

Strategy:
  1. Check if expected strings from step.expected appear in the output
  2. Check if the extracted text is substantive (non-empty, non-trivial)
  3. Check for failure signals (error pages, login walls, empty results)

Returns PASS, FAIL, or UNCERTAIN.
UNCERTAIN means "content is there but I can't confirm it's correct
— let the LLM verifier make the semantic call."
"""

from __future__ import annotations

import re

from core.models import ActionResult, Step, VerificationResult, VerificationStatus
from log.logger import get_logger

logger = get_logger(__name__)

# Phrases that indicate a failure page regardless of HTTP status
_FAILURE_SIGNALS = [
    "page not found",
    "404 not found",
    "403 forbidden",
    "access denied",
    "this page doesn't exist",
    "no results found",
    "sign in to continue",
    "please log in",
    "captcha",
    "unusual traffic",
    "verify you are human",
]

# Minimum meaningful content length
_MIN_CONTENT_LENGTH = 20


async def text_verify(
    step: Step,
    actual: ActionResult,
    page_text: str,
) -> VerificationResult:
    """
    Check page text against the step's expected outcome.

    Parameters
    ----------
    step      : Executed step with expected description.
    actual    : ActionResult from the tool.
    page_text : Full readable text of the current page.

    Returns
    -------
    VerificationResult with PASS, FAIL, or UNCERTAIN.
    """
    expected = step.expected.lower()
    tool = step.tool

    # Hard fail: action errored
    if not actual.success:
        return VerificationResult(
            result=VerificationStatus.FAIL,
            method="text",
            confidence=1.0,
            reason=actual.error or "Action failed",
        )

    # Check for failure signals in page text
    page_lower = page_text.lower()
    for signal in _FAILURE_SIGNALS:
        if signal in page_lower:
            return VerificationResult(
                result=VerificationStatus.FAIL,
                method="text",
                confidence=0.85,
                reason=f"Failure signal detected on page: '{signal}'",
            )

    # ── Extract tools: check output text directly ─────────────────────
    if tool in ("extract", "extract_page"):
        output_text = actual.output.get("text", "")

        if not output_text or len(output_text.strip()) < _MIN_CONTENT_LENGTH:
            return VerificationResult(
                result=VerificationStatus.FAIL,
                method="text",
                confidence=0.9,
                reason=f"Extracted text too short: {len(output_text)} chars",
            )

        # Check if extracted content matches keywords from expected description
        keywords = _extract_keywords(expected)
        matched = [kw for kw in keywords if kw in output_text.lower()]

        if keywords and len(matched) >= max(1, len(keywords) // 2):
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="text",
                confidence=0.85,
                reason=f"Output contains expected keywords: {matched}",
            )

        # Content present but keywords not matched — still likely a pass
        # for extraction tasks where we don't know the exact content
        if len(output_text.strip()) >= _MIN_CONTENT_LENGTH:
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="text",
                confidence=0.7,
                reason=f"Extracted {len(output_text)} chars of content",
            )

    # ── Navigate / click: check page has meaningful content ───────────
    if tool in ("navigate", "click", "click_text"):
        if len(page_text.strip()) >= _MIN_CONTENT_LENGTH:
            # Check if we landed on roughly the right kind of page
            keywords = _extract_keywords(expected)
            matched = [kw for kw in keywords if kw in page_lower]
            if matched:
                return VerificationResult(
                    result=VerificationStatus.PASS,
                    method="text",
                    confidence=0.8,
                    reason=f"Page content matches expected keywords: {matched}",
                )
            return VerificationResult(
                result=VerificationStatus.UNCERTAIN,
                method="text",
                confidence=0.5,
                reason="Page has content but expected keywords not found",
            )
        return VerificationResult(
            result=VerificationStatus.FAIL,
            method="text",
            confidence=0.8,
            reason="Page appears empty after action",
        )

    # ── Default: action succeeded with non-empty output ───────────────
    if actual.success and actual.output:
        return VerificationResult(
            result=VerificationStatus.UNCERTAIN,
            method="text",
            confidence=0.5,
            reason="Action succeeded — deferring semantic check to LLM verifier",
        )

    return VerificationResult(
        result=VerificationStatus.UNCERTAIN,
        method="text",
        confidence=0.3,
        reason="Cannot determine outcome from text alone",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_keywords(text: str) -> list[str]:
    """
    Pull meaningful words from an expected outcome description.
    Used to check if those words appear in the actual output.

    Strips stopwords, keeps nouns and domain-specific terms.
    """
    stopwords = {
        "the", "a", "an", "is", "was", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "from", "into", "that", "this", "these", "those",
        "page", "loaded", "found", "extracted", "returned", "completed",
        "successfully", "step", "data", "content", "text", "information",
        "result", "output", "contains", "shows", "displays",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in stopwords]