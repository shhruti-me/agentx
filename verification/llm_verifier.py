"""
verification/llm_verifier.py

LLM Verifier — semantic outcome verification using the LLM as a judge.

Most expensive verifier. Only called when DOM and text verifiers
return UNCERTAIN. Sends the expected outcome and actual page content
to the LLM and asks: "Was this step's goal achieved?"

Token cost: ~200-400 tokens per call.
Use sparingly — the chain of responsibility ensures it's a last resort.
"""

from __future__ import annotations

import json

from core.models import ActionResult, Step, VerificationResult, VerificationStatus
from browser.extractor import truncate_for_llm
from llm.factory import get_llm_client
from log.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a browser automation verifier. Your job is to determine whether a browser action achieved its intended goal.

You will be given:
- STEP_TOOL: the tool that was used
- STEP_EXPECTED: what success should look like
- ACTUAL_OUTPUT: what the action actually returned
- PAGE_TEXT: the current page content (may be truncated)

Respond with ONLY a JSON object in this exact format:
{"result": "pass" | "fail" | "uncertain", "confidence": 0.0-1.0, "reason": "one sentence explanation"}

Rules:
- "pass": the step clearly achieved its goal
- "fail": the step clearly did not achieve its goal
- "uncertain": you cannot determine from the available information
- confidence should reflect how certain you are (0.9+ = very sure, 0.5 = unsure)
- reason must be one concise sentence, no more"""


async def llm_verify(
    step: Step,
    actual: ActionResult,
    page_text: str,
) -> VerificationResult:
    """
    Ask the LLM whether this step achieved its goal.

    Parameters
    ----------
    step      : Executed step with expected description.
    actual    : ActionResult from the tool.
    page_text : Full readable text of the current page (will be truncated).

    Returns
    -------
    VerificationResult. Never raises — falls back to UNCERTAIN on any error.
    """
    client = get_llm_client()

    # Build a focused prompt — truncate page text to keep costs low
    output_preview = json.dumps(actual.output or {}, default=str)[:500]
    page_preview = truncate_for_llm(page_text, max_chars=3000)

    prompt = f"""STEP_TOOL: {step.tool}
STEP_EXPECTED: {step.expected}
ACTUAL_OUTPUT: {output_preview}
PAGE_TEXT:
{page_preview}

Did this step achieve its goal? Respond with the JSON object only."""

    try:
        response = await client.complete(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            max_tokens=150,
            temperature=0.0,  # deterministic for verification
        )
    except Exception as exc:
        logger.warning("llm_verify_call_failed", error=str(exc)[:200])
        return VerificationResult(
            result=VerificationStatus.UNCERTAIN,
            method="llm",
            confidence=0.0,
            reason=f"LLM verifier call failed: {exc}",
        )

    return _parse_llm_response(response.content)


def _parse_llm_response(content: str) -> VerificationResult:
    """
    Parse the LLM's JSON verdict into a VerificationResult.
    Tolerant of minor formatting deviations.
    """
    import re

    # Strip markdown fences if present
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"```$", "", content)
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("llm_verify_parse_failed", content=content[:200])
                return VerificationResult(
                    result=VerificationStatus.UNCERTAIN,
                    method="llm",
                    confidence=0.0,
                    reason="LLM verifier returned unparseable response",
                )
        else:
            return VerificationResult(
                result=VerificationStatus.UNCERTAIN,
                method="llm",
                confidence=0.0,
                reason="LLM verifier returned no JSON",
            )

    result_str = data.get("result", "uncertain").lower().strip()
    confidence = float(data.get("confidence", 0.5))
    reason = data.get("reason", "No reason provided")

    if result_str == "pass":
        status = VerificationStatus.PASS
    elif result_str == "fail":
        status = VerificationStatus.FAIL
    else:
        status = VerificationStatus.UNCERTAIN

    logger.debug(
        "llm_verify_result",
        result=result_str,
        confidence=confidence,
        reason=reason[:100],
    )

    return VerificationResult(
        result=status,
        method="llm",
        confidence=confidence,
        reason=reason,
    )