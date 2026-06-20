"""
verification/verifier.py

Verification Engine — checks whether a step's actual outcome matches expected.

Week 3 stub: passes everything that didn't error, fails hard errors.
Week 4: full three-method verification (DOM, text match, LLM judge).

The stub allows the full pipeline to run end-to-end in Week 3
so we can validate planning and execution before adding verification depth.
"""

from __future__ import annotations

from core.models import ActionResult, Step, VerificationResult, VerificationStatus
from log.logger import get_logger

logger = get_logger(__name__)


class Verifier:
    """
    Verifies that a step's actual outcome matches its expected outcome.

    Week 3 behaviour:
      - If action succeeded (result.success=True) → PASS
      - If action failed (result.success=False) → FAIL
      - No DOM/text/LLM checking yet

    Week 4 behaviour (full implementation):
      - DOM verifier: structural page changes
      - Text verifier: expected strings in page content
      - LLM verifier: semantic judgment for ambiguous cases
    """

    async def verify(self, step: Step, actual: ActionResult) -> VerificationResult:
        """
        Check actual outcome against step.expected.

        Parameters
        ----------
        step   : The step that was executed, including expected outcome.
        actual : What the tool actually returned.

        Returns
        -------
        VerificationResult with PASS, FAIL, or UNCERTAIN.
        """
        # Week 3 stub: trust the action's own success flag
        if actual.success:
            logger.debug(
                "verify_pass",
                tool=step.tool,
                method="action_success",
                step_index=step.step_index,
            )
            return VerificationResult(
                result=VerificationStatus.PASS,
                method="action_success",
                confidence=1.0,
                reason="Action completed without error",
            )
        else:
            logger.debug(
                "verify_fail",
                tool=step.tool,
                method="action_error",
                error=actual.error,
                step_index=step.step_index,
            )
            return VerificationResult(
                result=VerificationStatus.FAIL,
                method="action_error",
                confidence=1.0,
                reason=actual.error or "Action failed",
            )