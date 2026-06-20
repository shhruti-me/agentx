"""
core/planner.py

Planner — converts a natural language goal into an ExecutionDAG.

The Planner is where AI engineering depth lives:
  - Retrieval-augmented prompting: injects past successes before planning
  - Structured output parsing: extracts a valid step list from LLM response
  - Replan: regenerates the DAG from a failure point with updated context

Flow:
  1. Query memory for similar tasks and past successful actions
  2. Build a prompt: system instructions + tool catalog + memory + goal
  3. Call the LLM
  4. Parse the JSON step list from the response
  5. Construct and return an ExecutionDAG

The LLM is asked to respond with a JSON array of steps only.
No prose, no explanation — just the plan. This makes parsing reliable.
"""

from __future__ import annotations

import json
import re

from core.models import ExecutionDAG, GoalType, Step
from llm.factory import get_llm_client
from llm.base import LLMResponse
from memory.retrieval import PlanningContext, get_planning_context
from tools.registry import get_tool_catalog
from log.logger import get_logger

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a browser automation planner. Your job is to convert a natural language goal into a precise, minimal sequence of browser actions.

You will be given:
- A GOAL to accomplish
- A list of TOOLS you can use
- Optional MEMORY of past successful actions

You must respond with ONLY a JSON array of steps. No explanation, no markdown, no prose. Just the JSON array.

Each step must have exactly these fields:
{
  "tool": "<tool name from the available tools>",
  "input": {<tool-specific parameters>},
  "expected": "<one sentence describing what success looks like for this step>"
}

Rules:
1. Always start with a navigate step unless the browser is already on the right page.
2. Use the minimum number of steps to achieve the goal.
3. Use extract_page when you need to read content but don't know the exact selector.
4. Use extract with a specific selector when you know exactly where the data is.
5. Use click_text instead of click when you don't know the CSS selector.
6. The last step must extract or return the information requested in the goal.
7. Do not include steps that verify success — the system handles that separately.
8. Maximum 15 steps. If a goal requires more, break it into the core path only.

Example response for goal "Go to wikipedia and find when Python was created":
[
  {"tool": "navigate", "input": {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}, "expected": "Wikipedia Python page is loaded"},
  {"tool": "extract_page", "input": {}, "expected": "Page text is extracted containing release year information"},
  {"tool": "extract", "input": {"selector": ".infobox td"}, "expected": "Infobox data containing the release year 1991 is extracted"}
]"""


# ── Planner ───────────────────────────────────────────────────────────────────


class Planner:
    """
    Converts a natural language goal into an ExecutionDAG.

    One instance per Orchestrator. Stateless between calls —
    all context comes from memory queries and the current goal.
    """

    def __init__(self) -> None:
        self._client = get_llm_client()

    async def plan(self, goal: str, goal_type: str = "unknown") -> ExecutionDAG:
        """
        Generate an ExecutionDAG for the given goal.

        Queries memory for relevant past context, builds a prompt,
        calls the LLM, parses the response into Steps.

        Parameters
        ----------
        goal      : Natural language goal string.
        goal_type : Category hint — used to scope memory queries.

        Returns
        -------
        ExecutionDAG with steps ready for the Execution Engine.

        Raises
        ------
        PlannerError : LLM call failed or response could not be parsed
                       into a valid step list.
        """
        logger.info("planner_start", goal=goal[:80], goal_type=goal_type)

        # Retrieve memory context
        ctx = get_planning_context(goal=goal, goal_type=goal_type)

        # Build prompt
        prompt = _build_prompt(goal=goal, memory_ctx=ctx)

        # Call LLM
        try:
            response = await self._client.complete(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
            )
        except Exception as exc:
            raise PlannerError(f"LLM call failed during planning: {exc}") from exc

        logger.info(
            "planner_llm_response",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )

        # Parse response into steps
        steps = _parse_steps(response.content)
        if not steps:
            raise PlannerError(
                f"LLM returned no parseable steps.\nRaw response:\n{response.content[:500]}"
            )

        dag = ExecutionDAG(steps=steps)

        logger.info("planner_done", steps=len(steps), goal=goal[:80])
        return dag, response.total_tokens

    async def replan(
        self,
        goal: str,
        goal_type: str,
        completed_steps: list[Step],
        failed_step: Step,
        page_text: str,
    ) -> ExecutionDAG:
        """
        Regenerate the plan from the current position after a failure.

        Called by the Self-Correction Engine when strategy=REPLAN.
        Gives the LLM full context: what was done, what failed, what
        the page currently shows.

        Parameters
        ----------
        goal            : Original goal — unchanged.
        goal_type       : Original goal type.
        completed_steps : Steps that already succeeded.
        failed_step     : The step that failed.
        page_text       : Current page text (from extract_page).

        Returns
        -------
        New ExecutionDAG containing only the remaining steps.
        """
        logger.info("planner_replan", failed_tool=failed_step.tool, goal=goal[:80])

        completed_summary = "\n".join(
            f"  - {s.tool}({s.input}): SUCCESS" for s in completed_steps
        )

        prompt = f"""Original goal: {goal}

Steps already completed successfully:
{completed_summary or '  (none)'}

Step that FAILED:
  Tool: {failed_step.tool}
  Input: {failed_step.input}
  Error: {failed_step.error_message}
  Expected: {failed_step.expected}

Current page content (first 3000 chars):
{page_text[:3000]}

The original plan has failed at the step above.
Generate a NEW sequence of steps to complete the original goal from this point forward.
Do NOT repeat the steps that already succeeded.
Account for what you can see in the current page content.
Respond with ONLY a JSON array of remaining steps."""

        try:
            response = await self._client.complete(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
            )
        except Exception as exc:
            raise PlannerError(f"LLM call failed during replan: {exc}") from exc

        steps = _parse_steps(response.content)
        if not steps:
            raise PlannerError(
                f"Replan produced no parseable steps.\nRaw:\n{response.content[:500]}"
            )

        logger.info("planner_replan_done", new_steps=len(steps))
        return ExecutionDAG(steps=steps), response.total_tokens


# ── Prompt builder ────────────────────────────────────────────────────────────


def _build_prompt(goal: str, memory_ctx: PlanningContext) -> str:
    """
    Assemble the user-turn prompt from goal + tool catalog + memory.

    Structured in order of importance:
      1. Goal (most important — LLM reads top-down)
      2. Available tools
      3. Memory context (if any)
    """
    parts: list[str] = []

    parts.append(f"GOAL: {goal}")
    parts.append("")

    parts.append("AVAILABLE TOOLS:")
    parts.append(get_tool_catalog())

    if memory_ctx.has_prior_context:
        parts.append("MEMORY — past successful tasks similar to this goal:")
        for task in memory_ctx.similar_tasks[:3]:
            parts.append(f"  Goal: {task['goal']}")
            parts.append(f"  Result: {task['result']}")
            parts.append("")

        if memory_ctx.recent_successes:
            parts.append("MEMORY — recent successful actions for this goal type:")
            for action in memory_ctx.recent_successes[:3]:
                parts.append(f"  Tool: {action['tool']}, Context: {action['context']}")
            parts.append("")

    parts.append("Now generate the step-by-step plan as a JSON array:")

    return "\n".join(parts)


# ── Response parser ───────────────────────────────────────────────────────────


def _parse_steps(content: str) -> list[Step]:
    """
    Extract a JSON step array from the LLM response.

    LLMs sometimes wrap JSON in markdown code fences or add
    explanatory text before/after. This parser is forgiving:
    it finds the first [ ... ] block in the response and parses that.

    Returns empty list if no valid JSON array of steps found.
    """
    # Try direct parse first (ideal case — LLM returned pure JSON)
    content = content.strip()
    try:
        data = json.loads(content)
        return _validate_and_build(data)
    except json.JSONDecodeError:
        pass

    # Find JSON array in the response (handles markdown fences, prose wrapping)
    match = re.search(r"\[[\s\S]*?\]", content)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_and_build(data)
        except json.JSONDecodeError:
            pass

    # Last resort: find the largest [...] block
    matches = re.findall(r"\[[\s\S]*\]", content)
    for candidate in sorted(matches, key=len, reverse=True):
        try:
            data = json.loads(candidate)
            result = _validate_and_build(data)
            if result:
                return result
        except json.JSONDecodeError:
            continue

    logger.warning("planner_parse_failed", content_preview=content[:300])
    return []


def _validate_and_build(data: object) -> list[Step]:
    """
    Validate parsed JSON and construct Step objects.

    Tolerant of minor LLM deviations:
      - Missing 'expected' field → uses default string
      - Extra fields → ignored
      - Non-dict items → skipped with warning
    """
    if not isinstance(data, list):
        return []

    steps: list[Step] = []
    from tools.registry import get_tool

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("planner_skip_item", index=i, reason="not a dict")
            continue

        tool_name = item.get("tool", "").strip()
        if not tool_name:
            logger.warning("planner_skip_item", index=i, reason="missing tool name")
            continue

        if get_tool(tool_name) is None:
            logger.warning("planner_unknown_tool", tool=tool_name, index=i)
            # Don't skip — let the Execution Engine handle unknown tools
            # so the error is visible in the task record

        step = Step(
            tool=tool_name,
            input=item.get("input", {}),
            expected=item.get("expected", f"Step {i + 1} completed successfully"),
        )
        steps.append(step)

    return steps


# ── Exceptions ────────────────────────────────────────────────────────────────


class PlannerError(Exception):
    """Raised when the Planner cannot produce a valid ExecutionDAG."""