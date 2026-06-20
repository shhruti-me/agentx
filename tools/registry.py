"""
tools/registry.py

Static tool registry for AGENTX.

Maps tool names to their handler functions and input schemas.
The Planner reads descriptions to know what tools are available.
The Execution Engine calls get_tool() to resolve a step's tool.

Adding a new tool:
  1. Implement the handler in browser_tools.py or utility_tools.py
  2. Add one entry to TOOL_REGISTRY below
  Zero changes needed anywhere else.

Tool handler signature:
    async def handler(browser: BrowserController, **kwargs) -> ActionResult

All kwargs come from Step.input — the dict the Planner generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from core.models import ActionResult


@dataclass
class ToolDefinition:
    """
    Metadata for one tool.

    name        : Unique identifier used in plan steps.
    description : Plain English — injected into the Planner prompt
                  so the LLM knows when to use this tool.
    input_schema: Dict describing expected parameters.
                  Injected into the prompt so the LLM generates
                  correct input dicts.
    handler     : Async callable. Receives browser + kwargs from Step.input.
    """
    name:         str
    description:  str
    input_schema: dict[str, Any]
    handler:      Callable[..., Awaitable[ActionResult]]


# Populated at module load by _build_registry() below.
# Import this dict directly — do not instantiate ToolRegistry.
TOOL_REGISTRY: dict[str, ToolDefinition] = {}


def get_tool(name: str) -> ToolDefinition | None:
    """
    Resolve a tool by name. Returns None if not found.
    The Execution Engine calls this for every step.
    """
    return TOOL_REGISTRY.get(name)


def get_tool_catalog() -> str:
    """
    Return a formatted string of all tools and their descriptions.
    Injected into the Planner prompt so the LLM knows what's available.

    Format:
        TOOL: navigate
        DESCRIPTION: Load a URL in the browser and wait for the page to load.
        INPUT: {"url": "string — the full URL to navigate to"}

        TOOL: click
        ...
    """
    lines = []
    for tool in TOOL_REGISTRY.values():
        lines.append(f"TOOL: {tool.name}")
        lines.append(f"DESCRIPTION: {tool.description}")
        lines.append(f"INPUT: {tool.input_schema}")
        lines.append("")
    return "\n".join(lines)


def _build_registry() -> None:
    """
    Populate TOOL_REGISTRY with all available tools.
    Called once at module load. Import order matters:
    browser_tools imports registry, so we import lazily here.
    """
    from tools.browser_tools import (
        handle_navigate,
        handle_click,
        handle_click_text,
        handle_type,
        handle_scroll,
        handle_extract,
        handle_extract_page,
        handle_get_links,
        handle_dom_snapshot,
        handle_screenshot,
    )

    tools = [
        ToolDefinition(
            name="navigate",
            description=(
                "Load a URL in the browser and wait for the page to fully load. "
                "Use this as the first step for any task. "
                "Returns the page title and final URL after any redirects."
            ),
            input_schema={"url": "string — the full URL including https://"},
            handler=handle_navigate,
        ),
        ToolDefinition(
            name="click",
            description=(
                "Click an element on the page using a CSS selector. "
                "Use for buttons, links, tabs, and any clickable element "
                "when you know the CSS selector. "
                "Scrolls the element into view before clicking."
            ),
            input_schema={"selector": "string — CSS selector for the element to click"},
            handler=handle_click,
        ),
        ToolDefinition(
            name="click_text",
            description=(
                "Click the first element whose visible text matches the given string. "
                "Use this instead of 'click' when you don't know the CSS selector "
                "but you know what the button or link says. "
                "More resilient than CSS selectors on dynamic pages."
            ),
            input_schema={"text": "string — the exact visible text of the element to click"},
            handler=handle_click_text,
        ),
        ToolDefinition(
            name="type",
            description=(
                "Clear a text input field and type the given text into it. "
                "Use for search boxes, form fields, login inputs. "
                "Simulates human typing with realistic delays."
            ),
            input_schema={
                "selector": "string — CSS selector for the input field",
                "text": "string — the text to type",
            },
            handler=handle_type,
        ),
        ToolDefinition(
            name="scroll",
            description=(
                "Scroll the page up or down by a given number of pixels. "
                "Use when content is below the fold or to reveal more items. "
                "Default amount is 500px (roughly one screen height)."
            ),
            input_schema={
                "direction": "string — 'up' or 'down'",
                "amount": "integer — pixels to scroll, default 500",
            },
            handler=handle_scroll,
        ),
        ToolDefinition(
            name="extract",
            description=(
                "Extract text from specific elements matching a CSS selector. "
                "Use when you know exactly which element contains the data you need. "
                "Returns text from all matching elements joined with newlines."
            ),
            input_schema={"selector": "string — CSS selector for elements to extract text from"},
            handler=handle_extract,
        ),
        ToolDefinition(
            name="extract_page",
            description=(
                "Extract all readable text from the entire current page. "
                "Use when you need to read the full page content to find information, "
                "or when you don't know which specific element contains the answer. "
                "Returns clean text with all HTML removed."
            ),
            input_schema={},
            handler=handle_extract_page,
        ),
        ToolDefinition(
            name="get_links",
            description=(
                "Get all clickable links on the current page. "
                "Use when you need to find a navigation path, "
                "discover available pages, or find a specific link to follow."
            ),
            input_schema={},
            handler=handle_get_links,
        ),
        ToolDefinition(
            name="dom_snapshot",
            description=(
                "Get a compact structural outline of the current page's DOM. "
                "Use when you need to understand the page layout to find selectors, "
                "or when previous selectors have failed and you need to find alternatives."
            ),
            input_schema={},
            handler=handle_dom_snapshot,
        ),
        ToolDefinition(
            name="screenshot",
            description=(
                "Capture a screenshot of the current page viewport as PNG bytes. "
                "Use for visual debugging or when page structure is unclear."
            ),
            input_schema={},
            handler=handle_screenshot,
        ),
    ]

    for tool in tools:
        TOOL_REGISTRY[tool.name] = tool


# Build registry at import time
_build_registry()