"""
tools/browser_tools.py

Handler functions for all browser-based tools.

Each handler:
  - Receives a BrowserController instance + kwargs from Step.input
  - Calls exactly one BrowserController method
  - Returns ActionResult directly

These are thin. No logic lives here — logic lives in browser/actions.py.
The only job of a handler is to unpack Step.input kwargs and call
the right controller method.

Handler signature (all handlers must match this):
    async def handle_*(browser: BrowserController, **kwargs) -> ActionResult
"""

from __future__ import annotations

from browser.controller import BrowserController
from core.models import ActionResult


async def handle_navigate(browser: BrowserController, url: str = "", **_) -> ActionResult:
    if not url:
        return ActionResult.fail("navigate requires 'url' parameter")
    return await browser.navigate(url)


async def handle_click(browser: BrowserController, selector: str = "", **_) -> ActionResult:
    if not selector:
        return ActionResult.fail("click requires 'selector' parameter")
    return await browser.click(selector)


async def handle_click_text(browser: BrowserController, text: str = "", **_) -> ActionResult:
    if not text:
        return ActionResult.fail("click_text requires 'text' parameter")
    return await browser.click_text(text)


async def handle_type(
    browser: BrowserController,
    selector: str = "",
    text: str = "",
    **_,
) -> ActionResult:
    if not selector:
        return ActionResult.fail("type requires 'selector' parameter")
    if not text:
        return ActionResult.fail("type requires 'text' parameter")
    return await browser.type(selector, text)


async def handle_scroll(
    browser: BrowserController,
    direction: str = "down",
    amount: int = 500,
    **_,
) -> ActionResult:
    return await browser.scroll(direction=direction, amount=int(amount))


async def handle_extract(browser: BrowserController, selector: str = "", **_) -> ActionResult:
    if not selector:
        return ActionResult.fail("extract requires 'selector' parameter")
    return await browser.extract(selector)


async def handle_extract_page(browser: BrowserController, **_) -> ActionResult:
    return await browser.extract_page()


async def handle_get_links(browser: BrowserController, **_) -> ActionResult:
    return await browser.get_links()


async def handle_dom_snapshot(browser: BrowserController, **_) -> ActionResult:
    return await browser.get_dom_snapshot()


async def handle_screenshot(browser: BrowserController, **_) -> ActionResult:
    return await browser.screenshot()