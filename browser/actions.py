"""
browser/actions.py

Individual browser actions as standalone async functions.

Each function takes a Playwright Page object and returns an ActionResult.
No action imports from another action — they are fully independent.

The BrowserController (controller.py) is the only caller.
Nothing outside browser/ ever calls these directly.

Why standalone functions instead of methods:
  - Each action is independently testable with a mock Page
  - Adding a new action = adding one function, no class changes
  - Follows the Command pattern: discrete, named, logged operations

Human-like behaviour:
  - Small random delays between keystrokes (type action)
  - Scroll into view before clicking
  - Wait for network idle after navigation
  These reduce bot-detection failures on real sites.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from core.models import ActionResult
from browser.extractor import extract_page_text, extract_dom_snapshot, extract_links, clean_text

if TYPE_CHECKING:
    from playwright.async_api import Page

from log.logger import get_logger

logger = get_logger(__name__)


# ── navigate ──────────────────────────────────────────────────────────────────


async def navigate(page: "Page", url: str, timeout_ms: int = 30_000) -> ActionResult:
    """
    Load a URL and wait for the page to settle.

    Waits for networkidle — no pending network requests for 500ms.
    Falls back to domcontentloaded if networkidle times out, since
    some pages (SPAs with polling) never reach true networkidle.

    Returns page title and final URL (after any redirects).
    """
    start = time.monotonic()
    try:
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    except Exception:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as exc:
            return ActionResult.fail(
                error=f"Navigation failed: {exc}",
                duration_ms=_elapsed(start),
            )

    try:
        title = await page.title()
        final_url = page.url
    except Exception as exc:
        return ActionResult.fail(
            error=f"Could not read page title after navigation: {exc}",
            duration_ms=_elapsed(start),
        )

    logger.info("action_navigate", url=url, final_url=final_url, title=title[:80])

    return ActionResult.ok(
        output={"title": title, "url": final_url},
        duration_ms=_elapsed(start),
    )


# ── click ─────────────────────────────────────────────────────────────────────


async def click(
    page: "Page",
    selector: str,
    timeout_ms: int = 10_000,
) -> ActionResult:
    """
    Find an element and click it.

    Strategy:
      1. Wait for the element to be visible (not just present in DOM)
      2. Scroll it into view
      3. Click

    If the CSS selector fails, falls back to text-based matching.
    This handles the common case where a selector was valid when
    the plan was generated but the DOM changed slightly.
    """
    start = time.monotonic()

    try:
        element = await page.wait_for_selector(
            selector, state="visible", timeout=timeout_ms
        )
        if element is None:
            raise ValueError(f"Selector returned None: {selector}")

        await element.scroll_into_view_if_needed()
        await element.click()
        await page.wait_for_load_state("domcontentloaded", timeout=5_000)

        logger.info("action_click", selector=selector)
        return ActionResult.ok(
            output={"clicked": selector, "url": page.url},
            duration_ms=_elapsed(start),
        )

    except Exception as exc:
        error = str(exc)
        logger.warning("action_click_failed", selector=selector, error=error[:200])
        return ActionResult.fail(error=f"Click failed on {selector!r}: {error}", duration_ms=_elapsed(start))


async def click_text(page: "Page", text: str, timeout_ms: int = 10_000) -> ActionResult:
    """
    Click the first element whose visible text matches the given string.

    Used as a fallback when CSS selectors fail. Text matching is more
    resilient to DOM structure changes.
    """
    start = time.monotonic()
    selector = f"text={text}"
    try:
        await page.click(selector, timeout=timeout_ms)
        logger.info("action_click_text", text=text)
        return ActionResult.ok(
            output={"clicked_text": text, "url": page.url},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"Click by text failed for {text!r}: {exc}",
            duration_ms=_elapsed(start),
        )


# ── type ──────────────────────────────────────────────────────────────────────


async def type_text(
    page: "Page",
    selector: str,
    text: str,
    delay_ms: int = 50,
    timeout_ms: int = 10_000,
) -> ActionResult:
    """
    Clear a field and type text with human-like keystroke delays.

    Clears the field first (triple-click selects all, then type replaces).
    delay_ms adds a small pause between keystrokes — this reduces
    bot-detection failures on sites that check typing speed.
    """
    start = time.monotonic()
    try:
        element = await page.wait_for_selector(
            selector, state="visible", timeout=timeout_ms
        )
        if element is None:
            raise ValueError(f"Selector returned None: {selector}")

        await element.scroll_into_view_if_needed()
        await element.triple_click()
        await element.type(text, delay=delay_ms)

        logger.info("action_type", selector=selector, text_length=len(text))
        return ActionResult.ok(
            output={"typed": text, "selector": selector},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"Type failed on {selector!r}: {exc}",
            duration_ms=_elapsed(start),
        )


# ── scroll ────────────────────────────────────────────────────────────────────


async def scroll(
    page: "Page",
    direction: str = "down",
    amount: int = 500,
) -> ActionResult:
    """
    Scroll the viewport by `amount` pixels.

    direction : "up" or "down"
    amount    : Pixels to scroll. 500 = roughly one viewport height.
    """
    start = time.monotonic()
    y = amount if direction == "down" else -amount
    try:
        await page.evaluate(f"window.scrollBy(0, {y})")
        await asyncio.sleep(0.3)

        scroll_y = await page.evaluate("window.scrollY")
        logger.info("action_scroll", direction=direction, amount=amount, scroll_y=scroll_y)
        return ActionResult.ok(
            output={"direction": direction, "amount": amount, "scroll_y": scroll_y},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"Scroll failed: {exc}",
            duration_ms=_elapsed(start),
        )


# ── extract ───────────────────────────────────────────────────────────────────


async def extract(
    page: "Page",
    selector: str,
    timeout_ms: int = 10_000,
) -> ActionResult:
    """
    Extract text content from elements matching the selector.

    Returns text from all matching elements joined with newlines.
    Used for targeted extraction when the Planner knows the exact selector.
    """
    start = time.monotonic()
    # Use a shorter timeout for selector existence check — if it's not found
    # quickly, it likely doesn't exist. Correction engine handles the retry.
    selector_timeout = min(timeout_ms, 10_000)
    try:
        await page.wait_for_selector(selector, state="attached", timeout=selector_timeout)
        elements = await page.query_selector_all(selector)

        if not elements:
            return ActionResult.fail(
                error=f"No elements matched selector: {selector!r}",
                duration_ms=_elapsed(start),
            )

        texts = []
        for el in elements:
            text = await el.inner_text()
            if text.strip():
                texts.append(text.strip())

        combined = "\n".join(texts)
        logger.info("action_extract", selector=selector, elements=len(elements), chars=len(combined))

        return ActionResult.ok(
            output={"text": combined, "count": len(texts), "selector": selector},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"Extract failed on {selector!r}: {exc}",
            duration_ms=_elapsed(start),
        )


async def extract_page(page: "Page") -> ActionResult:
    """
    Extract all readable text from the current page.

    Uses Playwright's inner_text() on the body element — reads the
    live rendered DOM, not raw HTML source. This is the correct
    approach for JS-rendered pages (Wikipedia, React SPAs, etc.)
    where page.content() returns pre-JS source HTML.

    Falls back to page.content() + HTML parser if body not found.
    """
    start = time.monotonic()
    try:
        url = page.url
        title = await page.title()

        body = await page.query_selector("body")
        if body:
            raw = await body.inner_text()
            text = clean_text(raw)
        else:
            html = await page.content()
            text = extract_page_text(html)

        logger.info("action_extract_page", url=url, chars=len(text))
        return ActionResult.ok(
            output={"text": text, "title": title, "url": url, "char_count": len(text)},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"extract_page failed: {exc}",
            duration_ms=_elapsed(start),
        )


# ── screenshot ────────────────────────────────────────────────────────────────


async def screenshot(page: "Page") -> ActionResult:
    """
    Capture the current viewport as a PNG.
    Returns raw bytes in output["bytes"].
    """
    start = time.monotonic()
    try:
        png_bytes = await page.screenshot(type="png", full_page=False)
        logger.info("action_screenshot", bytes=len(png_bytes))
        return ActionResult.ok(
            output={"bytes": png_bytes, "size": len(png_bytes)},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"Screenshot failed: {exc}",
            duration_ms=_elapsed(start),
        )


# ── get_dom_snapshot ──────────────────────────────────────────────────────────


async def get_dom_snapshot(page: "Page") -> ActionResult:
    """
    Return a compact DOM outline of the current page for LLM consumption.

    Uses JavaScript to walk the live DOM directly.
    Depth 8, MAX 200 elements, includes tbody for table-heavy sites (HN).
    Used by selector_fix to find alternative selectors.
    """
    start = time.monotonic()
    try:
        # Build JS as a raw string to avoid Python escape conflicts
        js = (
            "() => {"
            "  const INC = new Set(['body','main','article','section','nav','header','footer',"
            "    'h1','h2','h3','h4','h5','h6','p','a','button','input','textarea','select','form',"
            "    'ul','ol','li','table','tbody','tr','td','th','div','span']);"
            "  const SKIP = new Set(['script','style','noscript','svg','path','head']);"
            "  const lines = [];"
            "  const MAX = 200;"
            "  function walk(el, depth) {"
            "    if (lines.length >= MAX || depth > 8) return;"
            "    const tag = el.tagName ? el.tagName.toLowerCase() : '';"
            "    if (!tag || SKIP.has(tag)) return;"
            "    if (INC.has(tag)) {"
            "      let label = tag;"
            "      if (el.id) label += '#' + el.id;"
            "      const cls = Array.from(el.classList).slice(0,3).join('.');"
            "      if (cls) label += '.' + cls;"
            "      if (tag === 'a') {"
            "        const href = el.getAttribute('href') || '';"
            "        if (href) label += ' href=\"' + href.slice(0,80) + '\"';"
            "      }"
            "      if (tag === 'input' && el.type) label += ' type=\"' + el.type + '\"';"
            "      const dt = Array.from(el.childNodes)"
            "        .filter(function(n){return n.nodeType===3;})"
            "        .map(function(n){return n.textContent.trim();})"
            "        .join(' ').trim().slice(0,60);"
            "      if (dt) label += ' \"' + dt + '\"';"
            "      lines.push('  '.repeat(depth) + label);"
            "    }"
            "    for (let i=0;i<el.children.length;i++) walk(el.children[i], depth+1);"
            "  }"
            "  walk(document.body, 0);"
            "  if (lines.length >= MAX) lines.push('[...more elements]');"
            "  return lines.join('\\n');"
            "}"
        )

        snapshot = await page.evaluate(js)

        if not snapshot:
            snapshot = ""

        line_count = len(snapshot.splitlines()) if snapshot else 0
        logger.info("action_dom_snapshot", chars=len(snapshot), lines=line_count)
        return ActionResult.ok(
            output={"snapshot": snapshot},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"DOM snapshot failed: {exc}",
            duration_ms=_elapsed(start),
        )


async def get_links(page: "Page") -> ActionResult:
    """
    Extract all links from the current page via the live DOM.
    Returns list of {text, href, absolute_url} dicts.
    """
    start = time.monotonic()
    try:
        links = await page.evaluate("""
        () => {
            const anchors = Array.from(document.querySelectorAll('a[href]'));
            return anchors
                .filter(a => {
                    const h = a.getAttribute('href') || '';
                    return h && !h.startsWith('#') && !h.startsWith('javascript:') && !h.startsWith('mailto:');
                })
                .slice(0, 500)
                .map(a => ({
                    text: (a.innerText || '').trim().slice(0, 80),
                    href: a.getAttribute('href'),
                    absolute_url: a.href
                }));
        }
        """)

        logger.info("action_get_links", count=len(links))
        return ActionResult.ok(
            output={"links": links, "count": len(links)},
            duration_ms=_elapsed(start),
        )
    except Exception as exc:
        return ActionResult.fail(
            error=f"get_links failed: {exc}",
            duration_ms=_elapsed(start),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _elapsed(start: float) -> int:
    return int((time.monotonic() - start) * 1000)