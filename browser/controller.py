"""
browser/controller.py

BrowserController — Playwright session lifecycle and action dispatch.

This is the Facade the rest of AGENTX talks to.
Nothing outside browser/ ever imports Playwright directly.

One controller instance per task. Created by the Orchestrator,
passed to the Execution Engine, closed when the task completes.

Usage
-----
    async with BrowserController() as browser:
        result = await browser.navigate("https://news.ycombinator.com")
        result = await browser.extract(".titleline > a")
        print(result.output["text"])

All methods return ActionResult. Callers check result.success.
No method raises — failures are captured in ActionResult.fail().
"""

from __future__ import annotations

import time
from typing import Any

from core.models import ActionResult
from config.settings import settings
from log.logger import get_logger
from browser import actions

logger = get_logger(__name__)


class BrowserController:
    """
    Manages a single Playwright browser session for one task.

    Lifecycle
    ---------
    1. Instantiate:  controller = BrowserController()
    2. Start:        await controller.start()      ← opens browser
    3. Use:          await controller.navigate(url)
    4. Stop:         await controller.stop()       ← closes browser

    Or use as async context manager (recommended):
        async with BrowserController() as browser:
            await browser.navigate(url)

    The browser is Chromium headless by default.
    Set BROWSER_HEADLESS=false in .env to watch it run.
    """

    def __init__(
        self,
        headless: bool | None = None,
        slow_mo: int | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self._headless = headless if headless is not None else settings.browser_headless
        self._slow_mo = slow_mo if slow_mo is not None else settings.browser_slow_mo_ms
        self._timeout_ms = timeout_ms if timeout_ms is not None else settings.browser_timeout_ms

        # Set in start()
        self._playwright = None
        self._browser = None
        self._page = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Launch Playwright and open a Chromium browser.
        Called once per task. Do not call twice without calling stop() first.
        """
        from playwright.async_api import async_playwright

        logger.info(
            "browser_starting",
            headless=self._headless,
            slow_mo=self._slow_mo,
        )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",  # reduce bot detection
            ],
        )
        self._page = await self._browser.new_page(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page.set_default_timeout(self._timeout_ms)

        logger.info("browser_ready")

    async def stop(self) -> None:
        """
        Close the browser and clean up Playwright resources.
        Safe to call even if start() was never called.
        """
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:
            logger.warning("browser_stop_error", error=str(exc))
        finally:
            self._playwright = None
            self._browser = None
            self._page = None
            logger.info("browser_stopped")

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── Actions ────────────────────────────────────────────────────────────────
    # Each method is a thin delegation to browser/actions.py.
    # All complexity (waits, retries, human delays) lives in actions.py.
    # This class owns only session state and the public interface.

    def _require_page(self) -> None:
        """Raise if the browser hasn't been started."""
        if self._page is None:
            raise RuntimeError(
                "BrowserController.start() must be called before any action. "
                "Use 'async with BrowserController() as browser:' to handle this automatically."
            )

    async def navigate(self, url: str) -> ActionResult:
        """Load a URL. Waits for network idle. Returns title and final URL."""
        self._require_page()
        return await actions.navigate(self._page, url, timeout_ms=self._timeout_ms)

    async def click(self, selector: str) -> ActionResult:
        """Click an element by CSS selector. Scrolls into view first."""
        self._require_page()
        return await actions.click(self._page, selector, timeout_ms=self._timeout_ms)

    async def click_text(self, text: str) -> ActionResult:
        """Click the first element whose visible text matches. Fallback for stale selectors."""
        self._require_page()
        return await actions.click_text(self._page, text, timeout_ms=self._timeout_ms)

    async def type(self, selector: str, text: str) -> ActionResult:
        """Clear a field and type text with human-like delays."""
        self._require_page()
        return await actions.type_text(self._page, selector, text, timeout_ms=self._timeout_ms)

    async def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        """Scroll the viewport. direction='up'|'down', amount in pixels."""
        self._require_page()
        return await actions.scroll(self._page, direction=direction, amount=amount)

    async def extract(self, selector: str) -> ActionResult:
        """Extract text from elements matching a CSS selector."""
        self._require_page()
        return await actions.extract(self._page, selector, timeout_ms=self._timeout_ms)

    async def extract_page(self) -> ActionResult:
        """Extract all readable text from the current page."""
        self._require_page()
        return await actions.extract_page(self._page)

    async def screenshot(self) -> ActionResult:
        """Capture the current viewport as PNG bytes."""
        self._require_page()
        return await actions.screenshot(self._page)

    async def get_dom_snapshot(self) -> ActionResult:
        """Return a compact DOM outline for LLM consumption."""
        self._require_page()
        return await actions.get_dom_snapshot(self._page)

    async def get_links(self) -> ActionResult:
        """Extract all links from the current page."""
        self._require_page()
        return await actions.get_links(self._page)

    async def current_url(self) -> str:
        """Return the current page URL. Empty string if browser not started."""
        if self._page is None:
            return ""
        return self._page.url

    async def current_title(self) -> str:
        """Return the current page title. Empty string if browser not started."""
        if self._page is None:
            return ""
        try:
            return await self._page.title()
        except Exception:
            return ""

    @property
    def is_running(self) -> bool:
        """True if the browser session is active."""
        return self._page is not None