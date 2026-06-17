"""
browser/extractor.py

Pure text and DOM extraction functions.

No Playwright dependency. Takes raw HTML strings and returns
structured data. Tested independently of the browser.

Why separate from actions.py:
  - Pure functions are trivially testable without a live browser
  - The Verification Engine can call these on stored HTML snapshots
  - Keeping extraction logic here means actions.py stays thin

Functions
---------
    extract_text(html, selector)   → list[str]
    extract_page_text(html)        → str
    extract_dom_snapshot(html)     → str
    extract_links(html, base_url)  → list[dict]
    clean_text(text)               → str
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin


# ── Text extraction ───────────────────────────────────────────────────────────


def extract_page_text(html: str) -> str:
    """
    Convert raw HTML to clean readable text.

    Strips all tags, collapses whitespace, removes script/style blocks.
    This is what the Text Verifier and LLM Verifier receive — they
    work on clean text, not raw HTML.

    Uses stdlib HTMLParser only — no BeautifulSoup dependency.

    Parameters
    ----------
    html : Raw HTML string from Playwright's page.content()

    Returns
    -------
    Clean text with normalised whitespace. Empty string if input is empty.
    """
    if not html:
        return ""

    # Remove script and style blocks entirely before parsing
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)

    parser = _TextExtractParser()
    parser.feed(html)
    raw = parser.get_text()
    return clean_text(raw)


def extract_links(html: str, base_url: str = "") -> list[dict]:
    """
    Extract all <a href> links from HTML.

    Returns list of dicts with keys: text, href, absolute_url.
    Used by the Planner to understand what navigation options
    exist on the current page.

    Parameters
    ----------
    html     : Raw HTML string
    base_url : Page URL used to resolve relative hrefs
    """
    links: list[dict] = []
    for match in re.finditer(
        r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        href = match.group(1).strip()
        text = clean_text(re.sub(r"<[^>]+>", "", match.group(2)))
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href) if base_url else href
        links.append({"text": text, "href": href, "absolute_url": absolute})
    return links


# ── DOM snapshot ──────────────────────────────────────────────────────────────


def extract_dom_snapshot(html: str, max_depth: int = 4) -> str:
    """
    Produce a simplified DOM outline for LLM consumption.

    The full HTML of a real page is 50–200KB — far too large to
    send to an LLM as context. This function extracts a compact
    structural summary: tag names, ids, classes, and text previews
    for the most important elements only.

    Output format (one element per line):
        h1#title "Python (programming language)"
        div.mw-content-text
          p "Python is a high-level..."
          ul
            li "Designed by Guido van Rossum"

    Parameters
    ----------
    html      : Raw HTML string
    max_depth : How many nesting levels to include

    Returns
    -------
    Compact multi-line string, suitable for LLM prompt injection.
    """
    if not html:
        return ""

    parser = _DOMSnapshotParser(max_depth=max_depth)
    parser.feed(html)
    return parser.get_snapshot()


# ── Helpers ───────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Normalise whitespace in extracted text.

    - Collapse all runs of whitespace (spaces, tabs, newlines) to single space
    - Preserve paragraph breaks as double newlines
    - Strip leading/trailing whitespace
    """
    # Preserve paragraph-level breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Clean up space before/after newlines
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def truncate_for_llm(text: str, max_chars: int = 8000) -> str:
    """
    Truncate text to fit within LLM context limits.

    Takes the first max_chars characters. Adds a note if truncated
    so the LLM knows there's more content it didn't see.

    Used when passing page text to the LLM Verifier.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[...truncated — {len(text) - max_chars} more characters]"


# ── Internal parsers ──────────────────────────────────────────────────────────


# Tags whose content we skip entirely
_SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "meta", "link"})

# Tags that imply a line break in readable text
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "header", "footer", "main",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "td", "th",
    "br", "hr",
    "blockquote", "pre",
})


class _TextExtractParser(HTMLParser):
    """HTMLParser subclass that collects visible text content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


# Tags worth including in the DOM snapshot
_SNAPSHOT_TAGS = frozenset({
    "html", "body", "main", "article", "section", "nav", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "a", "button", "input", "textarea", "select", "form",
    "ul", "ol", "li", "table", "tr", "td", "th",
    "div", "span",
})


class _DOMSnapshotParser(HTMLParser):
    """HTMLParser subclass that builds a compact DOM outline."""

    def __init__(self, max_depth: int = 4) -> None:
        super().__init__()
        self._max_depth = max_depth
        self._depth: int = 0
        self._lines: list[str] = []
        self._skip_depth: int = 0
        self._current_text: list[str] = []
        self._line_count: int = 0
        self._max_lines: int = 120  # Cap output size

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0 or self._depth > self._max_depth:
            self._depth += 1
            return

        if tag not in _SNAPSHOT_TAGS:
            self._depth += 1
            return

        attr_dict = dict(attrs)
        parts = [tag]
        if "id" in attr_dict:
            parts[0] = f"{tag}#{attr_dict['id']}"
        if "class" in attr_dict:
            classes = attr_dict["class"].split()[:2]  # max 2 classes
            parts[0] += "." + ".".join(classes)
        if tag == "a" and "href" in attr_dict:
            parts.append(f'href="{attr_dict["href"][:60]}"')
        if tag == "input":
            if "type" in attr_dict:
                parts.append(f'type="{attr_dict["type"]}"')
            if "placeholder" in attr_dict:
                parts.append(f'placeholder="{attr_dict["placeholder"][:40]}"')

        indent = "  " * self._depth
        self._lines.append(f"{indent}{' '.join(parts)}")
        self._line_count += 1
        self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._depth <= self._max_depth + 1:
            stripped = data.strip()
            if stripped:
                self._current_text.append(stripped[:80])
                # Attach text preview to last line
                if self._lines:
                    text_preview = " ".join(self._current_text)[:80]
                    self._lines[-1] += f' "{text_preview}"'
                    self._current_text = []

    def get_snapshot(self) -> str:
        lines = self._lines[:self._max_lines]
        if len(self._lines) > self._max_lines:
            lines.append(f"[...{len(self._lines) - self._max_lines} more elements]")
        return "\n".join(lines)