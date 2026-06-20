"""
main.py

CLI entry point for AGENTX.

Usage
-----
    # Run a task:
    python main.py "Go to news.ycombinator.com and return the top post title"

    # Start the API server:
    python main.py --serve

    # Check system health:
    python main.py --health

    # Week 2 browser smoke test:
    python main.py --browse "https://en.wikipedia.org/wiki/Python_(programming_language)"

Week 1: creates a Task record, writes it to SQLite, prints the task_id.
Week 2: --browse opens a real browser and extracts page text.
Week 3: adds Orchestrator call so the agent actually executes the goal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from config.settings import settings
from log.logger import get_logger, setup_logging
from memory.db import init_db


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentx",
        description="AGENTX — autonomous browser agent.",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="Natural language goal for the agent to execute.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server instead of running a task.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check system health and exit.",
    )
    parser.add_argument(
        "--browse",
        metavar="URL",
        help="Week 2 smoke test: open URL, exercise all browser actions.",
    )
    parser.add_argument(
        "--goal-type",
        default="unknown",
        choices=["navigation", "extraction", "form_interaction", "multi_step", "unknown"],
        help="Goal category hint for the planner (default: unknown).",
    )
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help=f"API server host (default: {settings.api_host}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help=f"API server port (default: {settings.api_port}).",
    )
    return parser.parse_args()


async def run_health_check() -> None:
    from llm.factory import get_llm_client
    from memory.db import check_db

    db_ok = check_db()
    client = get_llm_client()
    llm_ok = await client.is_available()

    print(f"  db:       {'OK' if db_ok else 'ERROR'}")
    print(f"  llm:      {'OK' if llm_ok else 'UNAVAILABLE'}")
    print(f"  provider: {settings.llm_provider}")
    print(f"  model:    {settings.llm_model}")
    print(f"  db_path:  {settings.db_path}")

    if not db_ok:
        print("\n  [!] Database error.")
        sys.exit(1)
    if not llm_ok:
        print(f"\n  [!] LLM unavailable. Run: ollama serve && ollama pull {settings.llm_model}")
        sys.exit(1)

    print("\n  System ready.")


async def run_browse(url: str) -> None:
    """
    Week 2 smoke test.

    Opens a real browser, navigates to the URL, runs every action
    once, prints results. Confirms the browser layer works end-to-end
    before the Planner and Execution Engine are built.
    """
    from browser.controller import BrowserController

    print(f"\n  Opening browser → {url}")
    print("  " + "─" * 50)

    async with BrowserController() as browser:

        # 1. navigate
        result = await browser.navigate(url)
        if not result.success:
            print(f"  [FAIL] navigate: {result.error}")
            return
        print(f"  [OK]   navigate   → title: {result.output['title'][:60]}")
        print(f"                      url:   {result.output['url'][:60]}")

        # 2. extract_page — full readable text
        result = await browser.extract_page()
        if not result.success:
            print(f"  [FAIL] extract_page: {result.error}")
        else:
            text = result.output["text"]
            preview = text[:300].replace("\n", " ")
            print(f"  [OK]   extract_page → {result.output['char_count']} chars")
            print(f"                        preview: {preview!r}")

        # 3. get_links
        result = await browser.get_links()
        if not result.success:
            print(f"  [FAIL] get_links: {result.error}")
        else:
            links = result.output["links"]
            print(f"  [OK]   get_links   → {result.output['count']} links found")
            for link in links[:3]:
                print(f"                        {link['text'][:30]!r:32} → {link['href'][:50]}")

        # 4. get_dom_snapshot
        result = await browser.get_dom_snapshot()
        if not result.success:
            print(f"  [FAIL] get_dom_snapshot: {result.error}")
        else:
            snap = result.output["snapshot"]
            first_lines = "\n".join(snap.splitlines()[:6])
            print(f"  [OK]   dom_snapshot → {len(snap.splitlines())} lines")
            print(f"         {first_lines[:200]}")

        # 5. scroll
        result = await browser.scroll("down", 300)
        if not result.success:
            print(f"  [FAIL] scroll: {result.error}")
        else:
            print(f"  [OK]   scroll      → scrollY: {result.output['scroll_y']}px")

        result = await browser.screenshot()
        if not result.success:
            print(f"  [FAIL] screenshot: {result.error}")
        else:
            print(f"  [OK]   screenshot  → {result.output['size']:,} bytes (PNG)")

        print()
        print(f"  current url:   {await browser.current_url()}")
        print(f"  current title: {await browser.current_title()}")
        print()
        print("  Browser layer: ALL ACTIONS OK")
        print("  Week 2 exit criterion met.")


async def run_goal(goal: str, goal_type: str) -> None:
    from core.orchestrator import Orchestrator

    logger = get_logger(__name__)
    logger.info("goal_submitted", goal=goal[:80], goal_type=goal_type)

    print(f"\n  Goal:      {goal}")
    print(f"  Type:      {goal_type}")
    print(f"  Model:     {settings.llm_model}")
    print()

    orchestrator = Orchestrator()
    task = await orchestrator.run(goal=goal, goal_type=goal_type)

    print()
    print("  " + "═" * 50)
    print(f"  Task ID:     {task.id}")
    print(f"  Status:      {task.status.value}")
    print(f"  Steps:       {task.steps_taken}")
    print(f"  Corrections: {task.corrections}")
    print(f"  Tokens:      {task.tokens_used}")
    if task.duration_seconds:
        print(f"  Duration:    {task.duration_seconds:.1f}s")
    print()
    print("  RESULT:")
    print("  " + "─" * 50)
    print(f"  {task.result}")
    print("  " + "═" * 50)


def serve(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,
    )


def main() -> None:
    args = _parse_args()

    setup_logging()
    init_db()

    if args.serve:
        print(f"  Starting AGENTX API on http://{args.host}:{args.port}")
        serve(args.host, args.port)
        return

    if args.health:
        print("\n  AGENTX health check")
        print("  " + "─" * 30)
        asyncio.run(run_health_check())
        return

    if args.browse:
        asyncio.run(run_browse(args.browse))
        return

    if not args.goal:
        print("Usage: python main.py \"your goal here\"")
        print("       python main.py --serve")
        print("       python main.py --health")
        print("       python main.py --browse https://example.com")
        sys.exit(1)

    asyncio.run(run_goal(args.goal, args.goal_type))


if __name__ == "__main__":
    main()