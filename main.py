"""
main.py — AGENTX CLI entry point.

Usage
-----
    python main.py "your goal here"
    python main.py --serve
    python main.py --health
    python main.py --browse https://example.com
    python main.py --benchmark
    python main.py --benchmark --suite easy
    python main.py --benchmark --ids nav_001,extract_001,nav_002
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
    parser.add_argument("--serve", action="store_true",
        help="Start the FastAPI server.")
    parser.add_argument("--health", action="store_true",
        help="Check system health and exit.")
    parser.add_argument("--browse", metavar="URL",
        help="Browser smoke test: navigate URL and exercise all actions.")
    parser.add_argument("--benchmark", action="store_true",
        help="Run the benchmark suite.")
    parser.add_argument("--suite", default="full",
        choices=["full", "easy", "medium", "hard"],
        help="Benchmark suite to run (default: full).")
    parser.add_argument("--category", default=None,
        help="Filter benchmark to one category.")
    parser.add_argument("--ids", default=None,
        help="Comma-separated benchmark task IDs to run (e.g. nav_001,nav_002).")
    parser.add_argument("--timeout", type=int, default=120,
        help="Per-task benchmark timeout in seconds (default: 120).")
    parser.add_argument("--goal-type", default="unknown",
        choices=["navigation", "extraction", "form_interaction", "multi_step", "unknown"],
        help="Goal category hint for the planner.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    return parser.parse_args()


# ── Handlers ──────────────────────────────────────────────────────────────────


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
    from browser.controller import BrowserController

    print(f"\n  Opening browser → {url}")
    print("  " + "─" * 50)

    async with BrowserController() as browser:
        result = await browser.navigate(url)
        if not result.success:
            print(f"  [FAIL] navigate: {result.error}")
            return
        print(f"  [OK]   navigate   → title: {result.output['title'][:60]}")
        print(f"                      url:   {result.output['url'][:60]}")

        result = await browser.extract_page()
        if result.success:
            preview = result.output["text"][:300].replace("\n", " ")
            print(f"  [OK]   extract_page → {result.output['char_count']} chars")
            print(f"                        preview: {preview!r}")
        else:
            print(f"  [FAIL] extract_page: {result.error}")

        result = await browser.get_links()
        if result.success:
            print(f"  [OK]   get_links   → {result.output['count']} links found")
            for link in result.output["links"][:3]:
                print(f"                        {link['text'][:30]!r:32} → {link['href'][:50]}")
        else:
            print(f"  [FAIL] get_links: {result.error}")

        result = await browser.get_dom_snapshot()
        if result.success:
            snap = result.output["snapshot"]
            print(f"  [OK]   dom_snapshot → {len(snap.splitlines())} lines")
            print(f"         {chr(10).join(snap.splitlines()[:4])}")
        else:
            print(f"  [FAIL] dom_snapshot: {result.error}")

        result = await browser.scroll("down", 300)
        if result.success:
            print(f"  [OK]   scroll      → scrollY: {result.output['scroll_y']}px")

        result = await browser.screenshot()
        if result.success:
            print(f"  [OK]   screenshot  → {result.output['size']:,} bytes")

        print()
        print(f"  url:   {await browser.current_url()}")
        print(f"  title: {await browser.current_title()}")
        print()
        print("  Browser layer: ALL ACTIONS OK")


async def run_goal(goal: str, goal_type: str) -> None:
    from core.orchestrator import Orchestrator

    logger = get_logger(__name__)
    logger.info("goal_submitted", goal=goal[:80], goal_type=goal_type)

    print(f"\n  Goal:  {goal}")
    print(f"  Type:  {goal_type}  |  Model: {settings.llm_model}")
    print()

    task = await Orchestrator().run(goal=goal, goal_type=goal_type)

    print()
    print("  " + "═" * 50)
    print(f"  Task ID:     {task.id}")
    print(f"  Status:      {task.status.value if hasattr(task.status, 'value') else task.status}")
    print(f"  Steps:       {task.steps_taken}  |  Corrections: {task.corrections}  |  Tokens: {task.tokens_used}")
    if task.duration_seconds:
        print(f"  Duration:    {task.duration_seconds:.1f}s")
    print()
    print("  RESULT:")
    print("  " + "─" * 50)
    # Trim long results — show first meaningful lines only
    result_text = (task.result or "").strip()
    lines = [l for l in result_text.splitlines() if l.strip()][:15]
    for line in lines:
        print(f"  {line}")
    if len(result_text.splitlines()) > 15:
        print(f"  ... ({len(result_text.splitlines())} lines total)")
    print("  " + "═" * 50)


async def run_benchmark(
    suite: str,
    category: str | None,
    ids: str | None,
    timeout: int,
) -> None:
    from evaluation.benchmark_runner import BenchmarkRunner
    from evaluation.metrics import MetricsCalculator
    from evaluation.reporter import Reporter

    task_ids = [x.strip() for x in ids.split(",")] if ids else None

    runner = BenchmarkRunner(
        suite=suite,
        category=category,
        task_ids=task_ids,
        timeout=timeout,
    )
    results = await runner.run()

    if not results:
        print("  No results. Check --suite / --ids / dataset.json.")
        return

    summary = MetricsCalculator.compute(results, run_id=runner._run_id)
    reporter = Reporter()
    reporter.print_report(summary, results)
    reporter.save_report(summary, results, runner._run_id)


def serve(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=False, log_config=None)


# ── Entry point ───────────────────────────────────────────────────────────────


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

    if args.benchmark:
        asyncio.run(run_benchmark(args.suite, args.category, args.ids, args.timeout))
        return

    if not args.goal:
        print("Usage:")
        print("  python main.py \"your goal here\"")
        print("  python main.py --benchmark --suite easy")
        print("  python main.py --benchmark --ids nav_001,nav_002,extract_001")
        print("  python main.py --serve")
        print("  python main.py --health")
        print("  python main.py --browse https://example.com")
        sys.exit(1)

    asyncio.run(run_goal(args.goal, args.goal_type))


if __name__ == "__main__":
    main()