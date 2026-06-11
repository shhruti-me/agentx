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

Week 1: creates a Task record, writes it to SQLite, prints the task_id.
Week 3: adds Orchestrator call so the agent actually executes the goal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from config.settings import settings
from log.logger import get_logger, setup_logging
from memory.db import init_db
from memory.task_memory import write_task
from core.models import GoalType, Task


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
        print("\n  [!] Database error. Run init_db() or check db/ directory permissions.")
        sys.exit(1)
    if not llm_ok:
        print(f"\n  [!] LLM provider unavailable. For Ollama: run 'ollama serve' then 'ollama pull {settings.llm_model}'")
        sys.exit(1)

    print("\n  System ready.")


async def run_goal(goal: str, goal_type_str: str) -> None:
    logger = get_logger(__name__)

    try:
        goal_type = GoalType(goal_type_str)
    except ValueError:
        goal_type = GoalType.UNKNOWN

    task = Task(goal=goal, goal_type=goal_type)
    write_task(task)

    logger.info(
        "task_submitted",
        task_id=task.id,
        goal=goal[:80],
        goal_type=goal_type,
    )

    print(f"\n  Task created")
    print(f"  ID:        {task.id}")
    print(f"  Status:    {task.status}")
    print(f"  Goal:      {goal}")
    print(f"  Goal type: {goal_type}")
    print()
    print("  Week 1: task is written to SQLite but not yet executed.")
    print("  The Orchestrator (Week 3) will execute it.")
    print()
    print(f"  Check status: GET /v1/status/{task.id}")


def serve(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,  # suppress uvicorn's default logger — we use ours
    )


def main() -> None:
    args = _parse_args()

    # Logging and DB must be ready before anything else
    setup_logging()
    init_db()

    if args.serve:
        print(f"  Starting AGENTX API on http://{args.host}:{args.port}")
        print(f"  Docs: http://{args.host}:{args.port}/docs")
        print(f"  API key: {settings.api_key}")
        serve(args.host, args.port)
        return

    if args.health:
        print("\n  AGENTX health check")
        print("  " + "─" * 30)
        asyncio.run(run_health_check())
        return

    if not args.goal:
        print("Usage: python main.py \"your goal here\"")
        print("       python main.py --serve")
        print("       python main.py --health")
        sys.exit(1)

    asyncio.run(run_goal(args.goal, args.goal_type))


if __name__ == "__main__":
    main()