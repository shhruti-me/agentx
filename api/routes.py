"""
api/routes.py

All HTTP route handlers for AGENTX.

Route handlers are thin. They:
  1. Validate input (Pydantic does this automatically)
  2. Call into core modules
  3. Return a standard response envelope

No business logic lives here. If a handler is more than ~15 lines,
the logic belongs in a core module.

Endpoints
---------
  GET  /health
  POST /v1/run
  GET  /v1/status/{task_id}
  GET  /v1/results/{task_id}
  GET  /v1/tasks
  POST /v1/benchmark          (stub — Week 6)
  GET  /v1/benchmark/{run_id} (stub — Week 6)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config.settings import settings
from log.logger import get_logger
from memory.db import check_db
from memory.task_memory import get_task, list_tasks, write_task
from core.models import GoalType, Task
from llm.factory import get_llm_client

logger = get_logger(__name__)
router = APIRouter()


# ── Auth ──────────────────────────────────────────────────────────────────────


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Depends(_api_key_header)) -> str:
    """
    Validate X-API-Key header against settings.api_key.
    Returns the key on success. Raises 401 on failure.
    """
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


# ── Response envelope ─────────────────────────────────────────────────────────


def ok(data: Any, request: Request) -> dict:
    return {
        "success": True,
        "data": data,
        "error": None,
        "request_id": getattr(request.state, "request_id", None),
    }


def err(message: str, request: Request) -> dict:
    return {
        "success": False,
        "data": None,
        "error": message,
        "request_id": getattr(request.state, "request_id", None),
    }


# ── Request / Response models ─────────────────────────────────────────────────


class RunRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=2000, description="Natural language goal.")
    goal_type: str = Field(
        default="unknown",
        description="One of: navigation, extraction, form_interaction, multi_step, unknown.",
    )


class RunResponse(BaseModel):
    task_id: str
    status: str


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/health", tags=["system"])
async def health(request: Request) -> dict:
    """
    Verify the system is operational.
    Checks: DB connectivity, table existence, LLM availability.
    """
    db_ok = check_db()

    client = get_llm_client()
    llm_ok = await client.is_available()

    status = "ok" if (db_ok and llm_ok) else "degraded"

    data = {
        "status": status,
        "db": "ok" if db_ok else "error",
        "llm": "ok" if llm_ok else "unavailable",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }

    logger.info("health_check", **data)
    return ok(data, request)


@router.post(
    "/v1/run",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def run_task(body: RunRequest, request: Request) -> dict:
    """
    Submit a goal for the agent to execute.

    Week 1: creates a Task record and returns the task_id.
    Week 3+: hands the task to the Orchestrator and runs it.

    Returns task_id immediately. Poll /v1/status/{task_id} for progress.
    """
    # Resolve goal_type — default to UNKNOWN if unrecognised
    try:
        goal_type = GoalType(body.goal_type)
    except ValueError:
        goal_type = GoalType.UNKNOWN

    task = Task(goal=body.goal, goal_type=goal_type)
    write_task(task)

    logger.info("task_created", task_id=task.id, goal_type=goal_type, goal=body.goal[:80])

    # Week 3: replace this block with await orchestrator.run(task)
    # For now, task sits in PENDING state until the Orchestrator exists.

    return ok({"task_id": task.id, "status": task.status}, request)


@router.get(
    "/v1/status/{task_id}",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def get_status(task_id: str, request: Request) -> dict:
    """
    Poll the status of a submitted task.
    """
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")

    return ok(
        {
            "task_id":      task.id,
            "status":       task.status,
            "steps_taken":  task.steps_taken,
            "corrections":  task.corrections,
            "result":       task.result,
            "started_at":   task.started_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration_seconds": task.duration_seconds,
        },
        request,
    )


@router.get(
    "/v1/results/{task_id}",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def get_results(task_id: str, request: Request) -> dict:
    """
    Retrieve full task results including plan and step details.
    """
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")

    steps = []
    if task.plan:
        steps = [s.to_dict() for s in task.plan.steps]

    return ok(
        {
            **task.to_dict(),
            "steps": steps,
        },
        request,
    )


@router.get(
    "/v1/tasks",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def get_tasks(
    request: Request,
    status: str | None = Query(default=None, description="Filter by status."),
    goal_type: str | None = Query(default=None, description="Filter by goal type."),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """
    List recent tasks with optional filters.
    """
    tasks = list_tasks(status=status, goal_type=goal_type, limit=limit)
    return ok({"tasks": tasks, "count": len(tasks)}, request)


# ── Benchmark stubs (Week 6) ──────────────────────────────────────────────────


@router.post(
    "/v1/benchmark",
    tags=["evaluation"],
    dependencies=[Depends(require_api_key)],
)
async def start_benchmark(request: Request) -> dict:
    """Stub — implemented in Week 6 when BenchmarkRunner exists."""
    raise HTTPException(status_code=501, detail="Benchmark runner not implemented yet. Coming in Week 6.")


@router.get(
    "/v1/benchmark/{run_id}",
    tags=["evaluation"],
    dependencies=[Depends(require_api_key)],
)
async def get_benchmark(run_id: str, request: Request) -> dict:
    """Stub — implemented in Week 6."""
    raise HTTPException(status_code=501, detail="Benchmark runner not implemented yet. Coming in Week 6.")