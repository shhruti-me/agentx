"""
api/app.py

FastAPI application factory for AGENTX.

Responsibilities:
  - Instantiate the FastAPI app
  - Run startup/shutdown logic via lifespan (init DB, check LLM)
  - Register routes
  - Add middleware (request ID, timing)

Nothing else. No route logic lives here.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response

from config.settings import settings
from log.logger import get_logger, setup_logging
from memory.db import init_db, check_db
from llm.factory import get_llm_client

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown logic.

    Startup:
      1. Initialise logging
      2. Initialise SQLite schema
      3. Warm up LLM client (instantiates provider, does not make a call)

    Shutdown:
      - Nothing to clean up in V1. Playwright sessions are per-task
        and closed after each task completes.
    """
    # Logging first — everything after this can log
    setup_logging()
    logger.info("agentx_startup", version="1.0.0", provider=settings.llm_provider)

    # Database
    init_db()
    logger.info("db_initialised", path=str(settings.db_path))

    # LLM client — instantiate now so the first request isn't slow
    client = get_llm_client()
    available = await client.is_available()
    if available:
        logger.info("llm_ready", provider=settings.llm_provider, model=settings.llm_model)
    else:
        logger.warning(
            "llm_unavailable",
            provider=settings.llm_provider,
            model=settings.llm_model,
            hint="Check Ollama is running: ollama serve",
        )

    yield

    logger.info("agentx_shutdown")


# ── App ───────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="AGENTX",
        description="Autonomous browser agent with LLM-based planning and self-correction.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────

    @app.middleware("http")
    async def request_middleware(request: Request, call_next) -> Response:
        """
        Attach a request_id to every request and log timing.
        The request_id is returned in the response header so callers
        can correlate their request with log lines.
        """
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.monotonic()

        response = await call_next(request)

        duration_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-MS"] = str(duration_ms)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        return response

    # ── Routes ─────────────────────────────────────────────────────────────────

    from api.routes import router
    app.include_router(router)

    return app


# Module-level app instance used by uvicorn
app = create_app()