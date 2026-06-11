"""
log/logger.py

Structured JSON logger for AGENTX.

Every significant event in the system — task started, step dispatched,
verification result, correction applied, LLM called — is logged as a
JSON object to both stdout and agentx.log.

Why structured JSON:
  - Every field is queryable: grep, jq, or any log aggregator
  - No regex parsing needed to extract task_id, step_index, duration_ms
  - tail -f agentx.log | python3 -m json.tool works out of the box
  - Consistent shape across all modules makes debugging deterministic

Usage
-----
    from log.logger import get_logger

    logger = get_logger(__name__)

    logger.info("step_completed", task_id="abc123", step_index=2, duration_ms=412)
    logger.warning("verification_uncertain", task_id="abc123", method="llm")
    logger.error("correction_failed", task_id="abc123", strategy="replan")

The first positional argument is the event name.
All keyword arguments become top-level fields in the JSON object.
Standard fields (ts, level, event, module) are added automatically.

Dependencies: Python stdlib only.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── JSON formatter ────────────────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.

    Standard fields always present:
      ts      : ISO-8601 UTC timestamp
      level   : DEBUG | INFO | WARNING | ERROR | CRITICAL
      event   : The event name (first arg passed to logger.info() etc.)
      module  : The logger name (__name__ of the calling module)

    Extra fields come from the `extra` dict passed to the log call,
    or from keyword arguments captured by AgentXLogger.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "level": record.levelname,
            "event": record.getMessage(),
            "module": record.name,
        }

        # Merge any extra fields passed via extra={} or captured kwargs
        for key, value in record.__dict__.items():
            if key not in _STDLIB_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value

        # Attach exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# Fields that exist on every LogRecord — exclude from the JSON payload
# to avoid noise. This list covers all stdlib LogRecord attributes.
_STDLIB_RECORD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})


# ── AgentXLogger wrapper ──────────────────────────────────────────────────────


class AgentXLogger:
    """
    Thin wrapper around stdlib Logger that accepts keyword arguments
    as structured fields instead of requiring an `extra={}` dict.

    This gives call sites a clean, readable interface:

        logger.info("step_completed", task_id="abc", duration_ms=412)

    instead of:

        logger.info("step_completed", extra={"task_id": "abc", "duration_ms": 412})

    Both forms work — the wrapper normalises them.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, extra=kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log ERROR with current exception info attached."""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, event, **kwargs)


# ── Setup ─────────────────────────────────────────────────────────────────────


def _setup_root_logger(log_level: str, log_file: Path) -> None:
    """
    Configure the root logger once at startup.

    Two handlers:
      stdout  — all log levels at or above log_level
      file    — same, rotating at 10MB, keeping 3 backups

    Called once by setup_logging(). Subsequent calls to get_logger()
    just return a child of the already-configured root.
    """
    root = logging.getLogger()

    # Guard: don't add handlers if already configured
    if root.handlers:
        return

    root.setLevel(log_level.upper())
    formatter = _JSONFormatter()

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # rotating file handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


_logging_configured = False


def setup_logging(log_level: str | None = None, log_file: Path | None = None) -> None:
    """
    Initialise the logging system from settings.

    Call this once at application startup — in main.py and in
    api/app.py lifespan. Safe to call multiple times; subsequent
    calls are no-ops.

    Parameters
    ----------
    log_level : Override settings.log_level. Useful in tests.
    log_file  : Override settings.log_file. Useful in tests.
    """
    global _logging_configured
    if _logging_configured:
        return

    from config.settings import settings

    _setup_root_logger(
        log_level=log_level or settings.log_level,
        log_file=log_file or settings.log_file,
    )
    _logging_configured = True


def get_logger(name: str) -> AgentXLogger:
    """
    Return an AgentXLogger for the given module name.

    Convention: every module calls this at the top of the file:

        from log.logger import get_logger
        logger = get_logger(__name__)

    If setup_logging() has not been called yet, the stdlib root
    logger will handle output using its default configuration
    (stderr, no JSON). Always call setup_logging() at startup.
    """
    return AgentXLogger(logging.getLogger(name))