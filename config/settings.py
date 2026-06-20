"""
config/settings.py

Single source of truth for all AGENTX configuration.

Every module that needs a config value imports from here.
No module should ever call os.environ directly — that makes
configuration invisible and untestable.

Pydantic BaseSettings reads values from environment variables
and a .env file automatically. Adding a new config value means
adding one field here and one line to .env.example.

Dependencies: pydantic-settings, python-dotenv
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: the directory containing this config/ folder
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    AGENTX runtime configuration.

    All values are read from environment variables or .env file.
    Defaults are set for local development — change via .env for
    any deployment.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM — provider-agnostic ───────────────────────────────────────
    #
    # Switch providers by changing LLM_PROVIDER in .env.
    # No code changes required anywhere in the system.
    #
    # Supported values: ollama | anthropic | openai
    # Default: ollama (local, zero cost, no API key required)
    #
    llm_provider: str = Field(
        default="ollama",
        description=(
            "LLM backend to use. One of: ollama, anthropic, openai. "
            "ollama is the default — runs locally, no API key required."
        ),
    )
    llm_model: str = Field(
        default="qwen3:latest",
        description=(
            "Model name passed to the active provider. "
            "ollama: 'qwen3:latest' | anthropic: 'claude-sonnet-4-20250514' | "
            "openai: 'gpt-4o'"
        ),
    )
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL for the LLM provider's API. "
            "Used by ollama. Ignored by anthropic and openai."
        ),
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens per LLM response.",
    )
    llm_temperature: float = Field(
        default=0.2,
        description=(
            "Temperature for LLM calls. Low value (0.0–0.3) reduces "
            "planning randomness and improves reproducibility."
        ),
    )
    llm_timeout_seconds: int = Field(
        default=120,
        description=(
            "HTTP timeout in seconds for LLM API calls. "
            "Local Ollama models can be slow on first load — 120s is safe."
        ),
    )

    # ── LLM — provider API keys (only read when provider is active) ───
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Only required when LLM_PROVIDER=anthropic.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Only required when LLM_PROVIDER=openai.",
    )
    groq_api_key: str = Field(
        default="",
        description=(
            "Groq API key. Only required when LLM_PROVIDER=groq. "
            "Free key at https://console.groq.com"
        ),
    )

    # ── Database ──────────────────────────────────────────────────────
    db_path: Path = Field(
        default=PROJECT_ROOT / "db" / "agentx.db",
        description="Path to the SQLite database file.",
    )

    # ── Browser ───────────────────────────────────────────────────────
    browser_headless: bool = Field(
        default=True,
        description="Run Playwright in headless mode. Set False to watch the browser.",
    )
    browser_timeout_ms: int = Field(
        default=30_000,
        description="Default Playwright action timeout in milliseconds.",
    )
    browser_slow_mo_ms: int = Field(
        default=50,
        description=(
            "Milliseconds between Playwright actions. "
            "Mimics human speed; also helps with sites that block instant automation."
        ),
    )

    # ── Execution ─────────────────────────────────────────────────────
    max_steps_per_task: int = Field(
        default=20,
        description="Hard limit on steps before a task is aborted.",
    )
    max_retries_per_step: int = Field(
        default=3,
        description="Maximum retry attempts before escalating to REPLAN or ABORT.",
    )
    max_corrections_per_task: int = Field(
        default=6,
        description="Maximum total self-correction events before a task is aborted.",
    )
    step_timeout_seconds: int = Field(
        default=60,
        description="Maximum time in seconds for a single step before it is marked timed out.",
    )

    # ── API ───────────────────────────────────────────────────────────
    api_host: str = Field(
        default="127.0.0.1",
        description="Host for the FastAPI server.",
    )
    api_port: int = Field(
        default=8000,
        description="Port for the FastAPI server.",
    )
    api_key: str = Field(
        default="dev-key-change-in-production",
        description=(
            "API key for authenticating requests to the AGENTX API. "
            "Checked via X-API-Key header."
        ),
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level. One of: DEBUG, INFO, WARNING, ERROR.",
    )
    log_file: Path = Field(
        default=PROJECT_ROOT / "agentx.log",
        description="Path to the structured JSON log file.",
    )

    # ── Evaluation ────────────────────────────────────────────────────
    benchmark_dataset_path: Path = Field(
        default=PROJECT_ROOT / "evaluation" / "benchmarks" / "dataset.json",
        description="Path to the benchmark task dataset JSON file.",
    )
    benchmark_reports_dir: Path = Field(
        default=PROJECT_ROOT / "evaluation" / "reports",
        description="Directory where benchmark reports are saved.",
    )


# Module-level singleton.
# Import this instance everywhere:
#   from config.settings import settings
#
# Never instantiate Settings() directly in other modules —
# that would re-read .env on every import, and would make
# patching in tests unreliable.
settings = Settings()