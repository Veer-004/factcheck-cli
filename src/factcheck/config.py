"""Run configuration and resource limits.

All limits are caps, not guarantees. Values come from defaults, optionally
overridden by environment variables (``FACTCHECK_*``) or CLI flags.
Credentials are read from the environment only and are never stored on the
config object that gets printed or serialised.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Literal, Optional

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

GEMINI_KEY_ENV = "GEMINI_API_KEY"
TAVILY_KEY_ENV = "TAVILY_API_KEY"

# Verified against https://ai.google.dev/gemini-api/docs/models on 2026-10-26.
# Override with FACTCHECK_MODEL (e.g. gemini-3.5-flash-lite for higher quota).
DEFAULT_MODEL = "gemini-3.8-flash"
# Local fallback via the Ollama HTTP API (docs.ollama.com/api/chat).
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


class ConfigError(ValueError):
    """Invalid configuration (CLI exit code 2)."""


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Input / claims
    max_input_chars: int = Field(4000, ge=1, le=20000)
    max_checkable_claims: int = Field(6, ge=1, le=12)
    max_extracted_claims: int = Field(20, ge=1, le=50)

    # Retrieval
    max_queries_per_round: int = Field(2, ge=1, le=4)
    max_results_per_query: int = Field(5, ge=1, le=10)
    max_urls_per_claim: int = Field(6, ge=1, le=12)
    max_extract_urls: int = Field(5, ge=0, le=10)
    max_fallback_rounds: int = Field(1, ge=0, le=1)
    search_credit_budget: int = Field(40, ge=0, le=40)
    search_timeout_s: float = Field(20.0, gt=0, le=60)

    # Evidence
    max_passages_for_judge: int = Field(8, ge=1, le=12)
    max_passage_chars: int = Field(1200, ge=200, le=2000)
    min_passages_for_sufficiency: int = Field(2, ge=1, le=8)

    # LLM
    # Primary LLM provider (FACTCHECK_PROVIDER). "gemini" is the default; "ollama" runs
    # local-only. New providers (e.g. an OpenAI-compatible one) plug in here: anything
    # implementing llm.LLMClient works with the graph unchanged.
    provider: Literal["gemini", "ollama"] = "gemini"
    model: str = DEFAULT_MODEL
    # Local fallback: when the Gemini request fails for an operational reason (rate
    # limit, 5xx, timeout, network), the same request is sent to Ollama. See
    # llm.FallbackLLM for the exact policy.
    ollama_fallback: bool = True
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    # Per-request limit for local inference (CPU inference of a 4B model is slow);
    # always clipped to the remaining run deadline.
    ollama_timeout_s: float = Field(90.0, gt=0, le=300)
    # Gemini 3 guidance (ai.google.dev/gemini-api/docs/gemini-3): keep temperature at
    # the model default (sending < 1.0 "may lead to unexpected behavior, such as
    # looping"), and unset thinking_level defaults to "high", which can take
    # "significantly longer to reach a first ... output token". None = model default.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    thinking_level: Optional[Literal["low", "medium", "high"]] = "low"
    llm_timeout_s: float = Field(60.0, gt=0, le=120)
    max_llm_calls: int = Field(14, ge=1, le=30)
    max_judge_repairs: int = Field(1, ge=0, le=1)
    max_429_retries: int = Field(2, ge=0, le=2)
    max_retry_wait_s: float = Field(30.0, ge=0, le=60)

    # Orchestration
    max_concurrency: int = Field(3, ge=1, le=6)
    run_timeout_s: float = Field(180.0, gt=0, le=600)
    recursion_limit: int = Field(25, ge=10, le=100)

    # Date used for recency scoring; fixed per run for determinism.
    as_of: date = Field(default_factory=date.today)

    @field_validator("model", "ollama_model")
    @classmethod
    def _model_name(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 80 or not all(c.isalnum() or c in "-._/:" for c in v) or ".." in v:
            raise ValueError("model name must be a plain model identifier")
        return v

    @field_validator("ollama_url")
    @classmethod
    def _ollama_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        parts = urlsplit(v)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
            raise ValueError("ollama_url must be an http(s) base URL such as http://localhost:11434")
        return v

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        """Build a config from FACTCHECK_* environment variables plus overrides."""
        env_map = {
            "model": "FACTCHECK_MODEL",
            "search_credit_budget": "FACTCHECK_SEARCH_BUDGET",
            "max_checkable_claims": "FACTCHECK_MAX_CLAIMS",
            "run_timeout_s": "FACTCHECK_RUN_TIMEOUT",
            "max_concurrency": "FACTCHECK_MAX_CONCURRENCY",
            "thinking_level": "FACTCHECK_THINKING_LEVEL",
            "provider": "FACTCHECK_PROVIDER",
            "ollama_fallback": "FACTCHECK_OLLAMA_FALLBACK",
            "ollama_url": "FACTCHECK_OLLAMA_URL",
            "ollama_model": "FACTCHECK_OLLAMA_MODEL",
            "ollama_timeout_s": "FACTCHECK_OLLAMA_TIMEOUT",
        }
        values: dict[str, object] = {}
        for field, env in env_map.items():
            raw = os.environ.get(env)
            if raw is not None and raw.strip():
                values[field] = raw.strip()
        values.update({k: v for k, v in overrides.items() if v is not None})
        try:
            return cls(**values)
        except Exception as exc:  # pydantic.ValidationError
            raise ConfigError(f"invalid configuration: {exc}") from None


def credential_status() -> dict[str, bool]:
    """Report which credentials are present without exposing values."""
    return {
        GEMINI_KEY_ENV: bool(os.environ.get(GEMINI_KEY_ENV, "").strip()),
        TAVILY_KEY_ENV: bool(os.environ.get(TAVILY_KEY_ENV, "").strip()),
    }
