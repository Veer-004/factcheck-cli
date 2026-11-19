"""API configuration from ``FACTCHECK_API_*`` environment variables.

Fact-checking settings (API keys, model, budgets) are not duplicated here: the
modules read them through ``factcheck.config.Config.from_env`` exactly as the CLI
commands do, including the optional ``.env`` file. This module only holds what
is specific to serving HTTP: upload limits, CORS and concurrency.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

MB = 1024 * 1024

DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173",
                        "http://127.0.0.1:5173")


class SettingsError(ValueError):
    """Invalid FACTCHECK_API_* configuration (raised at start-up)."""


class Settings(BaseModel):
    # Upload limits. The defaults equal the modules' own limits (factcheck_ocr.image_io: 25 MB;
    # factcheck-video / factcheck-audio: 500 MB), so the API never accepts what the module rejects.
    max_image_mb: int = Field(25, ge=1, le=25)
    max_video_mb: int = Field(500, ge=1, le=4096)
    max_audio_mb: int = Field(500, ge=1, le=4096)
    max_text_chars: int = Field(20_000, ge=1, le=100_000)  # request-size guard; the engine applies its own limit
    # Browser origins allowed to call the API (comma-separated). Credentials are never allowed.
    cors_origins: list[str] = Field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    # Simultaneous CPU-heavy jobs (video + audio share one limit; images have their own).
    media_concurrency: int = Field(1, ge=1, le=8)
    image_concurrency: int = Field(2, ge=1, le=16)
    text_concurrency: int = Field(4, ge=1, le=32)
    # Where uploads are written while processed (default: the system temp directory).
    temp_dir: Optional[str] = None

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, v: list[str]) -> list[str]:
        v = [o.strip().rstrip("/") for o in v if o.strip()]
        for origin in v:
            if origin == "*" or not origin.startswith(("http://", "https://")):
                raise ValueError(f"CORS origin must be an explicit http(s) origin, got {origin!r}")
        return v

    @classmethod
    def from_env(cls) -> "Settings":
        env = {
            "max_image_mb": "FACTCHECK_API_MAX_IMAGE_MB",
            "max_video_mb": "FACTCHECK_API_MAX_VIDEO_MB",
            "max_audio_mb": "FACTCHECK_API_MAX_AUDIO_MB",
            "max_text_chars": "FACTCHECK_API_MAX_TEXT_CHARS",
            "cors_origins": "FACTCHECK_API_CORS_ORIGINS",
            "media_concurrency": "FACTCHECK_API_MEDIA_CONCURRENCY",
            "image_concurrency": "FACTCHECK_API_IMAGE_CONCURRENCY",
            "text_concurrency": "FACTCHECK_API_TEXT_CONCURRENCY",
            "temp_dir": "FACTCHECK_API_TEMP_DIR",
        }
        values: dict[str, object] = {}
        for field, name in env.items():
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                continue
            values[field] = raw.split(",") if field == "cors_origins" else raw.strip()
        try:
            return cls(**values)
        except ValidationError as exc:
            err = exc.errors()[0]
            raise SettingsError(f"invalid {env[str(err['loc'][0])]}: {err['msg']}") from None

    def max_bytes(self, kind: str) -> int:
        return {"image": self.max_image_mb, "video": self.max_video_mb, "audio": self.max_audio_mb}[kind] * MB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
