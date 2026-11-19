"""Consistent JSON errors and the mapping from module exceptions to HTTP status codes.

Every error response has the same shape (``ErrorResponse``)::

    {"status": "error", "error": {"code": "...", "message": "...", "details": [...]}, "request_id": "..."}

Messages are sanitised: API keys (``factcheck.security.redact_secrets``) and
filesystem paths are removed, and unexpected exceptions only ever return a
generic message. The full exception (with stack trace) is logged server-side
under the same ``request_id``.

A completed fact-check is never an error, whatever its verdicts are: an
INSUFFICIENT_EVIDENCE or PROCESSING_FAILED claim is part of a 200 response.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from factcheck.security import redact_secrets

log = logging.getLogger("factcheck_api")

# Windows (C:\...) and POSIX (/tmp/..., /home/...) absolute paths.
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:tmp|var|home|Users|private|usr|opt|mnt|root)/)[^\s'\",;:)]*")


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code, e.g. `unsupported_media_type`.")
    message: str = Field(description="Human-readable explanation, safe to show to the user.")
    details: list[dict[str, Any]] = Field(default_factory=list,
                                          description="Per-field problems for validation errors; otherwise empty.")


class ErrorResponse(BaseModel):
    status: str = Field("error", description="Always `error`.")
    error: ErrorDetail
    request_id: Optional[str] = Field(None, description="Correlates the response with the server log.")


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: Optional[list[dict]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def clean_message(message: str, replacements: Optional[dict[str, str]] = None) -> str:
    """Remove secrets and filesystem paths from a message meant for the client."""
    for old, new in (replacements or {}).items():
        if old:
            message = message.replace(old, new)
    message = redact_secrets(message)
    return _PATH.sub("<file>", message)


def _request_id(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


def error_response(request: Request, status_code: int, code: str, message: str,
                   details: Optional[list[dict]] = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details or []),
                         request_id=_request_id(request))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    log.info("request %s failed: %s %s", _request_id(request), exc.status_code, exc.code)
    return error_response(request, exc.status_code, exc.code, exc.message, exc.details)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Echo field locations and messages, never the submitted values.
    details = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", ""), "type": e.get("type", "")}
               for e in exc.errors()]
    return error_response(request, 422, "validation_error", "The request is invalid; see details.", details)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    codes = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return error_response(request, exc.status_code, codes.get(exc.status_code, "http_error"), message)


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("request %s: unexpected %s", _request_id(request), type(exc).__name__)
    return error_response(request, 500, "internal_error", "An unexpected internal error occurred.")
