"""FastAPI application: ``uvicorn factcheck_api.app:app --reload`` (Swagger UI at /docs)."""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from factcheck import envfile

from .config import MB, Settings, get_settings
from .errors import (
    APIError,
    api_error_handler,
    error_response,
    http_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from .routes import router
from .services import Backend

log = logging.getLogger("factcheck_api")
_UPLOAD_ROUTE = re.compile(r"^/api/v1/(?:fact-check|plan)/(image|video|audio)/?$")
MULTIPART_OVERHEAD = 1 * MB  # form boundaries and the small option fields

DESCRIPTION = """
REST API for the four FactCheck modules (text, image, video, audio). Each endpoint hands the input to the
existing module, which remains the source of truth for claims, evidence and verdicts, and returns one
JSON schema (`FactCheckResponse`) for every input type.

* A completed fact-check is always **200**, whatever the verdicts (including `INSUFFICIENT_EVIDENCE`).
* Inputs that were processed but had nothing checkable return **200** with `status: not_checked`.
* Errors use one shape (`ErrorResponse`) with a stable `code`.
* No database, accounts or history: uploads are deleted as soon as the response is ready.
"""


def create_app(settings: Optional[Settings] = None, backend: Optional[Backend] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for warning in envfile.load_env_file():  # same .env convention as the CLI commands
            log.warning(warning)
        app.state.limiters = {"text": anyio.CapacityLimiter(settings.text_concurrency),
                              "image": anyio.CapacityLimiter(settings.image_concurrency),
                              "media": anyio.CapacityLimiter(settings.media_concurrency),
                              "plan": anyio.CapacityLimiter(2)}
        yield

    app = FastAPI(title="FactCheck API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = settings
    app.state.backend = backend or Backend()

    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.middleware("http")
    async def request_guard(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        m = _UPLOAD_ROUTE.match(request.url.path)
        length = request.headers.get("content-length")
        if m and request.method == "POST" and length and length.isdigit():
            limit = settings.max_bytes(m.group(1))
            if int(length) > limit + MULTIPART_OVERHEAD:  # reject before the body is read
                response = error_response(request, 413, "file_too_large",
                                          f"The {m.group(1)} file exceeds the {limit // MB} MB upload limit.")
                response.headers["X-Request-ID"] = request.state.request_id
                return response
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - logged in full; the client gets a generic 500
            # Handled here, not only by the app-level handler, so the response still passes through
            # CORS and carries the request id.
            response = await unexpected_error_handler(request, exc)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # Added last so it wraps everything, error responses included. No credentials: the API has no
    # cookies or auth, so explicit origins without allow_credentials are enough.
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"],
                       expose_headers=["X-Request-ID"], max_age=600)
    app.include_router(router)
    return app


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()
