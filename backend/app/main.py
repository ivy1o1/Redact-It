"""Application factory, lifespan wiring and the uniform error envelope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import api_router, system_router
from .config import Settings, get_settings
from .errors import RedactItError
from .llm.adapter import LlamaCppClient
from .logging_setup import configure_logging, get_logger
from .services.analysis import AnalysisService
from .storage import SessionStore

logger = get_logger("app")

_HTTP_CODES = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
}

_DESCRIPTION = """
Local-first, instruction-driven document redaction.

Upload a PDF or DOCX, describe in plain English what to remove, review the
proposed matches, and download sanitised copies. The originals are never
modified.

**Everything runs on this machine.** Text interpretation uses a local
llama.cpp server; document bytes, extracted text and instructions are never sent
to a cloud provider, and no telemetry is configured.

RedactIt proposes candidates - it does not understand documents. Detection can
miss sensitive data and can propose matches that are not sensitive. Nothing is
removed until a human approves it, and it is not a legal-compliance tool.
"""


def _error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _safe_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Validation detail without the submitted values.

    ``exc.errors()`` echoes back the offending input, which for the analyze
    endpoint is the user's instruction. That must not reach a log or a response
    body beyond the field path.
    """
    safe = []
    for error in exc.errors():
        location = error.get("loc", ())
        safe.append(
            {
                "field": ".".join(str(part) for part in location if part != "body"),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )
    return safe


async def _cleanup_loop(store: SessionStore, interval: int) -> None:
    """Delete expired sessions and download links.

    The purge does file I/O, so it runs off the event loop.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            sessions = await asyncio.to_thread(store.purge_expired)
            downloads = await asyncio.to_thread(store.purge_expired_downloads)
            if sessions or downloads:
                logger.info(
                    "cleanup_tick",
                    extra={"sessions_purged": sessions, "downloads_purged": downloads},
                )
        except Exception:  # pragma: no cover - the loop must survive any failure
            logger.exception("cleanup_failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)

    store = SessionStore(settings)
    client = LlamaCppClient(settings)
    service = AnalysisService(settings, store, client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.resolved_storage_root.mkdir(parents=True, exist_ok=True)
        orphans = await asyncio.to_thread(store.purge_orphans)
        cleanup = asyncio.create_task(
            _cleanup_loop(store, settings.cleanup_interval_seconds)
        )
        logger.info(
            "startup",
            extra={
                "version": settings.app_version,
                "environment": settings.environment,
                "interpretation_mode": settings.interpretation_mode,
                "llm_base_url": settings.llm_base_url,
                "llm_model": settings.llm_model,
                "storage_root": str(settings.resolved_storage_root),
                "orphaned_workdirs_removed": orphans,
            },
        )
        try:
            yield
        finally:
            cleanup.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup
            await client.aclose()
            purged = await asyncio.to_thread(store.purge_all)
            logger.info("shutdown", extra={"sessions_purged": purged})

    app = FastAPI(
        title=settings.app_name,
        description=_DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.llm_client = client
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=600,
    )

    @app.exception_handler(RedactItError)
    async def handle_redactit_error(request: Request, exc: RedactItError) -> JSONResponse:
        logger.warning(
            "request_failed",
            extra={
                "code": exc.code,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
            headers=exc.headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _safe_validation_errors(exc)
        logger.warning(
            "request_validation_failed",
            extra={"path": request.url.path, "fields": [e["field"] for e in errors][:20]},
        )
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "VALIDATION_ERROR",
                "The request did not match the expected shape. Check the field names "
                "and types below.",
                {"errors": errors},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        detail = exc.detail if isinstance(exc.detail, str) else code
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The traceback is logged, never returned: it could quote document text.
        logger.exception(
            "unhandled_error", extra={"path": request.url.path, "error_type": type(exc).__name__}
        )
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "INTERNAL_ERROR",
                "RedactIt hit an unexpected error while handling this request. "
                "Nothing was modified. Check the server log for the cause and retry.",
                {"error_type": type(exc).__name__},
            ),
        )

    app.include_router(system_router)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
