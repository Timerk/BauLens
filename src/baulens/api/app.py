"""FastAPI application factory: models are loaded once in the lifespan, not per request."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from baulens.api.models import API_PREFIX
from baulens.api.observability import RequestIdMiddleware, request_id_var
from baulens.api.report_service import ReportService, build_report_service
from baulens.api.routes import router
from baulens.api.services import load_cracks, load_strength
from baulens.api.state import AppState
from baulens.config import ApiConfig, Settings

logger = logging.getLogger(__name__)

DESCRIPTION = """
Decision support for concrete inspection: compressive-strength prediction with a conformal
interval, crack classification of surface images with Grad-CAM, and German report drafts
whose numbers are checked against the inputs.

**Not a substitute** for compressive-strength testing to EN 12390-3 or for an inspection
by a qualified engineer.
"""


def create_app(
    config: ApiConfig,
    settings: Settings | None = None,
    *,
    report_service: ReportService | None = None,
) -> FastAPI:
    """Build the app; ``report_service`` replaces the default adapter (tests, wiring)."""
    resolved = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        strength = await run_in_threadpool(load_strength, config)
        cracks = await run_in_threadpool(load_cracks, config)
        report = report_service or build_report_service(config, resolved)
        app.state.baulens = AppState(config, resolved, strength, cracks, report)
        for name, component in (("strength", strength), ("cracks", cracks), ("report", report)):
            info = component.status()
            logger.info("component %s ready=%s (%s)", name, info.ready, info.detail)
        yield

    app = FastAPI(
        title="BauLens API",
        version="1",
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.include_router(router, prefix=API_PREFIX)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)
    _register_error_handlers(app)
    return app


def _error(status_code: int, detail: object) -> JSONResponse:
    return JSONResponse(
        {"detail": detail, "request_id": request_id_var.get()}, status_code=status_code
    )


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error(error.status_code, error.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        # Drop the echoed input: it may contain uploaded content or free text.
        details = [
            {"loc": item.get("loc"), "msg": item.get("msg"), "type": item.get("type")}
            for item in error.errors()
        ]
        return _error(422, details)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error(500, "internal error")
