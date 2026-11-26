"""Request ids and structured access logs.

Every request gets an id (a well-formed ``X-Request-ID`` from the client is kept, anything
else is replaced) that is returned in the response header, attached to every log record
emitted while the request runs, and included in error bodies. Access logs carry method,
route path, status and duration only: no bodies, query strings, file names or client
addresses.
"""

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
access_logger = logging.getLogger("baulens.api.access")


class JsonFormatter(logging.Formatter):
    """One JSON object per line with the request id when there is one."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        payload |= getattr(record, "fields", {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("baulens")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False


class RequestIdMiddleware:
    """Pure ASGI middleware, so the context variable is visible in the route's threads."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(REQUEST_ID_HEADER.lower().encode(), b"").decode()
        request_id = supplied if _VALID_ID.fullmatch(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            route = scope.get("route")
            access_logger.info(
                "request",
                extra={
                    "fields": {
                        "method": scope["method"],
                        "path": getattr(route, "path", scope["path"]),
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                },
            )
            request_id_var.reset(token)
