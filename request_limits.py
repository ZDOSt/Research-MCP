"""ASGI middleware for enforcing request-body limits while streaming."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import Response


Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


class RequestBodyLimitMiddleware:
    """Reject declared and chunked bodies that exceed a fixed byte ceiling."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        declared_values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        if len(declared_values) > 1:
            await Response(status_code=400)(scope, receive, send)
            return
        if declared_values:
            try:
                declared = int(declared_values[0].decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                await Response(status_code=400)(scope, receive, send)
                return
            if declared < 0 or declared > self.max_bytes:
                await Response(status_code=413)(scope, receive, send)
                return

        messages: deque[dict[str, Any]] = deque()
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                await Response(status_code=413)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> dict[str, Any]:
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay, send)
