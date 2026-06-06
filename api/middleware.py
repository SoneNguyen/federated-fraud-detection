"""Request logging and rate-limiting middleware for the FastAPI inference gateway."""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.access")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-process per-IP rate limiter.

    Allows up to `max_requests` requests per `window_seconds` per client IP.
    Returns HTTP 429 when the limit is exceeded.
    Does not persist across restarts — suitable for prototype use only.
    """

    def __init__(self, app, max_requests: int = 1000, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {ip: [timestamp, ...]}
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        # Prune old timestamps
        self._windows[ip] = [t for t in self._windows[ip] if t > window_start]

        if len(self._windows[ip]) >= self.max_requests:
            return Response(
                content='{"detail":"Too many requests"}',
                status_code=429,
                media_type="application/json",
            )

        self._windows[ip].append(now)
        return await call_next(request)