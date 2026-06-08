"""Async token-bucket rate limiter for DeFi API endpoints.

Provides per-endpoint rate limiting using the token bucket algorithm,
which allows short bursts while enforcing a sustained request rate.
Designed for async/await use alongside protocol adapters.

Example::

    limiter = RateLimiter()
    limiter.configure("coingecko", requests_per_second=10, burst=20)

    async with limiter.acquire("coingecko"):
        response = await fetch("https://api.coingecko.com/...")

    print(limiter.stats("coingecko"))
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class LimitStrategy(str, Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    """Classic token bucket: tokens refill at a constant rate, requests
    consume tokens.  Allows bursts up to the bucket capacity."""

    SLIDING_WINDOW = "sliding_window"
    """Sliding window: tracks timestamps of recent requests and rejects
    when the count within the window exceeds the limit."""


@dataclass
class _BucketState:
    """Mutable state for a single token bucket."""

    tokens: float
    max_tokens: float
    refill_rate: float  # tokens per second
    last_refill: float
    total_acquired: int = 0
    total_waited_ms: float = 0.0
    total_throttled: int = 0

    def refill(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now


@dataclass
class _WindowState:
    """Mutable state for a sliding window limiter."""

    timestamps: list[float] = field(default_factory=list)
    max_requests: int = 0
    window_seconds: float = 60.0
    total_acquired: int = 0
    total_waited_ms: float = 0.0
    total_throttled: int = 0

    def prune(self, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]


@dataclass
class RateLimitStats:
    """Usage statistics for a single endpoint."""

    endpoint: str
    total_acquired: int
    total_waited_ms: float
    total_throttled: int
    current_tokens: Optional[float] = None
    max_tokens: Optional[float] = None
    current_window_count: Optional[int] = None
    max_requests: Optional[int] = None
    strategy: LimitStrategy = LimitStrategy.TOKEN_BUCKET

    @property
    def avg_wait_ms(self) -> float:
        """Average wait time per request in milliseconds."""
        if self.total_acquired == 0:
            return 0.0
        return self.total_waited_ms / self.total_acquired

    @property
    def throttle_rate(self) -> float:
        """Fraction of requests that were throttled (0-1)."""
        total = self.total_acquired + self.total_throttled
        if total == 0:
            return 0.0
        return self.total_throttled / total


class RateLimiter:
    """Async rate limiter with per-endpoint configuration.

    Supports two strategies:

    - **Token bucket** (default): Allows bursts, enforces sustained rate.
    - **Sliding window**: Strict request count per time window.

    Each endpoint is configured independently.  Unconfigured endpoints
    pass through without limiting (graceful degradation).

    Args:
        default_rps: Default requests-per-second for unconfigured endpoints.
            Set to 0 to disable default limiting (all endpoints pass through).
        default_burst: Default burst capacity.  If 0, defaults to 2x rps.
    """

    def __init__(
        self,
        default_rps: float = 0.0,
        default_burst: int = 0,
    ) -> None:
        self._default_rps = default_rps
        self._default_burst = default_burst if default_burst > 0 else int(default_rps * 2) if default_rps > 0 else 0
        self._buckets: dict[str, _BucketState] = {}
        self._windows: dict[str, _WindowState] = {}
        self._strategies: dict[str, LimitStrategy] = {}
        self._lock = asyncio.Lock()

    # ── Configuration ──────────────────────────────────────────────

    def configure(
        self,
        endpoint: str,
        requests_per_second: float = 10.0,
        burst: int = 0,
        strategy: LimitStrategy = LimitStrategy.TOKEN_BUCKET,
    ) -> None:
        """Configure rate limits for a specific endpoint.

        Args:
            endpoint: Endpoint name (e.g. ``"coingecko"``, ``"aave-api"``).
            requests_per_second: Sustained request rate.
            burst: Maximum burst size.  Defaults to 2x requests_per_second.
            strategy: Limiting strategy to use.

        Raises:
            ValueError: If requests_per_second is negative.
        """
        if requests_per_second < 0:
            raise ValueError(
                f"requests_per_second must be >= 0, got {requests_per_second}"
            )

        self._strategies[endpoint] = strategy

        if strategy is LimitStrategy.TOKEN_BUCKET:
            capacity = burst if burst > 0 else max(int(requests_per_second * 2), 1)
            self._buckets[endpoint] = _BucketState(
                tokens=float(capacity),
                max_tokens=float(capacity),
                refill_rate=requests_per_second,
                last_refill=time.monotonic(),
            )
            # Remove window state if switching strategies
            self._windows.pop(endpoint, None)
        else:  # SLIDING_WINDOW
            max_req = burst if burst > 0 else max(int(requests_per_second * 60), 1)
            self._windows[endpoint] = _WindowState(
                max_requests=max_req,
                window_seconds=60.0,
            )
            # Remove bucket state if switching strategies
            self._buckets.pop(endpoint, None)

    def configure_bulk(
        self, configs: dict[str, dict[str, object]]
    ) -> None:
        """Configure multiple endpoints at once.

        Args:
            configs: Mapping of endpoint name to configuration dict.
                Keys are passed to :meth:`configure` as keyword arguments.

        Example::

            limiter.configure_bulk({
                "coingecko": {"requests_per_second": 10, "burst": 20},
                "aave": {"requests_per_second": 5},
                "defillama": {"requests_per_second": 30, "burst": 50},
            })
        """
        for endpoint, cfg in configs.items():
            self.configure(endpoint, **cfg)  # type: ignore[arg-type]

    # ── Acquisition ────────────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self, endpoint: str) -> AsyncIterator[None]:
        """Acquire a request slot, waiting if necessary.

        Usage::

            async with limiter.acquire("coingecko"):
                # request is allowed
                response = await make_request()

        Args:
            endpoint: The endpoint to rate-limit against.

        Yields:
            Control after the rate limit is satisfied.
        """
        wait_ms = await self._acquire_internal(endpoint)
        # Track wait time outside the lock
        if endpoint in self._buckets:
            self._buckets[endpoint].total_waited_ms += wait_ms
            self._buckets[endpoint].total_acquired += 1
        elif endpoint in self._windows:
            self._windows[endpoint].total_waited_ms += wait_ms
            self._windows[endpoint].total_acquired += 1
        yield

    async def _acquire_internal(self, endpoint: str) -> float:
        """Core acquisition logic. Returns wait time in ms."""
        # Determine effective strategy
        strategy = self._strategies.get(endpoint)
        if strategy is None:
            if self._default_rps > 0:
                # Auto-configure with defaults
                self.configure(endpoint, self._default_rps, self._default_burst)
                strategy = self._strategies[endpoint]
            else:
                # No limiting configured — pass through
                return 0.0

        if strategy is LimitStrategy.TOKEN_BUCKET:
            return await self._acquire_token_bucket(endpoint)
        return await self._acquire_sliding_window(endpoint)

    async def _acquire_token_bucket(self, endpoint: str) -> float:
        """Token bucket acquisition with exponential backoff."""
        wait_ms = 0.0
        while True:
            async with self._lock:
                bucket = self._buckets[endpoint]
                now = time.monotonic()
                bucket.refill(now)

                if bucket.tokens >= 1.0:
                    bucket.tokens -= 1.0
                    return wait_ms

                # Calculate wait time for one token
                deficit = 1.0 - bucket.tokens
                wait_seconds = deficit / bucket.refill_rate if bucket.refill_rate > 0 else 1.0

            bucket.total_throttled += 1
            # Wait outside the lock
            await asyncio.sleep(wait_seconds)
            wait_ms += wait_seconds * 1000

    async def _acquire_sliding_window(self, endpoint: str) -> float:
        """Sliding window acquisition."""
        wait_ms = 0.0
        while True:
            async with self._lock:
                window = self._windows[endpoint]
                now = time.monotonic()
                window.prune(now)

                if len(window.timestamps) < window.max_requests:
                    window.timestamps.append(now)
                    return wait_ms

                # Wait until the oldest request falls out of the window
                oldest = window.timestamps[0]
                wait_seconds = (oldest + window.window_seconds) - now + 0.01

            window.total_throttled += 1
            await asyncio.sleep(max(wait_seconds, 0.01))
            wait_ms += max(wait_seconds, 0.01) * 1000

    # ── Non-blocking try ───────────────────────────────────────────

    async def try_acquire(self, endpoint: str) -> bool:
        """Try to acquire without waiting.

        Returns True if the request is allowed immediately, False if
        it would need to wait.

        Args:
            endpoint: The endpoint to check.

        Returns:
            Whether the request is allowed.
        """
        strategy = self._strategies.get(endpoint)
        if strategy is None:
            if self._default_rps > 0:
                self.configure(endpoint, self._default_rps, self._default_burst)
                strategy = self._strategies[endpoint]
            else:
                return True

        async with self._lock:
            if strategy is LimitStrategy.TOKEN_BUCKET:
                bucket = self._buckets[endpoint]
                now = time.monotonic()
                bucket.refill(now)
                if bucket.tokens >= 1.0:
                    bucket.tokens -= 1.0
                    bucket.total_acquired += 1
                    return True
                bucket.total_throttled += 1
                return False
            else:
                window = self._windows[endpoint]
                now = time.monotonic()
                window.prune(now)
                if len(window.timestamps) < window.max_requests:
                    window.timestamps.append(now)
                    window.total_acquired += 1
                    return True
                window.total_throttled += 1
                return False

    # ── Statistics ──────────────────────────────────────────────────

    def stats(self, endpoint: str) -> RateLimitStats:
        """Get usage statistics for an endpoint.

        Args:
            endpoint: Endpoint name.

        Returns:
            Current rate limit statistics.

        Raises:
            KeyError: If the endpoint has not been configured.
        """
        strategy = self._strategies.get(endpoint)
        if strategy is None:
            return RateLimitStats(
                endpoint=endpoint,
                total_acquired=0,
                total_waited_ms=0.0,
                total_throttled=0,
                strategy=LimitStrategy.TOKEN_BUCKET,
            )

        if strategy is LimitStrategy.TOKEN_BUCKET:
            bucket = self._buckets[endpoint]
            now = time.monotonic()
            bucket.refill(now)
            return RateLimitStats(
                endpoint=endpoint,
                total_acquired=bucket.total_acquired,
                total_waited_ms=bucket.total_waited_ms,
                total_throttled=bucket.total_throttled,
                current_tokens=round(bucket.tokens, 2),
                max_tokens=bucket.max_tokens,
                strategy=LimitStrategy.TOKEN_BUCKET,
            )
        else:
            window = self._windows[endpoint]
            now = time.monotonic()
            window.prune(now)
            return RateLimitStats(
                endpoint=endpoint,
                total_acquired=window.total_acquired,
                total_waited_ms=window.total_waited_ms,
                total_throttled=window.total_throttled,
                current_window_count=len(window.timestamps),
                max_requests=window.max_requests,
                strategy=LimitStrategy.SLIDING_WINDOW,
            )

    def all_stats(self) -> dict[str, RateLimitStats]:
        """Get statistics for all configured endpoints.

        Returns:
            Mapping of endpoint name to its statistics.
        """
        all_endpoints = set(self._strategies.keys())
        return {ep: self.stats(ep) for ep in sorted(all_endpoints)}

    # ── Management ──────────────────────────────────────────────────

    def reset(self, endpoint: str) -> None:
        """Reset all statistics for an endpoint (keeps configuration).

        Args:
            endpoint: Endpoint name to reset.
        """
        if endpoint in self._buckets:
            bucket = self._buckets[endpoint]
            bucket.tokens = bucket.max_tokens
            bucket.total_acquired = 0
            bucket.total_waited_ms = 0.0
            bucket.total_throttled = 0
            bucket.last_refill = time.monotonic()
        if endpoint in self._windows:
            window = self._windows[endpoint]
            window.timestamps.clear()
            window.total_acquired = 0
            window.total_waited_ms = 0.0
            window.total_throttled = 0

    def remove(self, endpoint: str) -> None:
        """Remove all configuration and state for an endpoint.

        Args:
            endpoint: Endpoint name to remove.
        """
        self._strategies.pop(endpoint, None)
        self._buckets.pop(endpoint, None)
        self._windows.pop(endpoint, None)

    @property
    def endpoints(self) -> list[str]:
        """List all configured endpoint names."""
        return sorted(self._strategies.keys())

    def __repr__(self) -> str:
        n = len(self._strategies)
        return f"RateLimiter(endpoints={n}, default_rps={self._default_rps})"
