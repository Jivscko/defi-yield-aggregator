"""Tests for the async token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from defi_yield_aggregator.core.rate_limiter import (
    LimitStrategy,
    RateLimiter,
    RateLimitStats,
)


class TestRateLimiterConfiguration:
    """Configuration and setup tests."""

    def test_default_init(self) -> None:
        """Default limiter has no configured endpoints."""
        limiter = RateLimiter()
        assert limiter.endpoints == []
        assert limiter._default_rps == 0.0

    def test_init_with_defaults(self) -> None:
        """Limiter with defaults auto-configures on first acquire."""
        limiter = RateLimiter(default_rps=10, default_burst=20)
        assert limiter._default_rps == 10
        assert limiter._default_burst == 20

    def test_configure_token_bucket(self) -> None:
        """Configure a token bucket endpoint."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=20)
        assert "api" in limiter.endpoints
        stats = limiter.stats("api")
        assert stats.strategy == LimitStrategy.TOKEN_BUCKET
        assert stats.max_tokens == 20.0
        assert stats.current_tokens == 20.0

    def test_configure_sliding_window(self) -> None:
        """Configure a sliding window endpoint."""
        limiter = RateLimiter()
        limiter.configure(
            "api",
            requests_per_second=10,
            burst=100,
            strategy=LimitStrategy.SLIDING_WINDOW,
        )
        stats = limiter.stats("api")
        assert stats.strategy == LimitStrategy.SLIDING_WINDOW
        assert stats.max_requests == 100

    def test_configure_default_burst(self) -> None:
        """Burst defaults to 2x requests_per_second."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10)
        stats = limiter.stats("api")
        assert stats.max_tokens == 20.0

    def test_configure_bulk(self) -> None:
        """Bulk configuration sets up multiple endpoints."""
        limiter = RateLimiter()
        limiter.configure_bulk({
            "coingecko": {"requests_per_second": 10, "burst": 20},
            "aave": {"requests_per_second": 5},
            "defillama": {"requests_per_second": 30, "burst": 50},
        })
        assert len(limiter.endpoints) == 3
        assert "coingecko" in limiter.endpoints
        assert "aave" in limiter.endpoints
        assert "defillama" in limiter.endpoints

    def test_configure_negative_rps_raises(self) -> None:
        """Negative RPS raises ValueError."""
        limiter = RateLimiter()
        with pytest.raises(ValueError, match="requests_per_second must be >= 0"):
            limiter.configure("api", requests_per_second=-1)

    def test_configure_zero_rps(self) -> None:
        """Zero RPS creates a bucket that blocks everything."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=0, burst=1)
        stats = limiter.stats("api")
        assert stats.max_tokens == 1.0

    def test_repr(self) -> None:
        """Repr shows endpoint count and default rps."""
        limiter = RateLimiter(default_rps=10)
        limiter.configure("api", requests_per_second=5)
        r = repr(limiter)
        assert "RateLimiter" in r
        assert "endpoints=1" in r


class TestTokenBucket:
    """Token bucket strategy tests."""

    @pytest.mark.asyncio
    async def test_acquire_within_burst(self) -> None:
        """Can acquire up to burst count without waiting."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=5)

        for _ in range(5):
            async with limiter.acquire("api"):
                pass  # Should not block

        stats = limiter.stats("api")
        assert stats.total_acquired == 5
        assert stats.total_throttled == 0

    @pytest.mark.asyncio
    async def test_acquire_refills_tokens(self) -> None:
        """Tokens refill over time."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=100, burst=2)

        # Exhaust tokens
        async with limiter.acquire("api"):
            pass
        async with limiter.acquire("api"):
            pass

        # Wait for refill (100 rps -> 1 token per 10ms)
        await asyncio.sleep(0.05)

        async with limiter.acquire("api"):
            pass  # Should succeed after refill

        stats = limiter.stats("api")
        assert stats.total_acquired == 3

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_exhausted(self) -> None:
        """Requests wait when tokens are exhausted."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=50, burst=1)

        # Exhaust the single token
        async with limiter.acquire("api"):
            pass

        # Next acquire should wait briefly then succeed
        start = time.monotonic()
        async with limiter.acquire("api"):
            elapsed = time.monotonic() - start

        # Should have waited roughly 1/50 = 0.02 seconds
        assert elapsed >= 0.01  # At least some wait

    @pytest.mark.asyncio
    async def test_try_acquire_within_budget(self) -> None:
        """try_acquire succeeds when tokens available."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=3)

        assert await limiter.try_acquire("api") is True
        assert await limiter.try_acquire("api") is True
        assert await limiter.try_acquire("api") is True

    @pytest.mark.asyncio
    async def test_try_acquire_exhausted(self) -> None:
        """try_acquire fails when no tokens available."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=0.1, burst=1)

        assert await limiter.try_acquire("api") is True
        assert await limiter.try_acquire("api") is False

    @pytest.mark.asyncio
    async def test_token_bucket_tracks_throttling(self) -> None:
        """Throttled requests are counted."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=0.1, burst=1)

        # Use the only token
        assert await limiter.try_acquire("api") is True
        # This should be throttled
        assert await limiter.try_acquire("api") is False

        stats = limiter.stats("api")
        assert stats.total_throttled == 1


class TestSlidingWindow:
    """Sliding window strategy tests."""

    @pytest.mark.asyncio
    async def test_acquire_within_window(self) -> None:
        """Can acquire up to max_requests without waiting."""
        limiter = RateLimiter()
        limiter.configure(
            "api",
            requests_per_second=10,
            burst=5,
            strategy=LimitStrategy.SLIDING_WINDOW,
        )

        for _ in range(5):
            async with limiter.acquire("api"):
                pass

        stats = limiter.stats("api")
        assert stats.total_acquired == 5

    @pytest.mark.asyncio
    async def test_sliding_window_try_acquire(self) -> None:
        """try_acquire works with sliding window."""
        limiter = RateLimiter()
        limiter.configure(
            "api",
            requests_per_second=10,
            burst=2,
            strategy=LimitStrategy.SLIDING_WINDOW,
        )

        assert await limiter.try_acquire("api") is True
        assert await limiter.try_acquire("api") is True
        assert await limiter.try_acquire("api") is False


class TestDefaultEndpoints:
    """Tests for auto-configured endpoints."""

    @pytest.mark.asyncio
    async def test_unconfigured_endpoint_passes_through(self) -> None:
        """Endpoints without config and no defaults pass through."""
        limiter = RateLimiter()
        async with limiter.acquire("unknown"):
            pass  # Should not block

    @pytest.mark.asyncio
    async def test_default_rps_auto_configures(self) -> None:
        """Default RPS auto-configures on first use."""
        limiter = RateLimiter(default_rps=10, default_burst=5)
        async with limiter.acquire("new-api"):
            pass

        assert "new-api" in limiter.endpoints
        stats = limiter.stats("new-api")
        assert stats.total_acquired == 1


class TestStatistics:
    """Statistics and monitoring tests."""

    def test_stats_unconfigured_endpoint(self) -> None:
        """Stats for unconfigured endpoint return zeros."""
        limiter = RateLimiter()
        stats = limiter.stats("nonexistent")
        assert stats.total_acquired == 0
        assert stats.total_waited_ms == 0.0
        assert stats.total_throttled == 0

    @pytest.mark.asyncio
    async def test_all_stats(self) -> None:
        """all_stats returns stats for all configured endpoints."""
        limiter = RateLimiter()
        limiter.configure("api1", requests_per_second=10, burst=5)
        limiter.configure("api2", requests_per_second=20, burst=10)

        async with limiter.acquire("api1"):
            pass
        async with limiter.acquire("api2"):
            pass

        all_s = limiter.all_stats()
        assert len(all_s) == 2
        assert all_s["api1"].total_acquired == 1
        assert all_s["api2"].total_acquired == 1

    @pytest.mark.asyncio
    async def test_avg_wait_ms(self) -> None:
        """avg_wait_ms computes correctly."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=100, burst=1)

        # First request: no wait
        async with limiter.acquire("api"):
            pass

        stats = limiter.stats("api")
        assert stats.avg_wait_ms == 0.0

    def test_throttle_rate_zero_when_no_throttling(self) -> None:
        """Throttle rate is 0 when no throttling occurred."""
        stats = RateLimitStats(
            endpoint="test",
            total_acquired=10,
            total_waited_ms=0.0,
            total_throttled=0,
        )
        assert stats.throttle_rate == 0.0

    def test_throttle_rate_calculation(self) -> None:
        """Throttle rate calculates correctly."""
        stats = RateLimitStats(
            endpoint="test",
            total_acquired=8,
            total_waited_ms=0.0,
            total_throttled=2,
        )
        assert stats.throttle_rate == pytest.approx(0.2)


class TestManagement:
    """Endpoint management tests."""

    def test_reset_clears_stats(self) -> None:
        """Reset clears statistics but keeps configuration."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=5)

        # Simulate some usage (manually adjust internals)
        bucket = limiter._buckets["api"]
        bucket.total_acquired = 10
        bucket.total_waited_ms = 50.0
        bucket.total_throttled = 3
        bucket.tokens = 2.0

        limiter.reset("api")

        stats = limiter.stats("api")
        assert stats.total_acquired == 0
        assert stats.total_waited_ms == 0.0
        assert stats.total_throttled == 0
        assert stats.current_tokens == 5.0  # Refilled to max

    def test_remove_endpoint(self) -> None:
        """Remove deletes all state for an endpoint."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=5)
        assert "api" in limiter.endpoints

        limiter.remove("api")
        assert "api" not in limiter.endpoints

    def test_remove_nonexistent_is_noop(self) -> None:
        """Removing a nonexistent endpoint doesn't raise."""
        limiter = RateLimiter()
        limiter.remove("nonexistent")  # Should not raise


class TestRateLimitStatsModel:
    """RateLimitStats dataclass tests."""

    def test_avg_wait_ms_zero_when_no_requests(self) -> None:
        """avg_wait_ms is 0 when no requests have been made."""
        stats = RateLimitStats(
            endpoint="test",
            total_acquired=0,
            total_waited_ms=0.0,
            total_throttled=0,
        )
        assert stats.avg_wait_ms == 0.0

    def test_avg_wait_ms_nonzero(self) -> None:
        """avg_wait_ms computes average correctly."""
        stats = RateLimitStats(
            endpoint="test",
            total_acquired=4,
            total_waited_ms=200.0,
            total_throttled=0,
        )
        assert stats.avg_wait_ms == 50.0

    def test_throttle_rate_edge_case(self) -> None:
        """throttle_rate returns 0 when both counters are 0."""
        stats = RateLimitStats(
            endpoint="test",
            total_acquired=0,
            total_waited_ms=0.0,
            total_throttled=0,
        )
        assert stats.throttle_rate == 0.0


class TestStrategySwitching:
    """Tests for switching strategies on the same endpoint."""

    def test_switch_token_to_sliding(self) -> None:
        """Switching from token bucket to sliding window cleans up."""
        limiter = RateLimiter()
        limiter.configure("api", requests_per_second=10, burst=5)
        assert "api" in limiter._buckets

        limiter.configure(
            "api",
            requests_per_second=10,
            burst=50,
            strategy=LimitStrategy.SLIDING_WINDOW,
        )
        assert "api" not in limiter._buckets
        assert "api" in limiter._windows

    def test_switch_sliding_to_token(self) -> None:
        """Switching from sliding window to token bucket cleans up."""
        limiter = RateLimiter()
        limiter.configure(
            "api",
            requests_per_second=10,
            burst=50,
            strategy=LimitStrategy.SLIDING_WINDOW,
        )
        assert "api" in limiter._windows

        limiter.configure("api", requests_per_second=10, burst=5)
        assert "api" not in limiter._windows
        assert "api" in limiter._buckets
