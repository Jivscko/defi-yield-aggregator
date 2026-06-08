"""Tests for async TTL cache."""

from __future__ import annotations

import asyncio

import pytest

from defi_yield_aggregator.core.cache import TTLCache, cached


# ---------------------------------------------------------------------------
# Helper class used to test the @cached decorator
# ---------------------------------------------------------------------------

class _DemoService:
    """A minimal service whose methods use the @cached decorator."""

    def __init__(self: _DemoService) -> None:
        self.call_count: int = 0

    @cached(ttl=1.0, max_size=4)
    async def compute(self: _DemoService, x: int) -> int:
        """Compute a value, tracking how many times it was actually called."""
        self.call_count += 1
        return x * 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetOrCompute:
    """Test the core get_or_compute method."""

    @pytest.mark.asyncio
    async def test_basic_caching(self: TestGetOrCompute) -> None:
        """Second call with the same key returns cached value; producer called once."""
        cache: TTLCache = TTLCache(default_ttl=10.0)
        call_count = 0

        async def producer() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        val1 = await cache.get_or_compute("k", producer)
        val2 = await cache.get_or_compute("k", producer)

        assert val1 == 42
        assert val2 == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys(self: TestGetOrCompute) -> None:
        """Different keys produce independent values."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def p1() -> str:
            return "a"

        async def p2() -> str:
            return "b"

        assert await cache.get_or_compute("a", p1) == "a"
        assert await cache.get_or_compute("b", p2) == "b"


class TestTTLExpiry:
    """Test that entries expire after their TTL."""

    @pytest.mark.asyncio
    async def test_entry_expires(self: TestTTLExpiry) -> None:
        """Expired entries trigger re-computation."""
        cache: TTLCache = TTLCache(default_ttl=10.0)
        call_count = 0

        async def producer() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        await cache.get_or_compute("k", producer, ttl=0.1)
        assert call_count == 1

        await asyncio.sleep(0.15)

        val = await cache.get_or_compute("k", producer, ttl=0.1)
        assert call_count == 2
        assert val == 2


class TestMaxSizeLRU:
    """Test LRU eviction when the cache is full."""

    @pytest.mark.asyncio
    async def test_eviction(self: TestMaxSizeLRU) -> None:
        """Least-recently-used entry is evicted when max_size is reached."""
        cache: TTLCache = TTLCache(default_ttl=10.0, max_size=2)

        async def make_producer(v: int):
            async def producer() -> int:
                return v
            return producer

        await cache.get_or_compute("a", await make_producer(1))
        await cache.get_or_compute("b", await make_producer(2))

        # Touch "a" so it becomes most recently used
        await cache.get_or_compute("a", await make_producer(1))

        # Insert "c" -> should evict "b" (LRU)
        await cache.get_or_compute("c", await make_producer(3))

        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    @pytest.mark.asyncio
    async def test_eviction_count(self: TestMaxSizeLRU) -> None:
        """Cache never exceeds max_size."""
        cache: TTLCache = TTLCache(default_ttl=10.0, max_size=3)

        for i in range(10):
            idx = i

            async def producer(idx: int = idx) -> int:
                return idx

            await cache.get_or_compute(f"key_{i}", producer)

        assert len(cache) <= 3


class TestInvalidation:
    """Test manual invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_single(self: TestInvalidation) -> None:
        """Invalidating a key removes it from the cache."""
        cache: TTLCache = TTLCache(default_ttl=10.0)
        call_count = 0

        async def producer() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        await cache.get_or_compute("k", producer)
        assert await cache.invalidate("k") is True
        assert await cache.invalidate("k") is False  # already gone

        val = await cache.get_or_compute("k", producer)
        assert val == 2  # producer ran again
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_clear_all(self: TestInvalidation) -> None:
        """clear() removes all entries and resets statistics."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def producer() -> int:
            return 1

        await cache.get_or_compute("a", producer)
        await cache.get_or_compute("b", producer)
        assert len(cache) == 2

        await cache.clear()
        assert len(cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0


class TestStatistics:
    """Test cache hit/miss statistics."""

    @pytest.mark.asyncio
    async def test_hit_miss_counts(self: TestStatistics) -> None:
        """hit_count and miss_count are tracked correctly."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def producer() -> int:
            return 1

        await cache.get_or_compute("a", producer)  # miss
        await cache.get_or_compute("a", producer)  # hit
        await cache.get_or_compute("b", producer)  # miss
        await cache.get_or_compute("b", producer)  # hit
        await cache.get_or_compute("b", producer)  # hit

        assert cache.miss_count == 2
        assert cache.hit_count == 3

    @pytest.mark.asyncio
    async def test_hit_rate(self: TestStatistics) -> None:
        """hit_rate returns the correct ratio."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        assert cache.hit_rate == 0.0  # no lookups yet

        async def producer() -> int:
            return 1

        await cache.get_or_compute("a", producer)  # miss
        await cache.get_or_compute("a", producer)  # hit

        assert cache.hit_rate == pytest.approx(0.5)


class TestCachedDecorator:
    """Test the @cached decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches(self: TestCachedDecorator) -> None:
        """Decorator caches return values and avoids redundant calls."""
        svc = _DemoService()

        r1 = await svc.compute(5)
        r2 = await svc.compute(5)

        assert r1 == 10
        assert r2 == 10
        assert svc.call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_different_args(self: TestCachedDecorator) -> None:
        """Different arguments produce different cache entries."""
        svc = _DemoService()

        assert await svc.compute(3) == 6
        assert await svc.compute(7) == 14
        assert svc.call_count == 2


class TestErrorHandling:
    """Test that exceptions from producers are not cached."""

    @pytest.mark.asyncio
    async def test_producer_exception_not_cached(self: TestErrorHandling) -> None:
        """If the producer raises, the error should propagate and the key should not be cached."""
        cache: TTLCache = TTLCache(default_ttl=10.0)
        call_count = 0

        async def bad_producer() -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await cache.get_or_compute("k", bad_producer)

        assert call_count == 1

        # Subsequent call should retry (key was not cached on error)
        async def good_producer() -> int:
            nonlocal call_count
            call_count += 1
            return 99

        val = await cache.get_or_compute("k", good_producer)
        assert val == 99
        assert call_count == 2


class TestConcurrentAccess:
    """Test concurrent access via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_concurrent_same_key(self: TestConcurrentAccess) -> None:
        """Multiple concurrent requests for the same key all get the correct value."""
        cache: TTLCache = TTLCache(default_ttl=10.0)
        call_count = 0

        async def slow_producer() -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return 42

        # First call populates the cache; concurrent ones after it completes
        # will hit the cache.
        await cache.get_or_compute("k", slow_producer)
        results = await asyncio.gather(
            *[cache.get_or_compute("k", slow_producer) for _ in range(10)]
        )

        assert all(r == 42 for r in results)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_different_keys(self: TestConcurrentAccess) -> None:
        """Concurrent requests for distinct keys all resolve correctly."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def make_producer(key: str):
            async def producer() -> str:
                await asyncio.sleep(0.01)
                return f"val_{key}"
            return producer

        keys = [f"key_{i}" for i in range(20)]
        producers = [await make_producer(k) for k in keys]
        results = await asyncio.gather(
            *[cache.get_or_compute(k, p) for k, p in zip(keys, producers)]
        )

        for key, result in zip(keys, results):
            assert result == f"val_{key}"


class TestDunderMethods:
    """Test __contains__, __len__, and __repr__."""

    @pytest.mark.asyncio
    async def test_contains(self: TestDunderMethods) -> None:
        """__contains__ reflects key presence and expiry."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def producer() -> int:
            return 1

        assert "k" not in cache
        await cache.get_or_compute("k", producer)
        assert "k" in cache

    @pytest.mark.asyncio
    async def test_contains_expired(self: TestDunderMethods) -> None:
        """__contains__ returns False for expired entries."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def producer() -> int:
            return 1

        await cache.get_or_compute("k", producer, ttl=0.05)
        await asyncio.sleep(0.1)
        assert "k" not in cache

    @pytest.mark.asyncio
    async def test_len(self: TestDunderMethods) -> None:
        """__len__ returns the number of stored entries."""
        cache: TTLCache = TTLCache(default_ttl=10.0)

        async def producer() -> int:
            return 1

        assert len(cache) == 0
        await cache.get_or_compute("a", producer)
        assert len(cache) == 1
        await cache.get_or_compute("b", producer)
        assert len(cache) == 2

    @pytest.mark.asyncio
    async def test_repr(self: TestDunderMethods) -> None:
        """__repr__ includes configuration and stats info."""
        cache: TTLCache = TTLCache(default_ttl=60.0, max_size=10)

        async def producer() -> int:
            return 1

        r = repr(cache)
        assert "TTLCache" in r
        assert "60" in r
        assert "10" in r

        await cache.get_or_compute("k", producer)
        r = repr(cache)
        assert "miss_count=1" in r
