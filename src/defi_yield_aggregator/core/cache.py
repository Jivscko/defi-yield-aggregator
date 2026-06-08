"""Generic async TTL cache with LRU eviction and statistics."""

from __future__ import annotations

import asyncio
import functools
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CacheEntry:
    """A single cache entry holding a value, its TTL, and insertion time.

    Attributes:
        value: The cached value.
        ttl: Time-to-live in seconds.
        created_at: Timestamp when the entry was created.
    """

    __slots__ = ("value", "ttl", "created_at")

    def __init__(self: CacheEntry, value: Any, ttl: float) -> None:
        self.value = value
        self.ttl = ttl
        self.created_at = time.monotonic()

    @property
    def expired(self: CacheEntry) -> bool:
        """Check whether this entry has exceeded its TTL.

        Returns:
            True if the entry is expired, False otherwise.
        """
        return (time.monotonic() - self.created_at) >= self.ttl


class TTLCache:
    """An async-aware TTL cache with LRU eviction and hit/miss statistics.

    This cache supports async value producers. On a cache miss, the producer
    coroutine is awaited to compute the value, which is then cached for the
    configured TTL duration. When the cache reaches ``max_size``, the
    least-recently-used entry is evicted.

    Attributes:
        default_ttl: Default time-to-live in seconds for cache entries.
        max_size: Maximum number of entries the cache can hold.
        hit_count: Total number of cache hits.
        miss_count: Total number of cache misses.
    """

    def __init__(self: TTLCache, default_ttl: float = 300.0, max_size: int = 128) -> None:
        """Initialize the TTL cache.

        Args:
            default_ttl: Default TTL in seconds for cached entries.
            max_size: Maximum number of entries before LRU eviction occurs.
        """
        if default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {default_ttl}")
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.default_ttl: float = default_ttl
        self.max_size: int = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock: asyncio.Lock = asyncio.Lock()
        self.hit_count: int = 0
        self.miss_count: int = 0

    @property
    def hit_rate(self: TTLCache) -> float:
        """Calculate the cache hit rate.

        Returns:
            Ratio of hits to total lookups, or 0.0 if no lookups yet.
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    async def get_or_compute(
        self: TTLCache,
        key: str,
        producer: Callable[[], Awaitable[T]],
        ttl: float | None = None,
    ) -> T:
        """Retrieve a value from cache or compute it via the async producer.

        If the key exists in the cache and has not expired, the cached value is
        returned. Otherwise the ``producer`` coroutine is awaited to compute a
        fresh value, which is then stored.

        Args:
            key: The cache key.
            producer: An async callable that produces the value on a miss.
            ttl: Optional per-entry TTL override in seconds. Defaults to the
                cache's ``default_ttl``.

        Returns:
            The cached or freshly computed value.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl

        async with self._lock:
            entry = self._store.get(key)
            if entry is not None and not entry.expired:
                self.hit_count += 1
                self._store.move_to_end(key)
                return entry.value

            self.miss_count += 1

        # Compute outside the lock so other keys aren't blocked.
        value = await producer()

        async with self._lock:
            self._put(key, CacheEntry(value, effective_ttl))

        return value

    def _put(self: TTLCache, key: str, entry: CacheEntry) -> None:
        """Insert an entry, evicting LRU entries if the cache is full.

        Args:
            key: The cache key.
            entry: The cache entry to store.
        """
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    async def invalidate(self: TTLCache, key: str) -> bool:
        """Remove a single key from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if the key was present, False otherwise.
        """
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self: TTLCache) -> None:
        """Remove all entries from the cache and reset statistics."""
        async with self._lock:
            self._store.clear()
            self.hit_count = 0
            self.miss_count = 0

    def __contains__(self: TTLCache, key: object) -> bool:
        """Check whether a key is present and not expired.

        Args:
            key: The key to check.

        Returns:
            True if the key exists and has not expired.
        """
        if not isinstance(key, str):
            return False
        entry = self._store.get(key)
        return entry is not None and not entry.expired

    def __len__(self: TTLCache) -> int:
        """Return the number of entries currently in the cache.

        Returns:
            The number of stored entries (including possibly expired ones).
        """
        return len(self._store)

    def __repr__(self: TTLCache) -> str:
        """Return a developer-friendly representation.

        Returns:
            A string showing cache configuration and statistics.
        """
        return (
            f"TTLCache(default_ttl={self.default_ttl}, max_size={self.max_size}, "
            f"size={len(self._store)}, hit_count={self.hit_count}, "
            f"miss_count={self.miss_count}, hit_rate={self.hit_rate:.2%})"
        )


def cached(
    ttl: float = 300.0,
    max_size: int = 128,
    key_func: Callable[..., str] | None = None,
) -> Callable[..., Any]:
    """Decorator that caches the return value of an async method.

    The decorated method's instance is used to maintain a per-instance cache
    stored in the ``__cache_<method_name>`` attribute.

    Args:
        ttl: Time-to-live in seconds for each cached result.
        max_size: Maximum number of cached results per instance.
        key_func: Optional callable that produces a cache key from the method
            arguments. If ``None``, a default key is built from ``*args`` and
            ``**kwargs``.

    Returns:
        A decorator wrapping the async method with caching logic.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cache_attr = f"__cache_{func.__qualname__}"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not args:
                raise RuntimeError(
                    "@cached decorator requires at least one positional argument "
                    "(the method 'self' parameter)."
                )
            self_obj = args[0]
            cache: TTLCache = getattr(self_obj, cache_attr, None)
            if cache is None:
                cache = TTLCache(default_ttl=ttl, max_size=max_size)
                setattr(self_obj, cache_attr, cache)

            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                parts = [repr(a) for a in args[1:]]
                parts.extend(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
                cache_key = f"{func.__qualname__}({', '.join(parts)})"

            async def producer() -> T:
                return await func(*args, **kwargs)

            return await cache.get_or_compute(cache_key, producer, ttl=ttl)

        return wrapper  # type: ignore[return-value]

    return decorator
