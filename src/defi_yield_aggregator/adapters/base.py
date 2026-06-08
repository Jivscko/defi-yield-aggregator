"""Base protocol adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


class BaseAdapter(ABC):
    """Abstract base class for protocol adapters."""

    def __init__(self, protocol: Protocol, base_url: str = "") -> None:
        self.protocol = protocol
        self.base_url = base_url

    @abstractmethod
    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch available pools from the protocol.

        Args:
            chain: Optional filter by chain.

        Returns:
            List of available pools.
        """
        ...

    @abstractmethod
    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detailed information for a specific pool.

        Args:
            pool_id: Unique pool identifier.

        Returns:
            Pool info or None if not found.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(protocol={self.protocol.value})"
