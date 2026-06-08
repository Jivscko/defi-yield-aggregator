"""Custom protocol adapter example.

Shows how to create a new adapter by subclassing ``BaseAdapter``.
Each adapter must implement ``fetch_pools`` and ``fetch_pool_detail``.

Run:
    python examples/custom_adapter.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from defi_yield_aggregator.adapters.base import BaseAdapter
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


# ----------------------------------------------------------------
# Step 1: Define a new Protocol enum value (or reuse an existing one)
#         For a real integration you'd extend the Protocol enum in
#         defi_yield_aggregator.core.models.  For this example we
#         reuse an existing value so we don't need to modify the
#         package source.
# ----------------------------------------------------------------


class CustomLendingAdapter(BaseAdapter):
    """Example adapter for a hypothetical lending protocol.

    In a real implementation, ``fetch_pools`` would call an HTTP API
    (e.g. with ``httpx``) and parse the JSON response into ``PoolInfo``
    objects.

    This example uses hardcoded mock data for illustration.
    """

    def __init__(self, base_url: str = "https://api.custom-lend.example.com") -> None:
        super().__init__(Protocol.AAVE, base_url)  # Reuse AAVE for demo

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch pools from the custom lending protocol.

        Args:
            chain: Optional chain filter. If ``None``, returns pools on all chains.

        Returns:
            List of available lending pools.
        """
        # In production, replace with actual API call:
        #   async with httpx.AsyncClient() as client:
        #       resp = await client.get(f"{self.base_url}/v1/pools")
        #       data = resp.json()
        #       return [PoolInfo(**item) for item in data]

        mock_pools = [
            PoolInfo(
                protocol=Protocol.AAVE,  # Replace with custom Protocol enum
                chain=Chain.ETHEREUM,
                pool_id="custom-lend-dai",
                pool_name="CustomLend DAI",
                token_pair="DAI",
                apy=0.042,
                tvl_usd=350_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ARBITRUM,
                pool_id="custom-lend-usdc-arb",
                pool_name="CustomLend USDC (Arbitrum)",
                token_pair="USDC",
                apy=0.048,
                tvl_usd=120_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
        ]

        if chain is not None:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch a single pool by its ID.

        Args:
            pool_id: Unique pool identifier (e.g. ``"custom-lend-dai"``).

        Returns:
            Pool info if found, ``None`` otherwise.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


# ----------------------------------------------------------------
# Step 2: Use the adapter alongside the built-in ones
# ----------------------------------------------------------------


async def main() -> None:
    from defi_yield_aggregator.adapters.protocols import get_all_adapters
    from defi_yield_aggregator.core.risk_engine import RiskEngine

    # Instantiate built-in adapters + our custom one
    adapters = [*get_all_adapters(), CustomLendingAdapter()]

    all_pools: list[PoolInfo] = []
    for adapter in adapters:
        pools = await adapter.fetch_pools()
        all_pools.extend(pools)
        print(f"  {adapter!r}: {len(pools)} pools")

    # Score everything with the risk engine
    engine = RiskEngine()
    scores = engine.score_pools(all_pools)

    # Find our custom pool's score
    for score in scores:
        if score.pool_id.startswith("custom-lend"):
            print(f"\n  Custom pool '{score.pool_id}':")
            print(f"    Risk level:   {score.risk_level.value}")
            print(f"    Risk score:   {score.overall_score:.1f}")
            print(f"    TVL score:    {score.tvl_score:.0f}")
            print(f"    Audit score:  {score.audit_score:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
