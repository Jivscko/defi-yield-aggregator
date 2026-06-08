"""Basic usage example for the DeFi Yield Aggregator library.

Demonstrates:
    1. Fetching pool data from protocol adapters.
    2. Scoring pools with the risk engine.
    3. Optimizing a portfolio allocation.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio

from defi_yield_aggregator.adapters.protocols import get_all_adapters
from defi_yield_aggregator.core.models import (
    Chain,
    Config,
    PoolInfo,
    Protocol,
)
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer
from defi_yield_aggregator.core.risk_engine import RiskEngine
from defi_yield_aggregator.core.apy_calculator import future_value, daily_yield


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Fetch pools from all protocol adapters
    # ------------------------------------------------------------------
    adapters = get_all_adapters()
    all_pools: list[PoolInfo] = []
    for adapter in adapters:
        pools = await adapter.fetch_pools()
        all_pools.extend(pools)
        print(f"  Fetched {len(pools)} pools from {adapter.protocol.value}")

    print(f"\nTotal pools: {len(all_pools)}\n")

    # ------------------------------------------------------------------
    # 2. Score risk for every pool
    # ------------------------------------------------------------------
    engine = RiskEngine()
    scores = engine.score_pools(all_pools)

    print("Risk Scores (sorted by risk, lowest first):")
    print(f"  {'Pool':<40} {'APY':>8} {'TVL ($B)':>10} {'Risk':>6} {'Level':<10}")
    print("  " + "-" * 78)

    for score in sorted(scores, key=lambda s: s.overall_score):
        pool = next(p for p in all_pools if p.pool_id == score.pool_id)
        tvl_b = pool.tvl_usd / 1e9
        print(
            f"  {pool.pool_name:<40} {pool.apy * 100:>7.2f}% "
            f"{tvl_b:>9.2f} {score.overall_score:>6.1f} "
            f"{score.risk_level.value:<10}"
        )

    # ------------------------------------------------------------------
    # 3. Optimize a $50,000 portfolio
    # ------------------------------------------------------------------
    config = Config(
        max_risk_score=50,
        min_tvl_usd=500_000_000,
        max_single_allocation_pct=0.25,
    )

    optimizer = PortfolioOptimizer(config=config)
    portfolio = optimizer.optimize(all_pools, investment_usd=50_000)

    print(f"\n{'=' * 60}")
    print(f"Optimized Portfolio for $50,000")
    print(f"{'=' * 60}")
    print(f"  Expected APY:     {portfolio.total_expected_apy * 100:.2f}%")
    print(f"  Weighted Risk:    {portfolio.weighted_risk_score:.1f}")
    print(f"  Positions:        {portfolio.num_positions}")
    print(f"  Diversification:  {portfolio.diversification_score:.1f}/100\n")

    print("  Allocations:")
    print(f"  {'Protocol':<12} {'Chain':<10} {'Pool':<25} {'Alloc':>8} {'Amount':>12} {'APY':>8}")
    print("  " + "-" * 78)
    for alloc in portfolio.allocations:
        print(
            f"  {alloc.protocol.value:<12} {alloc.chain.value:<10} "
            f"{alloc.token_pair:<25} {alloc.allocation_pct * 100:>7.1f}% "
            f"${alloc.amount_usd:>10,.2f} {alloc.expected_apy * 100:>7.2f}%"
        )

    # ------------------------------------------------------------------
    # 4. Yield projections
    # ------------------------------------------------------------------
    daily = daily_yield(50_000, portfolio.total_expected_apy)
    fv_1y = future_value(50_000, portfolio.total_expected_apy, 1.0)
    print(f"\n  Projections:")
    print(f"    Daily earnings:  ${daily:,.2f}")
    print(f"    After 1 year:    ${fv_1y:,.2f}  (+${fv_1y - 50_000:,.2f})")


if __name__ == "__main__":
    asyncio.run(main())
