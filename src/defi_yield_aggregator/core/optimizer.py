"""Portfolio optimizer - maximize yield given risk constraints."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from defi_yield_aggregator.core.models import (
    Config,
    OptimizedPortfolio,
    PoolInfo,
    PortfolioAllocation,
    RiskScore,
)
from defi_yield_aggregator.core.risk_engine import RiskEngine


def _greedy_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
) -> OptimizedPortfolio:
    """Greedy allocation: sort by risk-adjusted yield and allocate.

    Strategy:
    1. Score each pool by yield / risk ratio
    2. Allocate up to max_single_allocation_pct per pool
    3. Stop when budget exhausted or max_positions reached
    """
    # Sort by yield-to-risk ratio (higher = better)
    scored = []
    for pool, risk in candidates:
        ratio = pool.apy / max(risk.overall_score, 1.0)
        scored.append((pool, risk, ratio))

    scored.sort(key=lambda x: x[2], reverse=True)

    allocations: list[PortfolioAllocation] = []
    remaining_pct = 1.0

    for pool, risk, ratio in scored:
        if len(allocations) >= config.max_positions:
            break
        if remaining_pct <= 0.001:
            break

        alloc_pct = min(config.max_single_allocation_pct, remaining_pct)
        amount = investment_usd * alloc_pct

        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(alloc_pct, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )
        remaining_pct -= alloc_pct

    if not allocations:
        return OptimizedPortfolio(
            allocations=[],
            total_expected_apy=0.0,
            weighted_risk_score=0.0,
            total_investment_usd=0.0,
        )

    # Distribute remaining allocation proportionally
    if remaining_pct > 0.001 and allocations:
        bonus = remaining_pct / len(allocations)
        for a in allocations:
            a.allocation_pct = min(config.max_single_allocation_pct, a.allocation_pct + bonus)
            a.amount_usd = round(investment_usd * a.allocation_pct, 2)

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


class PortfolioOptimizer:
    """Optimizes DeFi portfolio allocation to maximize yield under risk constraints."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.risk_engine = RiskEngine()

    def optimize(
        self,
        pools: list[PoolInfo],
        investment_usd: float,
    ) -> OptimizedPortfolio:
        """Find optimal allocation across pools.

        Args:
            pools: Available yield farming pools.
            investment_usd: Total investment amount in USD.

        Returns:
            Optimized portfolio with allocations.

        Raises:
            ValueError: If no pools meet risk criteria.
        """
        if investment_usd <= 0:
            raise ValueError(f"Investment must be positive, got {investment_usd}")

        # Filter by minimum TVL
        eligible = [p for p in pools if p.tvl_usd >= self.config.min_tvl_usd]

        if self.config.stablecoins_only:
            eligible = [p for p in eligible if p.is_stable]

        # Filter by preferred chains/protocols
        if self.config.preferred_chains:
            eligible = [p for p in eligible if p.chain in self.config.preferred_chains]
        if self.config.preferred_protocols:
            eligible = [p for p in eligible if p.protocol in self.config.preferred_protocols]

        # Score and filter by risk
        risk_filtered = self.risk_engine.filter_by_risk(
            eligible, max_risk=self.config.max_risk_score
        )

        if not risk_filtered:
            return OptimizedPortfolio(
                allocations=[],
                total_expected_apy=0.0,
                weighted_risk_score=0.0,
                total_investment_usd=investment_usd,
            )

        return _greedy_optimize(risk_filtered, investment_usd, self.config)
