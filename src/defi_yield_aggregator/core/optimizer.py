"""Portfolio optimizer - maximize yield given risk constraints.

Supports three allocation strategies:

1. **Greedy** (default): Sort by yield/risk ratio and allocate top pools.
2. **Kelly Criterion**: Use Kelly formula to size positions based on expected
   edge and variance, maximizing long-term geometric growth rate.
3. **Risk Parity**: Allocate so each position contributes equally to total
   portfolio risk, using inverse-volatility weighting.
"""

from __future__ import annotations

import math
from enum import Enum
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


class AllocationStrategy(str, Enum):
    """Available portfolio allocation strategies."""

    GREEDY = "greedy"
    KELLY = "kelly"
    RISK_PARITY = "risk_rarity"


def _pool_volatility(pool: PoolInfo, risk: RiskScore) -> float:
    """Estimate volatility for a pool from available risk signals.

    Uses the risk score as a proxy for return uncertainty.  Higher risk pools
    are assumed to have wider return distributions.  The result is a per-period
    (daily-equivalent) standard deviation expressed as a decimal.

    Args:
        pool: Pool information.
        risk: Risk score for the pool.

    Returns:
        Estimated volatility (>= 0.001 to avoid division by zero).
    """
    # Base volatility from risk score (0-100 mapped to 0.01 - 0.40)
    base_vol = 0.01 + (risk.overall_score / 100.0) * 0.39

    # Impermanent loss amplifies volatility
    il_boost = pool.impermanent_loss_risk * 0.15

    # Stable pools are inherently less volatile
    stable_factor = 0.5 if pool.is_stable else 1.0

    vol = (base_vol + il_boost) * stable_factor
    return max(vol, 0.001)


def _pool_edge(pool: PoolInfo, risk: RiskScore) -> float:
    """Estimate the expected edge (net excess return) for a pool.

    Edge = APY minus a risk-adjusted discount.  This serves as the "win rate"
    analogue for Kelly sizing.

    Args:
        pool: Pool information.
        risk: Risk score for the pool.

    Returns:
        Estimated edge (can be negative for unprofitable pools).
    """
    # Risk-adjusted discount: higher risk -> larger haircut
    risk_discount = (risk.overall_score / 100.0) * pool.apy * 0.5
    return pool.apy - risk_discount


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


def _kelly_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
    kelly_fraction: float = 0.5,
) -> OptimizedPortfolio:
    """Kelly Criterion allocation: size positions for maximum geometric growth.

    The Kelly Criterion determines the optimal fraction of wealth to allocate
    to each bet (pool) to maximize the expected logarithm of terminal wealth.
    We use a fractional Kelly (default 0.5) to reduce volatility.

    For each pool the Kelly fraction is::

        f* = edge / variance

    where *edge* is the risk-adjusted expected excess return and *variance* is
    the squared estimated volatility.  Positions are then capped at
    ``max_single_allocation_pct`` and re-normalized.

    Args:
        candidates: Pools with their risk scores.
        investment_usd: Total capital to allocate.
        config: Portfolio constraints.
        kelly_fraction: Fractional Kelly multiplier (0-1).  Lower is more
            conservative.  Default 0.5 (half-Kelly) balances growth and
            drawdown risk.

    Returns:
        Optimized portfolio with Kelly-sized allocations.
    """
    if not candidates:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Compute raw Kelly fractions
    kelly_fracs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk in candidates:
        edge = _pool_edge(pool, risk)
        vol = _pool_volatility(pool, risk)
        variance = vol ** 2

        if variance <= 0 or edge <= 0:
            raw_kelly = 0.0
        else:
            raw_kelly = edge / variance

        # Apply fractional Kelly
        frac = raw_kelly * kelly_fraction
        kelly_fracs.append((pool, risk, frac))

    # Filter out zero-allocation pools
    kelly_fracs = [(p, r, f) for p, r, f in kelly_fracs if f > 1e-6]

    if not kelly_fracs:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Sort by Kelly fraction descending, take top N
    kelly_fracs.sort(key=lambda x: x[2], reverse=True)
    kelly_fracs = kelly_fracs[: config.max_positions]

    # Normalize fractions to sum to at most 1.0
    total_raw = sum(f for _, _, f in kelly_fracs)
    if total_raw > 1.0:
        kelly_fracs = [(p, r, f / total_raw) for p, r, f in kelly_fracs]

    # Cap at max_single_allocation_pct and re-normalize
    allocations: list[PortfolioAllocation] = []
    capped_fracs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk, frac in kelly_fracs:
        capped = min(frac, config.max_single_allocation_pct)
        capped_fracs.append((pool, risk, capped))

    # Re-normalize if capping caused total < 1.0
    capped_total = sum(f for _, _, f in capped_fracs)
    if capped_total > 0 and capped_total < 0.999:
        scale = min(1.0 / capped_total, 1.0)
        capped_fracs = [(p, r, f * scale) for p, r, f in capped_fracs]

    for pool, risk, frac in capped_fracs:
        if frac < 0.001:
            continue
        amount = investment_usd * frac
        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(frac, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )

    if not allocations:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


def _risk_parity_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
) -> OptimizedPortfolio:
    """Risk Parity allocation: equal risk contribution from each position.

    Allocates inversely proportional to volatility so that every pool
    contributes the same amount of portfolio risk.  This is the
    "inverse-volatility weighting" variant of risk parity — simple,
    effective, and widely used.

    Weight_i = (1 / vol_i) / sum(1 / vol_j)

    Args:
        candidates: Pools with their risk scores.
        investment_usd: Total capital to allocate.
        config: Portfolio constraints.

    Returns:
        Optimized portfolio with risk-parity allocations.
    """
    if not candidates:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Compute inverse-volatility weights
    inv_vol_pairs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk in candidates:
        vol = _pool_volatility(pool, risk)
        inv_vol = 1.0 / vol
        inv_vol_pairs.append((pool, risk, inv_vol))

    # Sort by inverse-volatility descending (most stable first)
    inv_vol_pairs.sort(key=lambda x: x[2], reverse=True)

    # Take top max_positions
    inv_vol_pairs = inv_vol_pairs[: config.max_positions]

    total_inv_vol = sum(iv for _, _, iv in inv_vol_pairs)
    if total_inv_vol <= 0:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Raw risk-parity weights
    rp_weights = [(p, r, iv / total_inv_vol) for p, r, iv in inv_vol_pairs]

    # Cap at max_single_allocation_pct and re-normalize
    allocations: list[PortfolioAllocation] = []
    capped_weights: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk, w in rp_weights:
        capped = min(w, config.max_single_allocation_pct)
        capped_weights.append((pool, risk, capped))

    capped_total = sum(w for _, _, w in capped_weights)
    if capped_total > 0 and capped_total < 0.999:
        scale = min(1.0 / capped_total, 1.0)
        capped_weights = [(p, r, w * scale) for p, r, w in capped_weights]

    for pool, risk, w in capped_weights:
        if w < 0.001:
            continue
        amount = investment_usd * w
        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(w, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )

    if not allocations:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


class PortfolioOptimizer:
    """Optimizes DeFi portfolio allocation to maximize yield under risk constraints.

    Supports three allocation strategies:

    - ``GREEDY``: Sort by yield/risk ratio and allocate top pools.
    - ``KELLY``: Kelly Criterion sizing for maximum geometric growth.
    - ``RISK_PARITY``: Inverse-volatility weighting for equal risk contribution.

    Args:
        config: Portfolio constraints and preferences.
        strategy: Allocation strategy to use.
        kelly_fraction: Fractional Kelly multiplier (only used with KELLY strategy).
            0.5 = half-Kelly (conservative), 1.0 = full Kelly (aggressive).

    Example::

        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(pools, investment_usd=100_000)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        strategy: AllocationStrategy = AllocationStrategy.GREEDY,
        kelly_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in (0, 1], got {kelly_fraction}"
            )
        self.config = config or Config()
        self.risk_engine = RiskEngine()
        self.strategy = strategy
        self.kelly_fraction = kelly_fraction

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

        # Dispatch to the selected strategy
        if self.strategy is AllocationStrategy.KELLY:
            return _kelly_optimize(
                risk_filtered, investment_usd, self.config, self.kelly_fraction
            )
        if self.strategy is AllocationStrategy.RISK_PARITY:
            return _risk_parity_optimize(risk_filtered, investment_usd, self.config)
        # Default: greedy
        return _greedy_optimize(risk_filtered, investment_usd, self.config)
