"""Impermanent Loss Calculator for DeFi liquidity positions.

Impermanent loss (IL) is the opportunity cost of providing liquidity compared
to simply holding the underlying tokens.  This module computes IL for various
AMM pool types and integrates with the yield aggregator to show net returns
after accounting for IL.

Supported pool types:

1. **Constant Product (x·y = k)** — Classic Uniswap V2 / SushiSwap 50/50 pools.
   IL = 2√r / (1 + r) − 1  where r = p₁/p₀ (price ratio).

2. **Weighted Pools** — Balancer-style pools with arbitrary weights.
   Generalises the constant-product formula to n tokens with weights wᵢ.

3. **StableSwap** — Curve-style pools for pegged assets with low IL.
   IL is heavily dampened by the amplification coefficient; modelled as a
   fraction of the constant-product IL.

4. **Concentrated Liquidity** — Uniswap V3 positions bounded by [p_a, p_b].
   IL depends on whether the price stays within range and the position width.

All calculations are deterministic and pure (no I/O), making them fast and
easy to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PoolType(str, Enum):
    """AMM pool type determines the IL formula used."""

    CONSTANT_PRODUCT = "constant_product"  # x*y=k (Uniswap V2, SushiSwap)
    WEIGHTED = "weighted"  # Balancer weighted pools
    STABLE_SWAP = "stable_swap"  # Curve StableSwap
    CONCENTRATED = "concentrated"  # Uniswap V3


@dataclass(frozen=True)
class ILResult:
    """Result of an impermanent loss calculation.

    Attributes:
        il_pct: Impermanent loss as a positive fraction (e.g. 0.05 = 5% loss).
            Always >= 0; a value of 0 means no IL.
        hold_value: USD value if tokens were simply held.
        lp_value: USD value of the LP position after price change.
        price_ratio: New price / old price for the volatile asset.
        annualized_il_pct: IL extrapolated to one year assuming the given
            price move happened over ``holding_days``.
        net_apy: Pool APY minus annualised IL — the true yield.
    """

    il_pct: float
    hold_value: float
    lp_value: float
    price_ratio: float
    annualized_il_pct: float = 0.0
    net_apy: float = 0.0


@dataclass(frozen=True)
class BreakEvenAnalysis:
    """Break-even analysis for an LP position.

    Attributes:
        break_even_price_ratio: Price ratio at which IL exactly equals
            cumulative yield earned.  None if IL never exceeds yield
            (position is always profitable).
        break_even_days: Estimated days until IL exceeds yield at the
            given daily price volatility.  None if never.
        max_tolerable_move: Maximum price ratio move before IL exceeds
            one year of yield at the pool APY.
    """

    break_even_price_ratio: Optional[float]
    break_even_days: Optional[float]
    max_tolerable_move: float


# ---------------------------------------------------------------------------
# Core IL formulas
# ---------------------------------------------------------------------------


def il_constant_product(price_ratio: float) -> float:
    """Impermanent loss for a 50/50 constant-product (x·y=k) pool.

    This is the canonical Uniswap V2 formula.

    Args:
        price_ratio: New price / old price of the volatile asset
            (e.g. 2.0 means the asset doubled).

    Returns:
        IL as a positive fraction (e.g. 0.006 = 0.6% loss).
        Returns 0.0 if price_ratio <= 0.
    """
    if price_ratio <= 0:
        return 0.0
    # IL = 2*sqrt(r) / (1+r) - 1  (negative value, we return positive)
    il = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
    return abs(il)


def il_weighted_pool(price_ratios: list[float], weights: list[float]) -> float:
    """Impermanent loss for a weighted pool (Balancer-style).

    Generalises the constant-product IL to n tokens with arbitrary weights.
    The formula is::

        IL = Π(rᵢ^wᵢ) / Σ(wᵢ · rᵢ) − 1

    where rᵢ is the price ratio and wᵢ is the weight of token i.

    Args:
        price_ratios: Price ratio for each token (new/old).  Must have the
            same length as ``weights``.
        weights: Portfolio weight of each token (must sum to 1.0).

    Returns:
        IL as a positive fraction.  Returns 0.0 on invalid input.

    Raises:
        ValueError: If lengths mismatch or weights don't sum to ~1.
    """
    if len(price_ratios) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(price_ratios)} price ratios vs {len(weights)} weights"
        )
    total_w = sum(weights)
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_w:.6f}")
    if any(r <= 0 for r in price_ratios):
        return 0.0

    # Geometric mean (weighted): Π(rᵢ^wᵢ)
    geometric = math.exp(sum(w * math.log(r) for r, w in zip(price_ratios, weights)))
    # Weighted average: Σ(wᵢ · rᵢ)
    weighted_avg = sum(w * r for r, w in zip(price_ratios, weights))

    if weighted_avg <= 0:
        return 0.0

    il = geometric / weighted_avg - 1
    return abs(il)


def il_stable_swap(
    price_ratio: float,
    amplification: float = 80.0,
) -> float:
    """Impermanent loss for a Curve StableSwap pool.

    StableSwap pools dampen IL significantly because the bonding curve
    is much flatter near the peg.  We model this as a fraction of the
    constant-product IL, where the fraction decreases with higher
    amplification::

        IL_stable ≈ IL_cp / (1 + A * deviation)

    ``deviation`` measures how far the price ratio is from 1.

    Args:
        price_ratio: New price / old price (e.g. 1.02 for 2% depeg).
        amplification: Curve amplification coefficient (A).  Typical values
            are 80-2000.  Higher = more like constant-sum (less IL).

    Returns:
        IL as a positive fraction.
    """
    if price_ratio <= 0:
        return 0.0

    base_il = il_constant_product(price_ratio)
    deviation = abs(price_ratio - 1.0)
    # Dampening factor: IL is reduced by (1 + A * deviation)
    dampening = 1.0 + amplification * deviation
    return base_il / dampening


def il_concentrated(
    price_ratio: float,
    p_lower: float,
    p_upper: float,
) -> float:
    """Impermanent loss for a concentrated liquidity (Uniswap V3) position.

    For a position in range [p_lower, p_upper] with initial price p₀, the
    LP value has a different formula than constant-product because liquidity
    is only active within the range.

    If the new price stays within range, the IL is amplified compared to a
    full-range position.  If the price exits the range, the position is
    100% in one asset.

    Args:
        price_ratio: New price / old price (p₁ / p₀).
        p_lower: Lower bound of the position range, relative to initial price.
            E.g. 0.75 means 25% below initial price.
        p_upper: Upper bound of the position range, relative to initial price.
            E.g. 1.50 means 50% above initial price.

    Returns:
        IL as a positive fraction.

    Raises:
        ValueError: If range bounds are invalid.
    """
    if p_lower <= 0 or p_upper <= 0:
        raise ValueError("Range bounds must be positive")
    if p_lower >= p_upper:
        raise ValueError(f"Lower bound ({p_lower}) must be < upper bound ({p_upper})")

    if price_ratio <= 0:
        return 0.0

    sqrt_p = math.sqrt(price_ratio)
    sqrt_pl = math.sqrt(p_lower)
    sqrt_pu = math.sqrt(p_upper)

    # Derivation for Uniswap V3 concentrated liquidity IL.
    #
    # We normalise liquidity L = 1 and initial price p₀ = 1.0.
    #
    # Initial token amounts:
    #   x₀ = 1/√p₀ − 1/√p_upper  = 1 − 1/√p_upper
    #   y₀ = √p₀ − √p_lower       = 1 − √p_lower
    #
    # Hold value at new price p:
    #   V_hold(p) = x₀·p + y₀
    #
    # LP value at new price p (within range):
    #   x(p) = 1/√p − 1/√p_upper
    #   y(p) = √p − √p_lower
    #   V_lp(p) = x(p)·p + y(p) = 2√p − p/√p_upper − √p_lower
    #
    # Below range (p < p_lower): all token0
    #   x = 1/√p_lower − 1/√p_upper,  y = 0
    #   V_lp = x · p
    #
    # Above range (p > p_upper): all token1
    #   x = 0,  y = √p_upper − √p_lower
    #   V_lp = y
    #
    # IL = |1 − V_lp / V_hold|

    x0 = 1.0 - 1.0 / sqrt_pu
    y0 = 1.0 - sqrt_pl
    hold_val = x0 * price_ratio + y0

    if price_ratio < p_lower:
        # All liquidity converted to token0
        x_below = 1.0 / sqrt_pl - 1.0 / sqrt_pu
        lp_val = x_below * price_ratio
    elif price_ratio > p_upper:
        # All liquidity converted to token1
        y_above = sqrt_pu - sqrt_pl
        lp_val = y_above
    else:
        # Position is in range
        lp_val = 2.0 * sqrt_p - price_ratio / sqrt_pu - sqrt_pl

    if hold_val <= 0:
        return 0.0

    return abs(1.0 - lp_val / hold_val)


# ---------------------------------------------------------------------------
# High-level calculator
# ---------------------------------------------------------------------------


def calculate_il(
    pool_type: PoolType,
    price_ratio: float,
    weights: Optional[list[float]] = None,
    amplification: float = 80.0,
    p_lower: Optional[float] = None,
    p_upper: Optional[float] = None,
    initial_value_usd: float = 10_000.0,
    pool_apy: float = 0.0,
    holding_days: int = 365,
) -> ILResult:
    """Calculate impermanent loss for a liquidity position.

    This is the main entry point that dispatches to the correct formula
    based on pool type.

    Args:
        pool_type: Type of AMM pool.
        price_ratio: New price / old price for the volatile asset.
        weights: Token weights for weighted pools.  Defaults to [0.5, 0.5].
        amplification: StableSwap amplification coefficient.
        p_lower: Lower price bound for concentrated positions (relative).
        p_upper: Upper price bound for concentrated positions (relative).
        initial_value_usd: Initial position value in USD.
        pool_apy: Pool APY as decimal (e.g. 0.05 = 5%).
        holding_days: Days held (for annualization).

    Returns:
        ILResult with IL percentage, hold vs LP values, and net APY.
    """
    if weights is None:
        weights = [0.5, 0.5]

    # Dispatch to correct formula
    if pool_type == PoolType.CONSTANT_PRODUCT:
        il = il_constant_product(price_ratio)
    elif pool_type == PoolType.WEIGHTED:
        # For weighted pools, assume the first token changed by price_ratio
        # and others stayed flat
        ratios = [price_ratio] + [1.0] * (len(weights) - 1)
        il = il_weighted_pool(ratios, weights)
    elif pool_type == PoolType.STABLE_SWAP:
        il = il_stable_swap(price_ratio, amplification)
    elif pool_type == PoolType.CONCENTRATED:
        if p_lower is None or p_upper is None:
            raise ValueError("p_lower and p_upper required for concentrated positions")
        il = il_concentrated(price_ratio, p_lower, p_upper)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")

    # Hold value: weighted average of token values after price change
    # For a 2-token pool with equal weight:
    hold_value = initial_value_usd * (0.5 + 0.5 * price_ratio)
    # LP value = hold_value * (1 - IL)
    lp_value = hold_value * (1 - il)

    # Annualize IL
    annualized_il = 0.0
    if holding_days > 0 and il > 0:
        # Compound the IL to annual scale
        periods_per_year = 365.0 / holding_days
        annualized_il = 1 - (1 - il) ** periods_per_year

    # Net APY = pool yield - annualized IL
    net_apy = pool_apy - annualized_il

    return ILResult(
        il_pct=round(il, 8),
        hold_value=round(hold_value, 2),
        lp_value=round(lp_value, 2),
        price_ratio=price_ratio,
        annualized_il_pct=round(annualized_il, 8),
        net_apy=round(net_apy, 8),
    )


def break_even_analysis(
    pool_type: PoolType,
    pool_apy: float,
    weights: Optional[list[float]] = None,
    amplification: float = 80.0,
    p_lower: Optional[float] = None,
    p_upper: Optional[float] = None,
    daily_volatility: float = 0.02,
) -> BreakEvenAnalysis:
    """Find the price move at which IL equals one year of yield.

    Useful for risk assessment: "How much can the price move before I
    lose money vs just holding?"

    Args:
        pool_type: Type of AMM pool.
        pool_apy: Pool APY as decimal (e.g. 0.05 = 5%).
        weights: Token weights for weighted pools.
        amplification: StableSwap amplification coefficient.
        p_lower: Lower price bound for concentrated positions.
        p_upper: Upper price bound for concentrated positions.
        daily_volatility: Daily price volatility for time-to-break-even.

    Returns:
        BreakEvenAnalysis with price ratio and time estimates.
    """
    if weights is None:
        weights = [0.5, 0.5]
    if pool_apy <= 0:
        return BreakEvenAnalysis(
            break_even_price_ratio=None,
            break_even_days=None,
            max_tolerable_move=float("inf"),
        )

    # Binary search for the break-even price ratio (upward move)
    def il_for_ratio(r: float) -> float:
        if pool_type == PoolType.CONSTANT_PRODUCT:
            return il_constant_product(r)
        elif pool_type == PoolType.WEIGHTED:
            ratios = [r] + [1.0] * (len(weights) - 1)
            return il_weighted_pool(ratios, weights)
        elif pool_type == PoolType.STABLE_SWAP:
            return il_stable_swap(r, amplification)
        elif pool_type == PoolType.CONCENTRATED:
            if p_lower is None or p_upper is None:
                return 0.0
            return il_concentrated(r, p_lower, p_upper)
        return 0.0

    # Search for the price ratio where IL = pool_apy
    lo, hi = 1.001, 20.0
    break_even_ratio: Optional[float] = None

    for _ in range(100):  # binary search iterations
        mid = (lo + hi) / 2
        il_at_mid = il_for_ratio(mid)
        if il_at_mid < pool_apy:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-8:
            break

    # Check if we actually found a break-even
    if il_for_ratio(hi) >= pool_apy:
        break_even_ratio = round(hi, 6)
    else:
        break_even_ratio = None

    # Also check downward (price_ratio < 1)
    if break_even_ratio is None:
        lo_d, hi_d = 0.01, 0.999
        for _ in range(100):
            mid = (lo_d + hi_d) / 2
            il_at_mid = il_for_ratio(mid)
            if il_at_mid < pool_apy:
                hi_d = mid
            else:
                lo_d = mid
            if abs(hi_d - lo_d) < 1e-8:
                break
        if il_for_ratio(lo_d) >= pool_apy:
            break_even_ratio = round(lo_d, 6)

    # Time to break-even
    break_even_days: Optional[float] = None
    if break_even_ratio is not None and daily_volatility > 0:
        # Expected time for |log(price_ratio)| using random walk:
        # E[|log(r)|] ≈ σ * sqrt(t)  =>  t = (log(r)/σ)²
        log_move = abs(math.log(break_even_ratio))
        if daily_volatility > 0:
            break_even_days = round((log_move / daily_volatility) ** 2, 1)

    # Max tolerable move = IL at ratio r equals pool_apy
    max_tolerable = break_even_ratio if break_even_ratio is not None else float("inf")

    return BreakEvenAnalysis(
        break_even_price_ratio=break_even_ratio,
        break_even_days=break_even_days,
        max_tolerable_move=max_tolerable,
    )


def il_sensitivity_table(
    pool_type: PoolType,
    price_ratios: list[float],
    weights: Optional[list[float]] = None,
    amplification: float = 80.0,
    p_lower: Optional[float] = None,
    p_upper: Optional[float] = None,
) -> list[ILResult]:
    """Generate an IL sensitivity table for a range of price movements.

    Useful for visualising how IL scales with price changes.

    Args:
        pool_type: Type of AMM pool.
        price_ratios: List of price ratios to evaluate.
        weights: Token weights for weighted pools.
        amplification: StableSwap amplification coefficient.
        p_lower: Lower price bound for concentrated positions.
        p_upper: Upper price bound for concentrated positions.

    Returns:
        List of ILResult for each price ratio.
    """
    results: list[ILResult] = []
    for r in price_ratios:
        result = calculate_il(
            pool_type=pool_type,
            price_ratio=r,
            weights=weights,
            amplification=amplification,
            p_lower=p_lower,
            p_upper=p_upper,
        )
        results.append(result)
    return results
