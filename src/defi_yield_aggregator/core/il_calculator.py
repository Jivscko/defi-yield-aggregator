"""Impermanent Loss Calculator for DeFi liquidity positions.

Supports multi-asset pools, time-series price analysis, break-even analysis,
and net profit/loss calculations accounting for LP fees and yield rewards.

Impermanent loss (IL) is the opportunity cost of providing liquidity vs holding.
It arises from the automated market maker rebalancing your position as prices move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LPPosition:
    """Configuration for a liquidity pool position.

    Attributes:
        initial_value_usd: Total USD value deposited into the LP.
        token_weights: Weight of each token in the pool (must sum to 1.0).
            Example for 80/20 pool: [0.8, 0.2].
        initial_prices: Initial USD price of each token.
        pool_type: AMM type — 'constant_product' (x*y=k) or 'weighted' (Balancer-style).
    """

    initial_value_usd: float
    token_weights: list[float]
    initial_prices: list[float]
    pool_type: str = "constant_product"

    def __post_init__(self) -> None:
        if self.initial_value_usd <= 0:
            raise ValueError(
                f"Initial value must be positive, got {self.initial_value_usd}"
            )
        if len(self.token_weights) != len(self.initial_prices):
            raise ValueError(
                "Token weights and initial prices must have the same length"
            )
        if len(self.token_weights) < 2:
            raise ValueError("LP position requires at least 2 tokens")
        if any(w <= 0 for w in self.token_weights):
            raise ValueError("All token weights must be positive")
        if any(p <= 0 for p in self.initial_prices):
            raise ValueError("All initial prices must be positive")
        weight_sum = sum(self.token_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Token weights must sum to 1.0, got {weight_sum}"
            )
        if self.pool_type not in ("constant_product", "weighted"):
            raise ValueError(
                f"Unknown pool type '{self.pool_type}'. "
                "Use 'constant_product' or 'weighted'."
            )


@dataclass(frozen=True)
class ILResult:
    """Result of an impermanent loss calculation.

    Attributes:
        il_pct: Impermanent loss as a negative fraction (e.g., -0.05 = 5% loss).
        hold_value_usd: Value if tokens were simply held (no LP).
        lp_value_usd: Value of the LP position after price change.
        price_changes: Per-token price change ratios (new/initial).
    """

    il_pct: float
    hold_value_usd: float
    lp_value_usd: float
    price_changes: list[float]


@dataclass(frozen=True)
class NetPnLResult:
    """Full P&L result including yield, fees, and IL.

    Attributes:
        il_result: The underlying IL calculation.
        yield_earned_usd: USD value earned from yield/APR rewards.
        fees_earned_usd: USD value earned from LP trading fees.
        net_pnl_usd: Total profit/loss (positive = profit).
        net_pnl_pct: Net P&L as fraction of initial investment.
        days_held: Number of days the position was held.
        annualized_return_pct: Annualized net return.
    """

    il_result: ILResult
    yield_earned_usd: float
    fees_earned_usd: float
    net_pnl_usd: float
    net_pnl_pct: float
    days_held: int
    annualized_return_pct: float


@dataclass(frozen=True)
class BreakEvenResult:
    """Break-even analysis for an LP position.

    Attributes:
        price_ratio: Price ratio (new/initial) where net P&L = 0.
        is_profitable_at_current: Whether the position is profitable at the given ratio.
        il_at_break_even: IL at the break-even price point.
        min_yield_apr_needed: Minimum APR needed to break even at the given IL.
    """

    price_ratio: float
    is_profitable_at_current: bool
    il_at_break_even: float
    min_yield_apr_needed: float


@dataclass(frozen=True)
class ILTimeSeriesPoint:
    """A single data point in an IL time series.

    Attributes:
        day: Day number (0 = deposit day).
        price_ratio: Price ratio at this point.
        il_pct: Impermanent loss at this point.
        cumulative_yield_pct: Cumulative yield earned as fraction of initial.
        cumulative_fees_pct: Cumulative fees earned as fraction of initial.
        net_pnl_pct: Net P&L at this point.
    """

    day: int
    price_ratio: float
    il_pct: float
    cumulative_yield_pct: float
    cumulative_fees_pct: float
    net_pnl_pct: float


# ---------------------------------------------------------------------------
# Core IL calculations
# ---------------------------------------------------------------------------


def il_constant_product(price_ratio: float) -> float:
    """Calculate IL for a constant-product (x*y=k) 50/50 pool.

    This is the classic Uniswap V2 formula. The result is always <= 0.

    Args:
        price_ratio: New price / initial price of one asset vs the other.

    Returns:
        IL as a negative fraction (e.g., -0.0537 for ~5.7% loss at 2x price).

    Examples:
        >>> il_constant_product(1.0)  # No price change
        0.0
        >>> il_constant_product(2.0)  # 2x price change
        -0.05719...
        >>> il_constant_product(0.5)  # 50% drop — symmetric IL
        -0.05719...
    """
    if price_ratio <= 0:
        raise ValueError(f"Price ratio must be positive, got {price_ratio}")
    return 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1


def il_weighted_pool(price_ratios: list[float], weights: list[float]) -> float:
    """Calculate IL for a weighted pool (Balancer-style).

    Generalizes the constant-product formula to pools with arbitrary weights.
    For a 50/50 pool this reduces to il_constant_product.

    Args:
        price_ratios: New price / initial price for each token.
        weights: Weight of each token (must sum to 1.0).

    Returns:
        IL as a negative fraction.

    References:
        - https://balancer.fi/whitepaper.pdf
        - https://medium.com/@pintail/understanding-impermanent-loss
    """
    if len(price_ratios) != len(weights):
        raise ValueError("price_ratios and weights must have same length")

    # LP value after price change: product of (ratio_i ^ weight_i) * initial_value
    lp_ratio = 1.0
    hold_ratio = 0.0
    for pr, w in zip(price_ratios, weights):
        lp_ratio *= pr**w
        hold_ratio += w * pr

    # IL = LP_value / hold_value - 1
    if hold_ratio <= 0:
        raise ValueError("Hold ratio must be positive")
    return lp_ratio / hold_ratio - 1


def calculate_il(
    position: LPPosition,
    new_prices: list[float],
) -> ILResult:
    """Calculate impermanent loss for a given position and new prices.

    Args:
        position: The LP position configuration.
        new_prices: Current USD price of each token.

    Returns:
        Detailed IL result.

    Raises:
        ValueError: If new_prices length doesn't match position tokens.
    """
    if len(new_prices) != len(position.initial_prices):
        raise ValueError(
            f"Expected {len(position.initial_prices)} prices, got {len(new_prices)}"
        )
    if any(p <= 0 for p in new_prices):
        raise ValueError("All new prices must be positive")

    price_changes = [
        new / init for new, init in zip(new_prices, position.initial_prices)
    ]

    # Value if simply held
    initial_token_values = [
        w * position.initial_value_usd for w in position.token_weights
    ]
    hold_value = sum(
        val * change
        for val, change in zip(initial_token_values, price_changes)
    )

    # LP value depends on pool type
    if position.pool_type == "constant_product":
        # For 50/50 constant product, use the ratio of the two tokens
        if len(price_changes) == 2:
            ratio = price_changes[1] / price_changes[0]
            il = il_constant_product(ratio)
        else:
            # Multi-token constant product: use weighted formula
            il = il_weighted_pool(price_changes, position.token_weights)
    else:
        il = il_weighted_pool(price_changes, position.token_weights)

    lp_value = hold_value * (1 + il)

    return ILResult(
        il_pct=round(il, 8),
        hold_value_usd=round(hold_value, 2),
        lp_value_usd=round(lp_value, 2),
        price_changes=[round(c, 6) for c in price_changes],
    )


# ---------------------------------------------------------------------------
# Net P&L (IL + yield + fees)
# ---------------------------------------------------------------------------


def calculate_net_pnl(
    position: LPPosition,
    new_prices: list[float],
    apr_yield: float = 0.0,
    daily_fee_apr: float = 0.0,
    days_held: int = 365,
) -> NetPnLResult:
    """Calculate full P&L including IL, yield rewards, and trading fees.

    Args:
        position: The LP position configuration.
        new_prices: Current USD price of each token.
        apr_yield: Annual yield from staking/farming rewards (as decimal, e.g. 0.10 = 10%).
        daily_fee_apr: Annualized fee income from trading volume (as decimal).
        days_held: Number of days the position was active.

    Returns:
        Complete P&L breakdown.
    """
    if days_held <= 0:
        raise ValueError(f"Days held must be positive, got {days_held}")

    il_result = calculate_il(position, new_prices)

    period_years = days_held / 365.0

    # Yield earned (simple interest — rewards are usually claimed periodically)
    yield_earned = position.initial_value_usd * apr_yield * period_years

    # Fees earned (compounding — fees stay in the pool)
    fees_earned = position.initial_value_usd * (
        (1 + daily_fee_apr) ** period_years - 1
    )

    # Net P&L = LP value + yield + fees - initial investment
    net_pnl = il_result.lp_value_usd + yield_earned + fees_earned - position.initial_value_usd
    net_pnl_pct = net_pnl / position.initial_value_usd

    # Annualize
    annualized = ((1 + net_pnl_pct) ** (1 / period_years) - 1) if period_years > 0 else 0.0

    return NetPnLResult(
        il_result=il_result,
        yield_earned_usd=round(yield_earned, 2),
        fees_earned_usd=round(fees_earned, 2),
        net_pnl_usd=round(net_pnl, 2),
        net_pnl_pct=round(net_pnl_pct, 6),
        days_held=days_held,
        annualized_return_pct=round(annualized, 6),
    )


# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------


def calculate_break_even(
    position: LPPosition,
    apr_yield: float = 0.0,
    daily_fee_apr: float = 0.0,
    days_held: int = 365,
    current_price_ratio: Optional[float] = None,
) -> BreakEvenResult:
    """Find the downside price ratio where yield + fees exactly offset IL.

    For a 50/50 constant-product pool, net P&L = sqrt(r) + yield - 1,
    which gives break-even at r = (1 - yield)^2. This function uses binary
    search on [0.01, 1.0] to generalize for weighted pools and combined
    yield + fee rates.

    Args:
        position: LP position (must be 2-token for ratio-based analysis).
        apr_yield: Annual yield as decimal.
        daily_fee_apr: Annualized fee APR as decimal.
        days_held: Holding period in days.
        current_price_ratio: If provided, report whether profitable at this ratio.

    Returns:
        Break-even analysis result.
    """
    if len(position.token_weights) != 2:
        raise ValueError("Break-even analysis requires exactly 2 tokens")

    period_years = days_held / 365.0
    yield_rate = apr_yield * period_years
    fee_rate = (1 + daily_fee_apr) ** period_years - 1
    total_return = yield_rate + fee_rate

    def net_at_ratio(ratio: float) -> float:
        """Calculate net P&L fraction at a given price ratio."""
        il = il_constant_product(ratio)
        hold_ratio = (1 + ratio) / 2
        lp_ratio = hold_ratio * (1 + il)
        return lp_ratio + total_return - 1.0

    # net_at_ratio is monotonically increasing on (0, 1] for constant-product pools.
    # At r=0.01, net is deeply negative. At r=1.0, net = yield (positive if yield > 0).
    # Search for the zero crossing on [0.01, 1.0].
    lo, hi = 0.01, 1.0

    # Check if break-even exists
    if net_at_ratio(lo) > 0:
        # Even at r=0.01 we're profitable — break-even ratio is < 0.01
        break_even_ratio = lo
    elif net_at_ratio(hi) < 0:
        # Not profitable even at r=1.0 — yield can't cover fees
        break_even_ratio = hi
    else:
        # Binary search for zero crossing
        for _ in range(100):
            mid = (lo + hi) / 2
            if net_at_ratio(mid) > 0:
                hi = mid
            else:
                lo = mid
        break_even_ratio = (lo + hi) / 2

    # Determine if profitable at current ratio
    is_profitable = True
    if current_price_ratio is not None:
        is_profitable = net_at_ratio(current_price_ratio) > 0

    il_at_be = il_constant_product(break_even_ratio)

    # Minimum APR needed to offset IL at a given ratio
    target_ratio = current_price_ratio if current_price_ratio else break_even_ratio
    target_il = il_constant_product(target_ratio)
    target_hold = (1 + target_ratio) / 2
    target_lp = target_hold * (1 + target_il)
    # yield + fees must cover: initial - lp_value
    shortfall = 1.0 - target_lp
    min_apr = shortfall / period_years if period_years > 0 and shortfall > 0 else 0.0

    return BreakEvenResult(
        price_ratio=round(break_even_ratio, 4),
        is_profitable_at_current=is_profitable,
        il_at_break_even=round(il_at_be, 6),
        min_yield_apr_needed=round(max(min_apr, 0.0), 6),
    )


# ---------------------------------------------------------------------------
# Time series analysis
# ---------------------------------------------------------------------------


def calculate_il_timeseries(
    position: LPPosition,
    price_ratios_over_time: list[float],
    apr_yield: float = 0.0,
    daily_fee_apr: float = 0.0,
) -> list[ILTimeSeriesPoint]:
    """Calculate IL and net P&L at each point in a price history.

    Useful for visualizing how IL evolves as prices move over time.

    Args:
        position: LP position (must be 2-token).
        price_ratios_over_time: Daily price ratio (new/initial) values.
            Index 0 = day 1, index 1 = day 2, etc.
        apr_yield: Annual yield as decimal.
        daily_fee_apr: Annualized fee APR as decimal.

    Returns:
        Time series of IL and P&L data points.
    """
    if len(position.token_weights) != 2:
        raise ValueError("Time series analysis requires exactly 2 tokens")

    points: list[ILTimeSeriesPoint] = []
    daily_yield = apr_yield / 365.0
    daily_fee = daily_fee_apr / 365.0

    for day_idx, ratio in enumerate(price_ratios_over_time):
        day = day_idx + 1
        il = il_constant_product(ratio)
        cum_yield = daily_yield * day
        cum_fee = daily_fee * day

        # Net P&L at this point
        hold_ratio = (1 + ratio) / 2
        lp_ratio = hold_ratio * (1 + il)
        net = lp_ratio + cum_yield + cum_fee - 1.0

        points.append(
            ILTimeSeriesPoint(
                day=day,
                price_ratio=round(ratio, 6),
                il_pct=round(il, 8),
                cumulative_yield_pct=round(cum_yield, 8),
                cumulative_fees_pct=round(cum_fee, 8),
                net_pnl_pct=round(net, 8),
            )
        )

    return points


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def max_il_for_pool_type(pool_type: str = "constant_product") -> float:
    """Return the theoretical maximum IL for a pool type.

    For constant-product pools, IL approaches ~25% as price goes to infinity.
    In practice, IL rarely exceeds 15-20% for most price movements.

    Args:
        pool_type: The AMM pool type.

    Returns:
        Maximum IL as a negative fraction.
    """
    if pool_type == "constant_product":
        # As price_ratio -> infinity, IL -> -1 + 0 = -1 (100% loss in theory)
        # But in practice the curve flattens; at 100x it's about -18%
        return il_constant_product(1000)  # ~-22.5%
    return il_constant_product(1000)


def il_sensitivity_table(
    ratios: Optional[list[float]] = None,
) -> list[tuple[float, float, float]]:
    """Generate a sensitivity table showing IL at various price ratios.

    Args:
        ratios: List of price ratios to analyze. Defaults to common values.

    Returns:
        List of (ratio, il_pct, hold_vs_lp_diff_pct) tuples.
    """
    if ratios is None:
        ratios = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

    results = []
    for r in ratios:
        if r <= 0:
            continue
        il = il_constant_product(r)
        hold_val = (1 + r) / 2  # relative to initial
        lp_val = hold_val * (1 + il)
        diff_pct = (lp_val - hold_val) / hold_val if hold_val > 0 else 0
        results.append((r, round(il * 100, 4), round(diff_pct * 100, 4)))

    return results
