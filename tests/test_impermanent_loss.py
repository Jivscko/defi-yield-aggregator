"""Tests for the Impermanent Loss Calculator.

Covers all pool types (constant product, weighted, stableswap, concentrated),
edge cases, break-even analysis, and sensitivity tables.
"""

from __future__ import annotations

import math

import pytest

from defi_yield_aggregator.core.impermanent_loss import (
    BreakEvenAnalysis,
    ILResult,
    PoolType,
    break_even_analysis,
    calculate_il,
    il_concentrated,
    il_constant_product,
    il_sensitivity_table,
    il_stable_swap,
    il_weighted_pool,
)


# ===================================================================
# Constant Product (Uniswap V2 / x*y=k)
# ===================================================================


class TestConstantProduct:
    """Tests for 50/50 constant-product IL formula."""

    def test_no_price_change(self) -> None:
        """IL is zero when price doesn't move."""
        assert il_constant_product(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_price_doubles(self) -> None:
        """2x price move → ~5.72% IL."""
        il = il_constant_product(2.0)
        # Expected: 2*sqrt(2)/(1+2) - 1 = 0.9428... - 1 = -0.0572...
        assert il == pytest.approx(0.05719, abs=1e-4)

    def test_price_halves(self) -> None:
        """0.5x price move → ~5.72% IL (symmetric)."""
        il = il_constant_product(0.5)
        assert il == pytest.approx(0.05719, abs=1e-4)

    def test_5x_price_move(self) -> None:
        """5x price → ~25.46% IL."""
        il = il_constant_product(5.0)
        assert il == pytest.approx(0.2546, abs=1e-3)

    def test_10x_price_move(self) -> None:
        """10x price → ~42.54% IL."""
        il = il_constant_product(10.0)
        assert il == pytest.approx(0.4254, abs=1e-3)

    def test_symmetry(self) -> None:
        """IL(r) == IL(1/r) for constant-product pools."""
        for ratio in [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]:
            assert il_constant_product(ratio) == pytest.approx(
                il_constant_product(1.0 / ratio), abs=1e-10
            )

    def test_zero_price_ratio(self) -> None:
        """IL is 0 for invalid (zero) price ratio."""
        assert il_constant_product(0.0) == 0.0

    def test_negative_price_ratio(self) -> None:
        """IL is 0 for invalid (negative) price ratio."""
        assert il_constant_product(-1.0) == 0.0

    def test_monotonically_increasing(self) -> None:
        """IL increases as price deviates further from 1.0."""
        ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        ils = [il_constant_product(r) for r in ratios]
        for i in range(1, len(ils)):
            assert ils[i] > ils[i - 1]


# ===================================================================
# Weighted Pools (Balancer)
# ===================================================================


class TestWeightedPool:
    """Tests for Balancer-style weighted pool IL."""

    def test_50_50_matches_constant_product(self) -> None:
        """50/50 weighted pool IL matches constant-product formula."""
        for ratio in [0.5, 1.0, 2.0, 5.0]:
            il_w = il_weighted_pool([ratio, 1.0], [0.5, 0.5])
            il_cp = il_constant_product(ratio)
            assert il_w == pytest.approx(il_cp, abs=1e-10)

    def test_no_price_change(self) -> None:
        """IL is zero when no price moves."""
        assert il_weighted_pool([1.0, 1.0], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-10)

    def test_80_20_pool(self) -> None:
        """80/20 weighted pool has less IL than 50/50 for same price move."""
        il_8020 = il_weighted_pool([2.0, 1.0], [0.8, 0.2])
        il_5050 = il_constant_product(2.0)
        # Heavier weighting on the volatile asset reduces IL
        assert il_8020 < il_5050
        assert il_8020 > 0

    def test_three_token_pool(self) -> None:
        """Three-token weighted pool calculation."""
        # Equal weight, one token doubles
        il = il_weighted_pool([2.0, 1.0, 1.0], [0.33, 0.33, 0.34])
        assert il > 0
        assert il < 0.1  # Should be modest with only one token moving

    def test_weight_mismatch_raises(self) -> None:
        """Length mismatch between prices and weights raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            il_weighted_pool([2.0, 1.0, 1.0], [0.5, 0.5])

    def test_weights_not_summing_to_one_raises(self) -> None:
        """Weights that don't sum to 1 raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            il_weighted_pool([2.0, 1.0], [0.6, 0.6])

    def test_zero_price_ratio(self) -> None:
        """IL is 0 for invalid (zero) price ratio in weighted pool."""
        assert il_weighted_pool([0.0, 1.0], [0.5, 0.5]) == 0.0

    def test_extreme_weight_pool(self) -> None:
        """95/5 pool — very concentrated weighting has minimal IL for the 95% token."""
        il = il_weighted_pool([2.0, 1.0], [0.95, 0.05])
        # IL should be very small since the volatile asset is 95% of the pool
        assert il < il_constant_product(2.0)
        assert il > 0


# ===================================================================
# StableSwap (Curve)
# ===================================================================


class TestStableSwap:
    """Tests for Curve StableSwap IL."""

    def test_no_depeg(self) -> None:
        """IL is zero when price stays at peg."""
        assert il_stable_swap(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_small_depeg(self) -> None:
        """Small depeg (2%) has very low IL for stableswap."""
        il_stable = il_stable_swap(1.02, amplification=80)
        il_cp = il_constant_product(1.02)
        # StableSwap should have much less IL
        assert il_stable < il_cp
        assert il_stable > 0

    def test_higher_amplification_less_il(self) -> None:
        """Higher A → less IL (closer to constant-sum)."""
        il_low_a = il_stable_swap(1.05, amplification=10)
        il_high_a = il_stable_swap(1.05, amplification=500)
        assert il_high_a < il_low_a

    def test_large_depeg(self) -> None:
        """Large depeg (50%) still has IL but dampened."""
        il = il_stable_swap(1.5, amplification=80)
        il_cp = il_constant_product(1.5)
        assert 0 < il < il_cp

    def test_below_peg(self) -> None:
        """Depeg below 1.0 also produces IL."""
        il = il_stable_swap(0.95, amplification=100)
        assert il > 0

    def test_zero_price_ratio(self) -> None:
        """IL is 0 for invalid (zero) price ratio."""
        assert il_stable_swap(0.0) == 0.0

    def test_default_amplification(self) -> None:
        """Default amplification (80) produces reasonable values."""
        il = il_stable_swap(1.05)
        assert il > 0
        assert il < 0.01  # Should be very small for 5% depeg with A=80


# ===================================================================
# Concentrated Liquidity (Uniswap V3)
# ===================================================================


class TestConcentrated:
    """Tests for Uniswap V3 concentrated liquidity IL."""

    def test_price_within_range(self) -> None:
        """IL when price stays within the position range."""
        il = il_concentrated(1.2, p_lower=0.75, p_upper=1.50)
        assert il > 0

    def test_price_below_range(self) -> None:
        """IL when price drops below the position range."""
        il = il_concentrated(0.5, p_lower=0.75, p_upper=1.50)
        assert il > 0
        # Below range means 100% in risky asset → significant IL
        assert il > 0.05

    def test_price_above_range(self) -> None:
        """IL when price goes above the position range."""
        il = il_concentrated(2.0, p_lower=0.75, p_upper=1.50)
        assert il > 0

    def test_no_price_change(self) -> None:
        """IL is zero when price stays at 1.0 (within range)."""
        il = il_concentrated(1.0, p_lower=0.75, p_upper=1.50)
        assert il == pytest.approx(0.0, abs=1e-10)

    def test_narrow_range_more_il(self) -> None:
        """Narrower range → more IL for same price move."""
        il_wide = il_concentrated(1.3, p_lower=0.5, p_upper=2.0)
        il_narrow = il_concentrated(1.3, p_lower=0.8, p_upper=1.2)
        assert il_narrow > il_wide

    def test_invalid_range_raises(self) -> None:
        """Lower >= upper raises ValueError."""
        with pytest.raises(ValueError, match="must be < upper"):
            il_concentrated(1.5, p_lower=1.5, p_upper=1.5)

    def test_zero_bound_raises(self) -> None:
        """Zero bounds raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            il_concentrated(1.5, p_lower=0.0, p_upper=1.5)

    def test_negative_price_ratio(self) -> None:
        """Negative price ratio returns 0."""
        assert il_concentrated(-1.0, p_lower=0.5, p_upper=2.0) == 0.0


# ===================================================================
# High-level calculate_il
# ===================================================================


class TestCalculateIL:
    """Tests for the main calculate_il entry point."""

    def test_constant_product_basic(self) -> None:
        """Basic constant product IL calculation."""
        result = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=2.0,
            initial_value_usd=10_000,
            pool_apy=0.05,
            holding_days=365,
        )
        assert isinstance(result, ILResult)
        assert result.il_pct == pytest.approx(0.05719, abs=1e-4)
        assert result.hold_value == 15_000.0  # 10k * (0.5 + 0.5*2)
        assert result.lp_value < result.hold_value
        assert result.net_apy < 0.05  # Net APY less than pool APY

    def test_weighted_pool_basic(self) -> None:
        """Weighted pool IL calculation."""
        result = calculate_il(
            pool_type=PoolType.WEIGHTED,
            price_ratio=2.0,
            weights=[0.8, 0.2],
        )
        assert result.il_pct > 0
        assert result.il_pct < il_constant_product(2.0)

    def test_stable_swap_basic(self) -> None:
        """StableSwap IL calculation."""
        result = calculate_il(
            pool_type=PoolType.STABLE_SWAP,
            price_ratio=1.05,
            amplification=100,
        )
        assert result.il_pct > 0
        assert result.il_pct < 0.01  # Very small for 5% depeg

    def test_concentrated_basic(self) -> None:
        """Concentrated liquidity IL calculation."""
        result = calculate_il(
            pool_type=PoolType.CONCENTRATED,
            price_ratio=1.2,
            p_lower=0.75,
            p_upper=1.50,
        )
        assert result.il_pct > 0

    def test_concentrated_requires_bounds(self) -> None:
        """Concentrated pool raises ValueError without bounds."""
        with pytest.raises(ValueError, match="p_lower and p_upper required"):
            calculate_il(pool_type=PoolType.CONCENTRATED, price_ratio=1.5)

    def test_net_apy_positive_for_small_move(self) -> None:
        """Net APY is positive for small price moves with decent yield."""
        result = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=1.05,  # 5% move
            pool_apy=0.10,  # 10% APY
            holding_days=365,
        )
        assert result.net_apy > 0

    def test_net_apy_negative_for_large_move(self) -> None:
        """Net APY turns negative for large price moves."""
        result = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=5.0,  # 5x move
            pool_apy=0.05,  # 5% APY
            holding_days=365,
        )
        assert result.net_apy < 0

    def test_annualized_il_scaling(self) -> None:
        """IL over shorter periods annualises correctly."""
        # IL over 30 days should annualise to a higher number
        result_30d = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=2.0,
            holding_days=30,
        )
        result_365d = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=2.0,
            holding_days=365,
        )
        # Same IL pct
        assert result_30d.il_pct == result_365d.il_pct
        # But annualised IL should be higher for shorter holding period
        assert result_30d.annualized_il_pct > result_365d.annualized_il_pct

    def test_holding_days_zero(self) -> None:
        """Zero holding days means no annualization."""
        result = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=2.0,
            holding_days=0,
        )
        assert result.annualized_il_pct == 0.0


# ===================================================================
# Break-even Analysis
# ===================================================================


class TestBreakEvenAnalysis:
    """Tests for break-even price and time analysis."""

    def test_basic_break_even(self) -> None:
        """Find break-even for a standard constant-product pool."""
        analysis = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT,
            pool_apy=0.05,
        )
        assert isinstance(analysis, BreakEvenAnalysis)
        assert analysis.break_even_price_ratio is not None
        assert analysis.break_even_price_ratio > 1.0
        assert analysis.break_even_days is not None
        assert analysis.break_even_days > 0

    def test_higher_apy_more_tolerant(self) -> None:
        """Higher APY allows larger price moves before breaking even."""
        low_apy = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT, pool_apy=0.02
        )
        high_apy = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT, pool_apy=0.20
        )
        assert low_apy.break_even_price_ratio is not None
        assert high_apy.break_even_price_ratio is not None
        assert high_apy.break_even_price_ratio > low_apy.break_even_price_ratio

    def test_zero_apy(self) -> None:
        """Zero APY means break-even at any price move."""
        analysis = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT, pool_apy=0.0
        )
        assert analysis.break_even_price_ratio is None
        assert analysis.break_even_days is None

    def test_stable_swap_more_tolerant(self) -> None:
        """StableSwap pools tolerate larger depegs before breaking even."""
        cp = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT, pool_apy=0.05
        )
        ss = break_even_analysis(
            pool_type=PoolType.STABLE_SWAP, pool_apy=0.05, amplification=100
        )
        assert cp.break_even_price_ratio is not None
        # StableSwap dampens IL so much that at A=100 the IL may never
        # exceed 5% APY within our search range — break_even can be None.
        # Verify via max_tolerable_move instead.
        if ss.break_even_price_ratio is not None:
            assert ss.break_even_price_ratio > cp.break_even_price_ratio
        else:
            # None means the pool never breaks even → effectively infinite tolerance
            assert ss.max_tolerable_move == float("inf")

    def test_break_even_days_with_volatility(self) -> None:
        """Higher volatility → fewer days to break even."""
        low_vol = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT,
            pool_apy=0.05,
            daily_volatility=0.01,
        )
        high_vol = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT,
            pool_apy=0.05,
            daily_volatility=0.05,
        )
        assert low_vol.break_even_days is not None
        assert high_vol.break_even_days is not None
        assert high_vol.break_even_days < low_vol.break_even_days

    def test_max_tolerable_move(self) -> None:
        """max_tolerable_move equals break_even_price_ratio."""
        analysis = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT, pool_apy=0.05
        )
        if analysis.break_even_price_ratio is not None:
            assert analysis.max_tolerable_move == analysis.break_even_price_ratio


# ===================================================================
# Sensitivity Table
# ===================================================================


class TestSensitivityTable:
    """Tests for the IL sensitivity table generator."""

    def test_basic_table(self) -> None:
        """Generate a basic sensitivity table."""
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        table = il_sensitivity_table(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratios=ratios,
        )
        assert len(table) == len(ratios)
        for entry in table:
            assert isinstance(entry, ILResult)
        # IL at 1.0 should be 0
        assert table[2].il_pct == pytest.approx(0.0, abs=1e-10)

    def test_table_ordering(self) -> None:
        """IL increases as price ratio deviates further from 1.0."""
        ratios = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        table = il_sensitivity_table(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratios=ratios,
        )
        for i in range(1, len(table)):
            assert table[i].il_pct > table[i - 1].il_pct

    def test_weighted_table(self) -> None:
        """Weighted pool sensitivity table."""
        ratios = [1.0, 2.0, 5.0]
        table = il_sensitivity_table(
            pool_type=PoolType.WEIGHTED,
            price_ratios=ratios,
            weights=[0.8, 0.2],
        )
        assert len(table) == 3
        # 80/20 pool should have less IL than 50/50
        for entry in table:
            cp_il = il_constant_product(entry.price_ratio)
            assert entry.il_pct <= cp_il + 1e-10

    def test_stable_swap_table(self) -> None:
        """StableSwap sensitivity table."""
        ratios = [0.95, 0.98, 1.0, 1.02, 1.05]
        table = il_sensitivity_table(
            pool_type=PoolType.STABLE_SWAP,
            price_ratios=ratios,
            amplification=200,
        )
        assert len(table) == 5
        # IL should be very small for these ranges
        for entry in table:
            assert entry.il_pct < 0.01

    def test_concentrated_table(self) -> None:
        """Concentrated liquidity sensitivity table."""
        ratios = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
        table = il_sensitivity_table(
            pool_type=PoolType.CONCENTRATED,
            price_ratios=ratios,
            p_lower=0.8,
            p_upper=1.2,
        )
        assert len(table) == 6
        # All should have IL >= 0
        for entry in table:
            assert entry.il_pct >= 0

    def test_empty_ratios(self) -> None:
        """Empty price ratios returns empty table."""
        table = il_sensitivity_table(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratios=[],
        )
        assert table == []


# ===================================================================
# Integration tests — cross-module
# ===================================================================


class TestILIntegration:
    """Integration tests combining IL with realistic DeFi scenarios."""

    def test_stablecoin_pool_scenario(self) -> None:
        """Stablecoin LP (USDC/USDT) with small depeg — very low IL."""
        result = calculate_il(
            pool_type=PoolType.STABLE_SWAP,
            price_ratio=1.005,  # 0.5% depeg
            amplification=500,
            initial_value_usd=100_000,
            pool_apy=0.03,
            holding_days=365,
        )
        # Very low IL for stablecoins
        assert result.il_pct < 0.0001
        # Net APY should be close to pool APY
        assert result.net_apy > 0.029

    def test_eth_lp_scenario(self) -> None:
        """ETH/USDC LP with moderate volatility."""
        result = calculate_il(
            pool_type=PoolType.CONSTANT_PRODUCT,
            price_ratio=1.5,  # ETH goes up 50%
            initial_value_usd=50_000,
            pool_apy=0.08,
            holding_days=180,
        )
        # IL is real but yield may cover it
        assert result.il_pct > 0
        assert result.hold_value > result.lp_value
        # Net APY check (annualised)
        assert isinstance(result.net_apy, float)

    def test_concentrated_eth_range(self) -> None:
        """Tight ETH concentrated position — higher IL but higher fees."""
        result = calculate_il(
            pool_type=PoolType.CONCENTRATED,
            price_ratio=1.3,
            p_lower=0.9,
            p_upper=1.1,
            initial_value_usd=20_000,
            pool_apy=0.30,  # Concentrated = higher fee APY
            holding_days=90,
        )
        assert result.il_pct > 0
        # Concentrated positions have higher IL but also higher yield
        assert result.net_apy != 0.0  # non-trivial result

    def test_balancer_80_20_eth_lp(self) -> None:
        """Balancer 80ETH/20USDC — less IL than 50/50 for ETH move."""
        result = calculate_il(
            pool_type=PoolType.WEIGHTED,
            price_ratio=2.0,
            weights=[0.8, 0.2],
            initial_value_usd=100_000,
            pool_apy=0.06,
            holding_days=365,
        )
        # 80/20 should have less IL than 50/50
        cp_il = il_constant_product(2.0)
        assert result.il_pct < cp_il
        assert result.net_apy > 0  # Still profitable with 6% APY

    def test_break_even_for_real_pool(self) -> None:
        """Break-even analysis for a real-world-like scenario."""
        analysis = break_even_analysis(
            pool_type=PoolType.CONSTANT_PRODUCT,
            pool_apy=0.04,  # 4% APY (typical stablecoin pool)
            daily_volatility=0.02,
        )
        assert analysis.break_even_price_ratio is not None
        # For 4% APY constant product: IL(r)=0.04 at r≈1.778
        assert 1.5 < analysis.break_even_price_ratio < 2.5
        assert analysis.break_even_days is not None
        assert analysis.break_even_days > 0
