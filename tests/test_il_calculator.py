"""Tests for the Impermanent Loss Calculator module."""

from __future__ import annotations

import math
import pytest

from defi_yield_aggregator.core.il_calculator import (
    BreakEvenResult,
    ILResult,
    ILTimeSeriesPoint,
    LPPosition,
    NetPnLResult,
    calculate_break_even,
    calculate_il,
    calculate_il_timeseries,
    calculate_net_pnl,
    il_constant_product,
    il_sensitivity_table,
    il_weighted_pool,
    max_il_for_pool_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eth_usdc_5050() -> LPPosition:
    """Standard ETH/USDC 50/50 pool."""
    return LPPosition(
        initial_value_usd=10000.0,
        token_weights=[0.5, 0.5],
        initial_prices=[2000.0, 1.0],
    )


@pytest.fixture
def eth_usdc_8020() -> LPPosition:
    """Weighted 80/20 ETH/USDC pool (Balancer-style)."""
    return LPPosition(
        initial_value_usd=10000.0,
        token_weights=[0.8, 0.2],
        initial_prices=[2000.0, 1.0],
        pool_type="weighted",
    )


@pytest.fixture
def wbtc_eth_5050() -> LPPosition:
    """WBTC/ETH 50/50 pool — correlated assets."""
    return LPPosition(
        initial_value_usd=50000.0,
        token_weights=[0.5, 0.5],
        initial_prices=[40000.0, 2000.0],
    )


# ---------------------------------------------------------------------------
# LPPosition validation
# ---------------------------------------------------------------------------


class TestLPPositionValidation:
    def test_valid_position(self, eth_usdc_5050: LPPosition) -> None:
        assert eth_usdc_5050.initial_value_usd == 10000.0
        assert len(eth_usdc_5050.token_weights) == 2

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LPPosition(-1000, [0.5, 0.5], [1.0, 1.0])

    def test_zero_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LPPosition(0, [0.5, 0.5], [1.0, 1.0])

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            LPPosition(1000, [0.5, 0.5], [1.0])

    def test_single_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            LPPosition(1000, [1.0], [1.0])

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            LPPosition(1000, [0.3, 0.3], [1.0, 1.0])

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LPPosition(1000, [-0.5, 1.5], [1.0, 1.0])

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LPPosition(1000, [0.5, 0.5], [-1.0, 1.0])

    def test_unknown_pool_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown pool type"):
            LPPosition(1000, [0.5, 0.5], [1.0, 1.0], pool_type="concentrated")

    def test_weighted_pool_accepted(self) -> None:
        pos = LPPosition(1000, [0.8, 0.2], [1.0, 1.0], pool_type="weighted")
        assert pos.pool_type == "weighted"


# ---------------------------------------------------------------------------
# il_constant_product (classic formula)
# ---------------------------------------------------------------------------


class TestILConstantProduct:
    def test_no_price_change(self) -> None:
        assert il_constant_product(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_2x_price_increase(self) -> None:
        il = il_constant_product(2.0)
        # Classic result: ~-5.72%
        assert il == pytest.approx(-0.05719, abs=1e-4)

    def test_symmetric(self) -> None:
        """IL is symmetric: 2x up and 0.5x down give the same IL."""
        assert il_constant_product(2.0) == pytest.approx(
            il_constant_product(0.5), abs=1e-10
        )

    def test_3x_price(self) -> None:
        il = il_constant_product(3.0)
        assert il == pytest.approx(-0.13397, abs=1e-4)

    def test_5x_price(self) -> None:
        il = il_constant_product(5.0)
        assert il == pytest.approx(-0.25464, abs=1e-4)

    def test_10x_price(self) -> None:
        il = il_constant_product(10.0)
        assert il == pytest.approx(-0.42504, abs=1e-4)

    def test_very_small_ratio(self) -> None:
        il = il_constant_product(0.01)
        assert il < -0.1  # Significant loss
        assert il > -1.0  # But not total

    def test_negative_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            il_constant_product(-1.0)

    def test_zero_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            il_constant_product(0.0)

    def test_monotonic_increase(self) -> None:
        """IL magnitude increases as price diverges further from 1.0."""
        ratios = [1.1, 1.5, 2.0, 3.0, 5.0, 10.0]
        ils = [il_constant_product(r) for r in ratios]
        # IL should be decreasing (more negative)
        for i in range(len(ils) - 1):
            assert ils[i] > ils[i + 1]


# ---------------------------------------------------------------------------
# il_weighted_pool
# ---------------------------------------------------------------------------


class TestILWeightedPool:
    def test_equal_weight_matches_constant_product(self) -> None:
        """50/50 weighted pool should match constant product formula."""
        ratio = 2.0
        weighted = il_weighted_pool([ratio, 1.0], [0.5, 0.5])
        constant = il_constant_product(ratio)
        assert weighted == pytest.approx(constant, abs=1e-10)

    def test_no_change_zero_il(self) -> None:
        assert il_weighted_pool([1.0, 1.0], [0.5, 0.5]) == pytest.approx(0.0)

    def test_8020_pool_less_il(self) -> None:
        """80/20 pool should have less IL than 50/50 for same price change."""
        ratio = 3.0
        il_5050 = il_weighted_pool([ratio, 1.0], [0.5, 0.5])
        il_8020 = il_weighted_pool([ratio, 1.0], [0.8, 0.2])
        # 80/20 should have smaller magnitude IL
        assert abs(il_8020) < abs(il_5050)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            il_weighted_pool([1.0, 1.0, 1.0], [0.5, 0.5])

    def test_three_token_pool(self) -> None:
        """Test a 3-token equal-weight pool."""
        il = il_weighted_pool([1.5, 1.0, 0.8], [1 / 3, 1 / 3, 1 / 3])
        assert il < 0  # Any divergence causes IL
        assert il > -0.1  # But moderate for small changes


# ---------------------------------------------------------------------------
# calculate_il
# ---------------------------------------------------------------------------


class TestCalculateIL:
    def test_no_price_change(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_il(eth_usdc_5050, [2000.0, 1.0])
        assert result.il_pct == pytest.approx(0.0, abs=1e-6)
        assert result.hold_value_usd == pytest.approx(10000.0)
        assert result.lp_value_usd == pytest.approx(10000.0)

    def test_eth_doubles(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_il(eth_usdc_5050, [4000.0, 1.0])
        # price_ratio = 1.0/2.0 = 0.5 (USDC vs ETH)
        assert result.il_pct < 0
        assert result.lp_value_usd < result.hold_value_usd
        # Hold value: 50% ETH doubled + 50% USDC same = 0.5*20000 + 0.5*10000 = 15000
        assert result.hold_value_usd == pytest.approx(15000.0)

    def test_eth_halves(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_il(eth_usdc_5050, [1000.0, 1.0])
        assert result.il_pct < 0
        # Hold: 0.5*5000 + 0.5*10000 = 7500
        assert result.hold_value_usd == pytest.approx(7500.0)

    def test_weighted_pool(self, eth_usdc_8020: LPPosition) -> None:
        result = calculate_il(eth_usdc_8020, [4000.0, 1.0])
        assert result.il_pct < 0
        # IL should be less than 50/50 pool
        result_5050 = calculate_il(
            LPPosition(10000, [0.5, 0.5], [2000.0, 1.0]), [4000.0, 1.0]
        )
        assert abs(result.il_pct) < abs(result_5050.il_pct)

    def test_mismatched_prices_rejected(self, eth_usdc_5050: LPPosition) -> None:
        with pytest.raises(ValueError, match="Expected 2 prices"):
            calculate_il(eth_usdc_5050, [2000.0])

    def test_negative_price_rejected(self, eth_usdc_5050: LPPosition) -> None:
        with pytest.raises(ValueError, match="positive"):
            calculate_il(eth_usdc_5050, [-1.0, 1.0])

    def test_price_changes_reported(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_il(eth_usdc_5050, [4000.0, 1.0])
        assert result.price_changes[0] == pytest.approx(2.0)  # ETH doubled
        assert result.price_changes[1] == pytest.approx(1.0)  # USDC same


# ---------------------------------------------------------------------------
# calculate_net_pnl
# ---------------------------------------------------------------------------


class TestCalculateNetPnL:
    def test_yield_offsets_il(self, eth_usdc_5050: LPPosition) -> None:
        """Yield should partially or fully offset IL."""
        # 2x price change → ~5.7% IL
        pnl_no_yield = calculate_net_pnl(
            eth_usdc_5050, [4000.0, 1.0], apr_yield=0.0, days_held=365
        )
        pnl_with_yield = calculate_net_pnl(
            eth_usdc_5050, [4000.0, 1.0], apr_yield=0.15, days_held=365
        )
        assert pnl_with_yield.net_pnl_usd > pnl_no_yield.net_pnl_usd

    def test_fees_contribute(self, eth_usdc_5050: LPPosition) -> None:
        pnl_no_fees = calculate_net_pnl(
            eth_usdc_5050, [2500.0, 1.0], daily_fee_apr=0.0, days_held=90
        )
        pnl_with_fees = calculate_net_pnl(
            eth_usdc_5050, [2500.0, 1.0], daily_fee_apr=0.20, days_held=90
        )
        assert pnl_with_fees.net_pnl_usd > pnl_no_fees.net_pnl_usd

    def test_proportional_to_days(self, eth_usdc_5050: LPPosition) -> None:
        pnl_30 = calculate_net_pnl(
            eth_usdc_5050, [2000.0, 1.0], apr_yield=0.10, days_held=30
        )
        pnl_365 = calculate_net_pnl(
            eth_usdc_5050, [2000.0, 1.0], apr_yield=0.10, days_held=365
        )
        # More days → more yield
        assert pnl_365.yield_earned_usd > pnl_30.yield_earned_usd

    def test_zero_days_rejected(self, eth_usdc_5050: LPPosition) -> None:
        with pytest.raises(ValueError, match="positive"):
            calculate_net_pnl(eth_usdc_5050, [2000.0, 1.0], days_held=0)

    def test_no_change_pure_yield(self, eth_usdc_5050: LPPosition) -> None:
        """With no price change, net P&L = yield + fees."""
        pnl = calculate_net_pnl(
            eth_usdc_5050, [2000.0, 1.0], apr_yield=0.10, daily_fee_apr=0.05, days_held=365
        )
        assert pnl.il_result.il_pct == pytest.approx(0.0, abs=1e-6)
        assert pnl.yield_earned_usd == pytest.approx(1000.0)
        assert pnl.net_pnl_usd > 0

    def test_annualized_return(self, eth_usdc_5050: LPPosition) -> None:
        pnl = calculate_net_pnl(
            eth_usdc_5050, [2000.0, 1.0], apr_yield=0.10, days_held=365
        )
        assert pnl.annualized_return_pct == pytest.approx(0.10, abs=1e-4)


# ---------------------------------------------------------------------------
# calculate_break_even
# ---------------------------------------------------------------------------


class TestCalculateBreakEven:
    def test_finds_ratio(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_break_even(eth_usdc_5050, apr_yield=0.10, days_held=365)
        # Break-even is downside: price must drop below this ratio to lose money
        assert 0.0 < result.price_ratio < 1.0
        assert result.il_at_break_even < 0
        # For 10% yield: break-even at r = (1-0.1)^2 = 0.81
        assert result.price_ratio == pytest.approx(0.81, abs=0.01)

    def test_higher_yield_lower_breakeven(self, eth_usdc_5050: LPPosition) -> None:
        low_yield = calculate_break_even(eth_usdc_5050, apr_yield=0.05, days_held=365)
        high_yield = calculate_break_even(eth_usdc_5050, apr_yield=0.20, days_held=365)
        # Higher yield → lower break-even ratio (can tolerate more downside)
        assert high_yield.price_ratio < low_yield.price_ratio

    def test_profitable_at_no_change(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_break_even(
            eth_usdc_5050, apr_yield=0.10, days_held=365, current_price_ratio=1.0
        )
        assert result.is_profitable_at_current is True

    def test_not_profitable_at_extreme(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_break_even(
            eth_usdc_5050, apr_yield=0.01, days_held=30, current_price_ratio=0.05
        )
        assert result.is_profitable_at_current is False

    def test_three_token_rejected(self) -> None:
        pos = LPPosition(1000, [1 / 3, 1 / 3, 1 / 3], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="exactly 2 tokens"):
            calculate_break_even(pos)

    def test_min_yield_apr(self, eth_usdc_5050: LPPosition) -> None:
        result = calculate_break_even(
            eth_usdc_5050, current_price_ratio=0.5, days_held=365
        )
        assert result.min_yield_apr_needed > 0  # Need yield to offset IL at 2x


# ---------------------------------------------------------------------------
# calculate_il_timeseries
# ---------------------------------------------------------------------------


class TestCalculateILTimeSeries:
    def test_stable_price(self, eth_usdc_5050: LPPosition) -> None:
        ratios = [1.0] * 30
        ts = calculate_il_timeseries(eth_usdc_5050, ratios)
        assert len(ts) == 30
        for point in ts:
            assert point.il_pct == pytest.approx(0.0, abs=1e-6)
            assert point.net_pnl_pct >= 0  # Pure yield, no IL

    def test_increasing_loss(self, eth_usdc_5050: LPPosition) -> None:
        """IL magnitude should increase as price diverges."""
        ratios = [1.0 + 0.1 * i for i in range(10)]  # 1.0, 1.1, ..., 1.9
        ts = calculate_il_timeseries(eth_usdc_5050, ratios)
        # IL should become more negative over time
        for i in range(1, len(ts)):
            assert ts[i].il_pct <= ts[i - 1].il_pct

    def test_yield_accumulates(self, eth_usdc_5050: LPPosition) -> None:
        ratios = [1.0] * 10
        ts = calculate_il_timeseries(eth_usdc_5050, ratios, apr_yield=0.365)
        for i in range(1, len(ts)):
            assert ts[i].cumulative_yield_pct > ts[i - 1].cumulative_yield_pct

    def test_two_token_required(self) -> None:
        pos = LPPosition(1000, [1 / 3, 1 / 3, 1 / 3], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="exactly 2 tokens"):
            calculate_il_timeseries(pos, [1.0])

    def test_day_numbering(self, eth_usdc_5050: LPPosition) -> None:
        ratios = [1.0, 1.1, 1.2]
        ts = calculate_il_timeseries(eth_usdc_5050, ratios)
        assert ts[0].day == 1
        assert ts[1].day == 2
        assert ts[2].day == 3


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_max_il_constant_product(self) -> None:
        max_il = max_il_for_pool_type("constant_product")
        assert max_il < 0
        assert max_il > -1.0  # Never total loss

    def test_sensitivity_table_default(self) -> None:
        table = il_sensitivity_table()
        assert len(table) > 0
        for ratio, il_pct, diff_pct in table:
            assert ratio > 0
            assert il_pct <= 0  # IL is always negative or zero
            # At ratio=1.0, IL should be 0
            if abs(ratio - 1.0) < 1e-6:
                assert il_pct == pytest.approx(0.0, abs=1e-3)

    def test_sensitivity_table_custom(self) -> None:
        table = il_sensitivity_table([1.0, 2.0, 5.0])
        assert len(table) == 3
        assert table[0][1] == pytest.approx(0.0, abs=1e-3)  # No change → no IL

    def test_sensitivity_table_symmetric(self) -> None:
        """IL% should be same for reciprocal ratios."""
        table = il_sensitivity_table([2.0, 0.5])
        assert table[0][1] == pytest.approx(table[1][1], abs=1e-3)
