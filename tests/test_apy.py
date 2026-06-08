"""Tests for APY calculator."""

import math

import pytest

from defi_yield_aggregator.core.apy_calculator import (
    apy_to_apr,
    apr_to_apy,
    daily_yield,
    effective_apy,
    future_value,
    impermanent_loss,
)


class TestApyConversions:
    """Test APY/APR conversions."""

    def test_apy_to_apr_zero(self) -> None:
        assert apy_to_apr(0.0) == 0.0

    def test_apy_to_apr_roundtrip(self) -> None:
        original_apy = 0.05
        apr = apy_to_apr(original_apy, compounding_periods=365)
        recovered_apy = apr_to_apy(apr, compounding_periods=365)
        assert abs(recovered_apy - original_apy) < 1e-10

    def test_apr_to_apy_continuous(self) -> None:
        # With 365 periods, APY should be slightly higher than APR
        apr = 0.05
        apy = apr_to_apy(apr, 365)
        assert apy > apr

    def test_apr_to_apy_monthly(self) -> None:
        apy = apr_to_apy(0.12, 12)
        # 12% APR monthly compounding ≈ 12.68% APY
        assert abs(apy - 0.126825) < 0.001

    def test_apy_to_apr_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            apy_to_apr(-0.01)

    def test_apy_to_apr_zero_periods_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            apy_to_apr(0.05, compounding_periods=0)


class TestFutureValue:
    """Test future value calculations."""

    def test_zero_apy(self) -> None:
        assert future_value(1000, 0.0, 1.0) == 1000.0

    def test_one_year_5_percent(self) -> None:
        fv = future_value(10000, 0.05, 1.0)
        # With daily compounding, ~$10,512.67
        assert abs(fv - 10512.67) < 1

    def test_five_years_compound(self) -> None:
        fv = future_value(10000, 0.10, 5.0)
        # With daily compounding, result is slightly higher than simple annual
        expected = 10000 * (1 + 0.10 / 365) ** (365 * 5)
        assert abs(fv - expected) < 1

    def test_negative_principal_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            future_value(-100, 0.05, 1.0)

    def test_zero_years(self) -> None:
        assert future_value(1000, 0.05, 0.0) == 1000.0


class TestDailyYield:
    """Test daily yield calculations."""

    def test_zero_apy(self) -> None:
        assert daily_yield(10000, 0.0) == 0.0

    def test_positive_yield(self) -> None:
        dy = daily_yield(100000, 0.05)
        assert dy > 0
        # ~$13.36/day at 5% on $100k
        assert 10 < dy < 20


class TestEffectiveApy:
    """Test effective APY with fees."""

    def test_no_fees(self) -> None:
        e = effective_apy(0.05, 0.02, 0.0, 0.0, 365)
        assert e > 0.05

    def test_with_fees(self) -> None:
        e = effective_apy(0.10, 0.0, 0.01, 0.01, 365)
        assert e < 0.10

    def test_zero_holding_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            effective_apy(0.05, 0.0, 0.0, 0.0, 0)


class TestImpermanentLoss:
    """Test impermanent loss calculations."""

    def test_no_price_change(self) -> None:
        il = impermanent_loss(1.0)
        assert abs(il) < 1e-10

    def test_price_doubles(self) -> None:
        il = impermanent_loss(2.0)
        # IL should be ~-5.7% when price doubles
        assert -0.10 < il < -0.04
        assert abs(il - (-0.0572)) < 0.001

    def test_price_5x(self) -> None:
        il = impermanent_loss(5.0)
        assert il < -0.02  # >2% IL

    def test_zero_price_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            impermanent_loss(0.0)
