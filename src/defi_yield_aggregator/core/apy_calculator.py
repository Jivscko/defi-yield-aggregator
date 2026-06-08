"""APY calculation utilities with compound interest formulas."""

from __future__ import annotations

import math
from typing import Optional


def apy_to_apr(apy: float, compounding_periods: int = 365) -> float:
    """Convert APY to APR given compounding frequency.

    Args:
        apy: Annual Percentage Yield as decimal (e.g., 0.05 for 5%)
        compounding_periods: Number of compounding periods per year.

    Returns:
        APR as decimal.
    """
    if apy < 0:
        raise ValueError(f"APY must be non-negative, got {apy}")
    if compounding_periods <= 0:
        raise ValueError(f"Compounding periods must be positive, got {compounding_periods}")
    return compounding_periods * ((1 + apy) ** (1 / compounding_periods) - 1)


def apr_to_apy(apr: float, compounding_periods: int = 365) -> float:
    """Convert APR to APY given compounding frequency.

    Args:
        apr: Annual Percentage Rate as decimal.
        compounding_periods: Number of compounding periods per year.

    Returns:
        APY as decimal.
    """
    if compounding_periods <= 0:
        raise ValueError(f"Compounding periods must be positive, got {compounding_periods}")
    return (1 + apr / compounding_periods) ** compounding_periods - 1


def future_value(
    principal: float,
    apy: float,
    years: float,
    compounding_periods: int = 365,
) -> float:
    """Calculate future value with compound interest.

    Args:
        principal: Initial investment amount.
        apy: Annual Percentage Yield as decimal.
        years: Investment duration in years.
        compounding_periods: Compounding frequency per year.

    Returns:
        Future value of the investment.
    """
    if principal < 0:
        raise ValueError(f"Principal must be non-negative, got {principal}")
    if years < 0:
        raise ValueError(f"Years must be non-negative, got {years}")
    return principal * (1 + apy / compounding_periods) ** (compounding_periods * years)


def daily_yield(principal: float, apy: float) -> float:
    """Calculate daily yield from an APY.

    Args:
        principal: Investment amount in USD.
        apy: Annual Percentage Yield as decimal.

    Returns:
        Expected daily earnings in USD.
    """
    daily_rate = (1 + apy) ** (1 / 365) - 1
    return principal * daily_rate


def effective_apy(
    base_apy: float,
    reward_apy: float = 0.0,
    deposit_fee: float = 0.0,
    withdrawal_fee: float = 0.0,
    holding_period_days: int = 365,
) -> float:
    """Calculate effective APY accounting for fees and rewards.

    Args:
        base_apy: Base lending/LP APY.
        reward_apy: Additional reward token APY.
        deposit_fee: One-time deposit fee as fraction.
        withdrawal_fee: One-time withdrawal fee as fraction.
        holding_period_days: Expected holding period in days.

    Returns:
        Effective net APY after fees.
    """
    if holding_period_days <= 0:
        raise ValueError(f"Holding period must be positive, got {holding_period_days}")

    total_apy = base_apy + reward_apy
    period_years = holding_period_days / 365

    # Net value after deposit fee, growth, and withdrawal fee
    net_multiplier = (1 - deposit_fee) * (1 + total_apy) ** period_years * (1 - withdrawal_fee)
    effective_annual = net_multiplier ** (1 / period_years) - 1

    return max(effective_annual, 0.0)


def impermanent_loss(price_ratio: float) -> float:
    """Calculate impermanent loss for a 50/50 LP position.

    Args:
        price_ratio: New price / initial price of one asset vs the other.

    Returns:
        Impermanent loss as a negative fraction (e.g., -0.05 = 5% loss).
    """
    if price_ratio <= 0:
        raise ValueError(f"Price ratio must be positive, got {price_ratio}")
    il = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
    return il
