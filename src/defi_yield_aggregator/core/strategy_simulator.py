"""Yield farming strategy simulator / backtester.

Simulates portfolio performance over historical APY data, comparing
different allocation strategies (greedy, Kelly, risk parity) with
configurable rebalancing frequency, gas costs, and market variability.

Supports:
- Deterministic backtests against historical APY sequences
- Monte Carlo simulation with configurable APY noise
- Performance metrics: total return, max drawdown, Sharpe, Sortino
- Multi-strategy comparison reports
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from defi_yield_aggregator.core.models import (
    Config,
    PoolInfo,
    RiskScore,
)
from defi_yield_aggregator.core.optimizer import (
    AllocationStrategy,
    PortfolioOptimizer,
)
from defi_yield_aggregator.core.risk_engine import RiskEngine


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a strategy simulation run.

    Attributes:
        initial_capital: Starting portfolio value in USD.
        duration_days: Simulation duration in calendar days.
        rebalance_frequency_days: How often to rebalance (0 = never).
        gas_cost_per_rebalance: Fixed gas cost in USD per rebalance event.
        slippage_bps: Slippage in basis points on each trade.
        apy_noise_std: Standard deviation of APY noise for Monte Carlo
            (0.0 = deterministic, e.g. 0.005 = 0.5% std dev noise).
        seed: Random seed for reproducibility. None = non-deterministic.
    """

    initial_capital: float = 10_000.0
    duration_days: int = 365
    rebalance_frequency_days: int = 7
    gas_cost_per_rebalance: float = 5.0
    slippage_bps: float = 10.0
    apy_noise_std: float = 0.0
    seed: Optional[int] = None


@dataclass
class DailySnapshot:
    """Portfolio state on a single simulation day.

    Attributes:
        day: Day number (0-indexed).
        date: Simulated calendar date.
        portfolio_value: Total portfolio value in USD.
        daily_yield: Yield earned that day in USD.
        allocations: Current pool allocation percentages.
        rebalanced: Whether a rebalance occurred this day.
    """

    day: int
    date: datetime
    portfolio_value: float
    daily_yield: float
    allocations: dict[str, float]
    rebalanced: bool


@dataclass
class SimulationResult:
    """Result of a single strategy simulation.

    Attributes:
        strategy: The allocation strategy used.
        config: Simulation configuration.
        snapshots: Daily portfolio snapshots.
        total_return_pct: Total return as a percentage.
        annualized_return_pct: Annualized return percentage.
        max_drawdown_pct: Maximum peak-to-trough drawdown.
        sharpe_ratio: Risk-adjusted return (annualized).
        sortino_ratio: Downside-risk-adjusted return (annualized).
        total_gas_cost: Cumulative gas and slippage costs.
        num_rebalances: Number of rebalance events.
        final_value: Final portfolio value in USD.
        volatility_annualized: Annualized daily return volatility.
    """

    strategy: AllocationStrategy
    config: SimulationConfig
    snapshots: list[DailySnapshot]
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_gas_cost: float = 0.0
    num_rebalances: int = 0
    final_value: float = 0.0
    volatility_annualized: float = 0.0


@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo simulation.

    Attributes:
        strategy: The allocation strategy used.
        num_runs: Number of Monte Carlo iterations.
        mean_return_pct: Mean total return across runs.
        median_return_pct: Median total return.
        std_return_pct: Standard deviation of returns.
        percentile_5: 5th percentile return (worst case).
        percentile_25: 25th percentile return.
        percentile_75: 75th percentile return.
        percentile_95: 95th percentile return (best case).
        mean_max_drawdown: Mean maximum drawdown across runs.
        mean_sharpe: Mean Sharpe ratio across runs.
        probability_of_loss: Fraction of runs with negative return.
    """

    strategy: AllocationStrategy
    num_runs: int
    mean_return_pct: float = 0.0
    median_return_pct: float = 0.0
    std_return_pct: float = 0.0
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    mean_max_drawdown: float = 0.0
    mean_sharpe: float = 0.0
    probability_of_loss: float = 0.0


@dataclass
class ComparisonReport:
    """Side-by-side comparison of multiple strategy simulations.

    Attributes:
        results: Simulation results keyed by strategy name.
        best_return: Strategy with highest total return.
        best_risk_adjusted: Strategy with highest Sharpe ratio.
        lowest_drawdown: Strategy with lowest max drawdown.
    """

    results: dict[str, SimulationResult]
    best_return: str = ""
    best_risk_adjusted: str = ""
    lowest_drawdown: str = ""


def _generate_apy_sequence(
    base_apy: float,
    num_days: int,
    noise_std: float,
    rng: random.Random,
    mean_reversion_speed: float = 0.05,
    long_term_mean: Optional[float] = None,
) -> list[float]:
    """Generate a daily APY sequence with mean-reverting noise.

    Uses an Ornstein-Uhlenbeck process to model APY fluctuations:
        dAPY = theta * (mu - APY) * dt + sigma * dW

    This produces realistic APY paths that fluctuate around a mean
    without wandering off to unrealistic values.

    Args:
        base_apy: Starting APY value (decimal, e.g. 0.05 for 5%).
        num_days: Number of daily observations to generate.
        noise_std: Daily noise standard deviation.
        rng: Random number generator for reproducibility.
        mean_reversion_speed: How quickly APY reverts to mean (0-1).
        long_term_mean: Target mean for reversion. Defaults to base_apy.

    Returns:
        List of daily APY values (length = num_days).
    """
    if noise_std <= 0:
        return [base_apy] * num_days

    mu = long_term_mean if long_term_mean is not None else base_apy
    theta = mean_reversion_speed
    sigma = noise_std

    apys = [base_apy]
    current = base_apy

    for _ in range(num_days - 1):
        # Ornstein-Uhlenbeck step (dt = 1 day)
        drift = theta * (mu - current)
        shock = sigma * rng.gauss(0, 1)
        current = current + drift + shock
        # Floor at 0 (negative APY is unrealistic for most DeFi pools)
        current = max(current, 0.0)
        apys.append(current)

    return apys


def _apply_slippage_and_gas(
    portfolio_value: float,
    gas_cost: float,
    slippage_bps: float,
) -> float:
    """Deduct rebalancing costs from portfolio value.

    Args:
        portfolio_value: Current portfolio value.
        gas_cost: Fixed gas cost in USD.
        slippage_bps: Slippage in basis points.

    Returns:
        Portfolio value after costs.
    """
    slippage_cost = portfolio_value * (slippage_bps / 10_000)
    return max(portfolio_value - gas_cost - slippage_cost, 0.0)


class StrategySimulator:
    """Simulates yield farming strategies over historical or synthetic data.

    The simulator models daily yield accrual, periodic rebalancing,
    gas costs, and slippage. It can run deterministic backtests or
    Monte Carlo simulations with APY noise.

    Args:
        pools: Available yield farming pools.
        config: Simulation parameters.

    Example::

        pools = [pool1, pool2, pool3]
        sim = StrategySimulator(pools=pools)
        result = sim.run(AllocationStrategy.KELLY)
        print(f"Return: {result.total_return_pct:.2f}%")
    """

    def __init__(
        self,
        pools: list[PoolInfo],
        config: Optional[SimulationConfig] = None,
        pool_config: Optional[Config] = None,
    ) -> None:
        if not pools:
            raise ValueError("At least one pool is required for simulation")
        self._pools = pools
        self._sim_config = config or SimulationConfig()
        self._pool_config = pool_config or Config()
        self._risk_engine = RiskEngine()
        self._risk_scores = self._risk_engine.score_pools(pools)

    def _get_base_apys(self) -> dict[str, float]:
        """Map pool_id to base APY for simulation."""
        return {p.pool_id: p.apy for p in self._pools}

    def _daily_rate(self, annual_apy: float) -> float:
        """Convert annual APY to daily compound rate."""
        return (1 + annual_apy) ** (1 / 365) - 1

    def run(
        self,
        strategy: AllocationStrategy,
        historical_apys: Optional[dict[str, list[float]]] = None,
    ) -> SimulationResult:
        """Run a deterministic simulation for a single strategy.

        Args:
            strategy: Allocation strategy to simulate.
            historical_apys: Optional dict mapping pool_id to a list of
                daily APY values. If None, generates synthetic APY paths
                from pool base APYs using the simulation config's noise.

        Returns:
            Simulation result with daily snapshots and performance metrics.
        """
        rng = random.Random(self._sim_config.seed)
        num_days = self._sim_config.duration_days
        base_apys = self._get_base_apys()

        # Build daily APY sequences per pool
        daily_apys: dict[str, list[float]] = {}
        for pool_id, base_apy in base_apys.items():
            if historical_apys and pool_id in historical_apys:
                apys = historical_apys[pool_id]
                # Pad or truncate to match duration
                if len(apys) < num_days:
                    apys = apys + [apys[-1]] * (num_days - len(apys))
                daily_apys[pool_id] = apys[:num_days]
            else:
                daily_apys[pool_id] = _generate_apy_sequence(
                    base_apy=base_apy,
                    num_days=num_days,
                    noise_std=self._sim_config.apy_noise_std,
                    rng=rng,
                )

        # Initial allocation
        optimizer = PortfolioOptimizer(
            config=self._pool_config,
            strategy=strategy,
        )
        portfolio = optimizer.optimize(self._pools, self._sim_config.initial_capital)

        # Build allocation map: pool_id -> weight
        alloc_map: dict[str, float] = {}
        for a in portfolio.allocations:
            alloc_map[a.pool_id] = a.allocation_pct

        # If optimizer produced no allocations, fall back to equal weight
        if not alloc_map:
            eligible = [
                p for p in self._pools
                if p.tvl_usd >= self._pool_config.min_tvl_usd
            ]
            if not eligible:
                eligible = self._pools[:1]
            weight = 1.0 / len(eligible)
            alloc_map = {p.pool_id: weight for p in eligible}

        # Simulate day by day
        portfolio_value = self._sim_config.initial_capital
        snapshots: list[DailySnapshot] = []
        total_gas = 0.0
        num_rebalances = 0
        start_date = datetime.utcnow()

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            rebalanced = False

            # Check if we should rebalance
            if (
                self._sim_config.rebalance_frequency_days > 0
                and day > 0
                and day % self._sim_config.rebalance_frequency_days == 0
            ):
                # Re-optimize with current pool APYs
                current_pools = []
                for pool in self._pools:
                    if pool.pool_id in daily_apys:
                        # Create a copy with current APY
                        updated = pool.model_copy(
                            update={"apy": daily_apys[pool.pool_id][day]}
                        )
                        current_pools.append(updated)
                    else:
                        current_pools.append(pool)

                new_portfolio = optimizer.optimize(
                    current_pools, portfolio_value
                )
                if new_portfolio.allocations:
                    alloc_map = {
                        a.pool_id: a.allocation_pct
                        for a in new_portfolio.allocations
                    }

                # Apply gas and slippage costs
                portfolio_value = _apply_slippage_and_gas(
                    portfolio_value,
                    self._sim_config.gas_cost_per_rebalance,
                    self._sim_config.slippage_bps,
                )
                gas_now = (
                    self._sim_config.gas_cost_per_rebalance
                    + portfolio_value * (self._sim_config.slippage_bps / 10_000)
                )
                total_gas += gas_now
                num_rebalances += 1
                rebalanced = True

            # Calculate daily yield
            daily_yield = 0.0
            for pool_id, weight in alloc_map.items():
                if pool_id in daily_apys:
                    pool_apy = daily_apys[pool_id][day]
                    pool_value = portfolio_value * weight
                    daily_yield += pool_value * self._daily_rate(pool_apy)

            portfolio_value += daily_yield

            snapshots.append(
                DailySnapshot(
                    day=day,
                    date=current_date,
                    portfolio_value=round(portfolio_value, 2),
                    daily_yield=round(daily_yield, 4),
                    allocations=dict(alloc_map),
                    rebalanced=rebalanced,
                )
            )

        # Calculate performance metrics
        initial = self._sim_config.initial_capital
        final = portfolio_value
        total_return = ((final - initial) / initial) * 100 if initial > 0 else 0.0
        years = num_days / 365
        ann_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and initial > 0 and final > 0 else 0.0

        # Daily returns for Sharpe/Sortino
        daily_returns: list[float] = []
        for i in range(1, len(snapshots)):
            prev_val = snapshots[i - 1].portfolio_value
            if prev_val > 0:
                daily_returns.append(
                    (snapshots[i].portfolio_value - prev_val) / prev_val
                )

        # Max drawdown
        peak = initial
        max_dd = 0.0
        for snap in snapshots:
            if snap.portfolio_value > peak:
                peak = snap.portfolio_value
            dd = (peak - snap.portfolio_value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        max_dd_pct = max_dd * 100

        # Sharpe ratio (annualized, risk-free rate = 0 for simplicity)
        if daily_returns and len(daily_returns) > 1:
            mean_daily = statistics.mean(daily_returns)
            std_daily = statistics.stdev(daily_returns)
            vol_annual = std_daily * math.sqrt(365)
            sharpe = (mean_daily * 365) / vol_annual if vol_annual > 0 else 0.0

            # Sortino ratio (downside deviation only)
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns and len(negative_returns) > 1:
                downside_dev = statistics.stdev(negative_returns)
                downside_annual = downside_dev * math.sqrt(365)
                sortino = (
                    (mean_daily * 365) / downside_annual
                    if downside_annual > 0
                    else 0.0
                )
            else:
                sortino = sharpe * 1.5  # No negative days = better than Sharpe
        else:
            vol_annual = 0.0
            sharpe = 0.0
            sortino = 0.0

        return SimulationResult(
            strategy=strategy,
            config=self._sim_config,
            snapshots=snapshots,
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(ann_return, 4),
            max_drawdown_pct=round(max_dd_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            total_gas_cost=round(total_gas, 2),
            num_rebalances=num_rebalances,
            final_value=round(final, 2),
            volatility_annualized=round(vol_annual, 6),
        )

    def run_all_strategies(
        self,
        historical_apys: Optional[dict[str, list[float]]] = None,
    ) -> ComparisonReport:
        """Run simulation for all allocation strategies and compare.

        Args:
            historical_apys: Optional historical APY data per pool.

        Returns:
            Comparison report with results for each strategy.
        """
        results: dict[str, SimulationResult] = {}

        for strategy in AllocationStrategy:
            # Each strategy gets a fresh RNG with offset seed
            base_seed = self._sim_config.seed
            strat_config = SimulationConfig(
                initial_capital=self._sim_config.initial_capital,
                duration_days=self._sim_config.duration_days,
                rebalance_frequency_days=self._sim_config.rebalance_frequency_days,
                gas_cost_per_rebalance=self._sim_config.gas_cost_per_rebalance,
                slippage_bps=self._sim_config.slippage_bps,
                apy_noise_std=self._sim_config.apy_noise_std,
                seed=base_seed + strategy.value.__hash__() % 10000 if base_seed is not None else None,
            )
            sim = StrategySimulator(
                pools=self._pools,
                config=strat_config,
                pool_config=self._pool_config,
            )
            results[strategy.value] = sim.run(strategy, historical_apys)

        # Determine bests
        best_return = max(results, key=lambda k: results[k].total_return_pct)
        best_sharpe = max(results, key=lambda k: results[k].sharpe_ratio)
        lowest_dd = min(results, key=lambda k: results[k].max_drawdown_pct)

        return ComparisonReport(
            results=results,
            best_return=best_return,
            best_risk_adjusted=best_sharpe,
            lowest_drawdown=lowest_dd,
        )

    def monte_carlo(
        self,
        strategy: AllocationStrategy,
        num_runs: int = 100,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation with random APY noise.

        Executes the same strategy multiple times with different
        random APY paths to estimate the distribution of outcomes.

        Args:
            strategy: Allocation strategy to simulate.
            num_runs: Number of Monte Carlo iterations.

        Returns:
            Aggregated Monte Carlo statistics.
        """
        returns: list[float] = []
        drawdowns: list[float] = []
        sharpes: list[float] = []

        base_seed = self._sim_config.seed

        for i in range(num_runs):
            run_seed = (base_seed + i) if base_seed is not None else None
            run_config = SimulationConfig(
                initial_capital=self._sim_config.initial_capital,
                duration_days=self._sim_config.duration_days,
                rebalance_frequency_days=self._sim_config.rebalance_frequency_days,
                gas_cost_per_rebalance=self._sim_config.gas_cost_per_rebalance,
                slippage_bps=self._sim_config.slippage_bps,
                apy_noise_std=self._sim_config.apy_noise_std
                or 0.01,  # Ensure some noise
                seed=run_seed,
            )
            sim = StrategySimulator(
                pools=self._pools,
                config=run_config,
                pool_config=self._pool_config,
            )
            result = sim.run(strategy)
            returns.append(result.total_return_pct)
            drawdowns.append(result.max_drawdown_pct)
            sharpes.append(result.sharpe_ratio)

        returns.sort()
        n = len(returns)
        prob_loss = sum(1 for r in returns if r < 0) / n if n > 0 else 0.0

        return MonteCarloResult(
            strategy=strategy,
            num_runs=num_runs,
            mean_return_pct=round(statistics.mean(returns), 4) if returns else 0.0,
            median_return_pct=round(statistics.median(returns), 4) if returns else 0.0,
            std_return_pct=round(statistics.stdev(returns), 4) if len(returns) > 1 else 0.0,
            percentile_5=round(returns[int(n * 0.05)], 4) if n > 0 else 0.0,
            percentile_25=round(returns[int(n * 0.25)], 4) if n > 0 else 0.0,
            percentile_75=round(returns[int(n * 0.75)], 4) if n > 0 else 0.0,
            percentile_95=round(returns[int(n * 0.95)], 4) if n > 0 else 0.0,
            mean_max_drawdown=round(statistics.mean(drawdowns), 4) if drawdowns else 0.0,
            mean_sharpe=round(statistics.mean(sharpes), 4) if sharpes else 0.0,
            probability_of_loss=round(prob_loss, 4),
        )
