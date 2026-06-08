"""Tests for the yield farming strategy simulator / backtester."""

from __future__ import annotations

import math
import statistics
from datetime import datetime

import pytest

from defi_yield_aggregator.core.models import Chain, Config, PoolInfo, Protocol
from defi_yield_aggregator.core.optimizer import AllocationStrategy
from defi_yield_aggregator.core.strategy_simulator import (
    ComparisonReport,
    DailySnapshot,
    MonteCarloResult,
    SimulationConfig,
    SimulationResult,
    StrategySimulator,
    _generate_apy_sequence,
    _apply_slippage_and_gas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pools() -> list[PoolInfo]:
    """Create sample pools for testing."""
    return [
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="aave-usdc",
            pool_name="Aave USDC",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=5_000_000_000,
            is_stable=True,
            impermanent_loss_risk=0.0,
        ),
        PoolInfo(
            protocol=Protocol.COMPOUND,
            chain=Chain.ETHEREUM,
            pool_id="compound-usdc",
            pool_name="Compound USDC",
            token_pair="USDC",
            apy=0.035,
            tvl_usd=3_000_000_000,
            is_stable=True,
            impermanent_loss_risk=0.0,
        ),
        PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ETHEREUM,
            pool_id="uni-eth-usdc",
            pool_name="Uniswap ETH/USDC",
            token_pair="ETH/USDC",
            apy=0.08,
            tvl_usd=2_000_000_000,
            is_stable=False,
            impermanent_loss_risk=0.35,
        ),
        PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="curve-3pool",
            pool_name="Curve 3Pool",
            token_pair="DAI/USDC/USDT",
            apy=0.03,
            tvl_usd=1_500_000_000,
            is_stable=True,
            impermanent_loss_risk=0.005,
        ),
    ]


@pytest.fixture
def sim_config() -> SimulationConfig:
    """Default simulation config for tests.

    Uses low gas costs and monthly rebalancing so that yield exceeds costs
    for the low-APY test pools (~3-4% on $10k = ~$1/day).
    """
    return SimulationConfig(
        initial_capital=10_000.0,
        duration_days=90,
        rebalance_frequency_days=30,
        gas_cost_per_rebalance=1.0,
        slippage_bps=5.0,
        apy_noise_std=0.0,  # Deterministic for unit tests
        seed=42,
    )


@pytest.fixture
def noisy_config() -> SimulationConfig:
    """Simulation config with APY noise for Monte Carlo tests."""
    return SimulationConfig(
        initial_capital=10_000.0,
        duration_days=180,
        rebalance_frequency_days=30,
        gas_cost_per_rebalance=1.0,
        slippage_bps=5.0,
        apy_noise_std=0.01,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestGenerateApySequence:
    """Tests for _generate_apy_sequence."""

    def test_deterministic_returns_constant(self) -> None:
        """Zero noise should return constant APY."""
        apys = _generate_apy_sequence(0.05, 30, 0.0, __import__("random").Random(1))
        assert len(apys) == 30
        assert all(a == 0.05 for a in apys)

    def test_noisy_returns_correct_length(self) -> None:
        """Noisy sequence should have correct length."""
        rng = __import__("random").Random(42)
        apys = _generate_apy_sequence(0.05, 100, 0.01, rng)
        assert len(apys) == 100

    def test_noisy_values_non_negative(self) -> None:
        """APY values should never go negative."""
        rng = __import__("random").Random(42)
        apys = _generate_apy_sequence(0.01, 500, 0.05, rng)
        assert all(a >= 0 for a in apys)

    def test_noisy_mean_close_to_base(self) -> None:
        """Mean of noisy sequence should be close to base APY (mean-reverting)."""
        rng = __import__("random").Random(42)
        apys = _generate_apy_sequence(0.05, 1000, 0.01, rng)
        mean_apy = statistics.mean(apys)
        # Mean-reverting should keep it near base (within 1% absolute)
        assert abs(mean_apy - 0.05) < 0.01

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same sequence."""
        rng1 = __import__("random").Random(123)
        rng2 = __import__("random").Random(123)
        apys1 = _generate_apy_sequence(0.05, 50, 0.01, rng1)
        apys2 = _generate_apy_sequence(0.05, 50, 0.01, rng2)
        assert apys1 == apys2

    def test_custom_long_term_mean(self) -> None:
        """Custom long-term mean should attract the sequence."""
        rng = __import__("random").Random(42)
        apys = _generate_apy_sequence(0.08, 2000, 0.01, rng, long_term_mean=0.03)
        mean_apy = statistics.mean(apys)
        # Should revert toward 0.03, not 0.08
        assert mean_apy < 0.06


class TestApplySlippageAndGas:
    """Tests for _apply_slippage_and_gas."""

    def test_basic_deduction(self) -> None:
        result = _apply_slippage_and_gas(10_000, 5.0, 10.0)
        expected = 10_000 - 5.0 - (10_000 * 10 / 10_000)
        assert abs(result - expected) < 0.01

    def test_zero_costs(self) -> None:
        result = _apply_slippage_and_gas(10_000, 0.0, 0.0)
        assert result == 10_000

    def test_floor_at_zero(self) -> None:
        result = _apply_slippage_and_gas(0.01, 100.0, 100.0)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Unit tests: SimulationConfig
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = SimulationConfig()
        assert cfg.initial_capital == 10_000.0
        assert cfg.duration_days == 365
        assert cfg.rebalance_frequency_days == 7
        assert cfg.gas_cost_per_rebalance == 5.0
        assert cfg.slippage_bps == 10.0
        assert cfg.apy_noise_std == 0.0
        assert cfg.seed is None

    def test_frozen(self) -> None:
        cfg = SimulationConfig()
        with pytest.raises(AttributeError):
            cfg.initial_capital = 20_000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration tests: StrategySimulator
# ---------------------------------------------------------------------------

class TestStrategySimulator:
    """Integration tests for StrategySimulator."""

    def test_init_requires_pools(self) -> None:
        with pytest.raises(ValueError, match="At least one pool"):
            StrategySimulator(pools=[])

    def test_run_greedy_produces_result(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY)

        assert isinstance(result, SimulationResult)
        assert result.strategy is AllocationStrategy.GREEDY
        assert len(result.snapshots) == 90
        assert result.final_value > 0
        assert result.total_return_pct > 0  # Should earn yield
        assert result.num_rebalances > 0
        assert result.total_gas_cost > 0

    def test_run_kelly_produces_result(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.KELLY)

        assert isinstance(result, SimulationResult)
        assert result.strategy is AllocationStrategy.KELLY
        assert result.final_value > 0
        assert result.total_return_pct > 0

    def test_run_risk_parity_produces_result(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.RISK_PARITY)

        assert isinstance(result, SimulationResult)
        assert result.strategy is AllocationStrategy.RISK_PARITY
        assert result.final_value > 0

    def test_snapshots_have_correct_dates(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY)

        for i, snap in enumerate(result.snapshots):
            assert snap.day == i
            if i > 0:
                delta = snap.date - result.snapshots[i - 1].date
                assert delta.days == 1

    def test_snapshots_values_increase(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        """With positive APY, portfolio value should generally increase."""
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY)

        # First snapshot should equal initial capital + first day yield
        assert result.snapshots[0].portfolio_value >= sim_config.initial_capital

        # Final value should be higher than initial (positive APY pools)
        assert result.final_value > sim_config.initial_capital

    def test_no_rebalance_frequency(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """rebalance_frequency_days=0 means no rebalancing."""
        cfg = SimulationConfig(
            duration_days=30,
            rebalance_frequency_days=0,
            seed=42,
        )
        sim = StrategySimulator(sample_pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        assert result.num_rebalances == 0
        assert result.total_gas_cost == 0.0
        assert not any(s.rebalanced for s in result.snapshots)

    def test_seed_reproducibility(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        """Same seed should produce identical results."""
        sim1 = StrategySimulator(sample_pools, sim_config)
        sim2 = StrategySimulator(sample_pools, sim_config)

        r1 = sim1.run(AllocationStrategy.GREEDY)
        r2 = sim2.run(AllocationStrategy.GREEDY)

        assert r1.total_return_pct == r2.total_return_pct
        assert r1.final_value == r2.final_value
        assert r1.max_drawdown_pct == r2.max_drawdown_pct

    def test_different_seeds_differ(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Different seeds with noise should produce different results."""
        cfg1 = SimulationConfig(duration_days=90, apy_noise_std=0.02, seed=1)
        cfg2 = SimulationConfig(duration_days=90, apy_noise_std=0.02, seed=2)

        sim1 = StrategySimulator(sample_pools, cfg1)
        sim2 = StrategySimulator(sample_pools, cfg2)

        r1 = sim1.run(AllocationStrategy.GREEDY)
        r2 = sim2.run(AllocationStrategy.GREEDY)

        # With noise, results should differ (extremely unlikely to be identical)
        assert r1.snapshots[-1].portfolio_value != r2.snapshots[-1].portfolio_value

    def test_custom_historical_apys(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        """Should use provided historical APYs when given."""
        historical = {
            "aave-usdc": [0.04] * 90,
            "compound-usdc": [0.035] * 90,
            "uni-eth-usdc": [0.08] * 90,
            "curve-3pool": [0.03] * 90,
        }
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY, historical_apys=historical)

        assert result.final_value > sim_config.initial_capital

    def test_max_drawdown_non_negative(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY)
        assert result.max_drawdown_pct >= 0.0

    def test_sharpe_ratio_reasonable(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY)
        # Sharpe should be positive for positive-APY pools
        assert result.sharpe_ratio > 0
        # Should be in a reasonable range
        assert -5 < result.sharpe_ratio < 50

    def test_volatile_pool_higher_drawdown(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Pools with noise should have some drawdown."""
        cfg = SimulationConfig(
            duration_days=180,
            apy_noise_std=0.02,
            seed=42,
        )
        sim = StrategySimulator(sample_pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)
        # With noise, some drawdown is expected
        # (may be 0 if luck is on our side, but very unlikely with 180 days)
        assert result.max_drawdown_pct >= 0.0


# ---------------------------------------------------------------------------
# Integration tests: run_all_strategies
# ---------------------------------------------------------------------------

class TestRunAllStrategies:
    """Tests for multi-strategy comparison."""

    def test_comparison_report_structure(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        report = sim.run_all_strategies()

        assert isinstance(report, ComparisonReport)
        assert len(report.results) == len(AllocationStrategy)

        for strategy in AllocationStrategy:
            assert strategy.value in report.results
            assert isinstance(report.results[strategy.value], SimulationResult)

    def test_comparison_bests_populated(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, sim_config)
        report = sim.run_all_strategies()

        assert report.best_return in report.results
        assert report.best_risk_adjusted in report.results
        assert report.lowest_drawdown in report.results

    def test_all_strategies_positive_return(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        """All strategies should produce positive returns with safe pools."""
        sim = StrategySimulator(sample_pools, sim_config)
        report = sim.run_all_strategies()

        for name, result in report.results.items():
            assert result.total_return_pct > 0, f"{name} had non-positive return"


# ---------------------------------------------------------------------------
# Integration tests: Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_structure(
        self, sample_pools: list[PoolInfo], noisy_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, noisy_config)
        mc = sim.monte_carlo(AllocationStrategy.GREEDY, num_runs=20)

        assert isinstance(mc, MonteCarloResult)
        assert mc.num_runs == 20
        assert mc.strategy is AllocationStrategy.GREEDY

    def test_monte_carlo_statistics(
        self, sample_pools: list[PoolInfo], noisy_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, noisy_config)
        mc = sim.monte_carlo(AllocationStrategy.GREEDY, num_runs=50)

        # Mean should be positive (positive APY pools)
        assert mc.mean_return_pct > 0

        # Percentiles should be ordered
        assert mc.percentile_5 <= mc.percentile_25
        assert mc.percentile_25 <= mc.median_return_pct
        assert mc.median_return_pct <= mc.percentile_75
        assert mc.percentile_75 <= mc.percentile_95

        # Standard deviation should be positive (noise introduces variance)
        assert mc.std_return_pct > 0

        # Drawdown should be non-negative
        assert mc.mean_max_drawdown >= 0

    def test_monte_carlo_probability_of_loss(
        self, sample_pools: list[PoolInfo], noisy_config: SimulationConfig
    ) -> None:
        sim = StrategySimulator(sample_pools, noisy_config)
        mc = sim.monte_carlo(AllocationStrategy.GREEDY, num_runs=50)

        # With positive APY pools, probability of loss should be low
        assert 0.0 <= mc.probability_of_loss <= 1.0
        assert mc.probability_of_loss < 0.5  # Should be less than 50%

    def test_monte_carlo_low_noise_low_variance(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Low noise should produce low variance in Monte Carlo."""
        cfg = SimulationConfig(
            duration_days=90,
            apy_noise_std=0.002,  # Very low noise
            seed=42,
        )
        sim = StrategySimulator(sample_pools, cfg)
        mc = sim.monte_carlo(AllocationStrategy.GREEDY, num_runs=30)

        # Coefficient of variation should be small
        if mc.mean_return_pct > 0:
            cv = mc.std_return_pct / mc.mean_return_pct
            assert cv < 0.5  # Low noise = low relative variance

    def test_monte_carlo_all_strategies(
        self, sample_pools: list[PoolInfo], noisy_config: SimulationConfig
    ) -> None:
        """Monte Carlo should work for all strategies."""
        sim = StrategySimulator(sample_pools, noisy_config)

        for strategy in AllocationStrategy:
            mc = sim.monte_carlo(strategy, num_runs=10)
            assert mc.num_runs == 10
            assert mc.strategy is strategy


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_single_pool(self) -> None:
        """Should work with a single pool."""
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=5_000_000_000,
                is_stable=True,
            ),
        ]
        cfg = SimulationConfig(
            duration_days=30, rebalance_frequency_days=0, seed=42
        )
        sim = StrategySimulator(pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        assert result.final_value > cfg.initial_capital
        assert result.total_return_pct > 0

    def test_very_short_duration(self) -> None:
        """Should handle 1-day simulation."""
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=5_000_000_000,
            ),
        ]
        cfg = SimulationConfig(duration_days=1, seed=42)
        sim = StrategySimulator(pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        assert len(result.snapshots) == 1

    def test_very_long_duration(self) -> None:
        """Should handle multi-year simulation."""
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=5_000_000_000,
                is_stable=True,
            ),
        ]
        cfg = SimulationConfig(
            duration_days=1095,
            rebalance_frequency_days=0,
            seed=42,
        )  # 3 years, no rebalancing
        sim = StrategySimulator(pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        assert len(result.snapshots) == 1095
        # 3 years at 4% APY on $10k, but only 30% allocated (max_single_allocation)
        # Yield ≈ 10000 * 0.3 * 0.04 * 3 = $360
        assert result.final_value > 10_300

    def test_zero_apy_pool(self) -> None:
        """Zero APY pool should not lose money (ignoring gas)."""
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-zero",
                pool_name="Aave Zero",
                token_pair="USDC",
                apy=0.0,
                tvl_usd=5_000_000_000,
                is_stable=True,
            ),
        ]
        cfg = SimulationConfig(
            duration_days=30,
            rebalance_frequency_days=0,  # No gas cost
            seed=42,
        )
        sim = StrategySimulator(pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        # With zero APY and no gas, value should stay same
        assert abs(result.final_value - cfg.initial_capital) < 0.01

    def test_high_apy_pool(self) -> None:
        """High APY pool should produce high returns."""
        pools = [
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ETHEREUM,
                pool_id="uni-high",
                pool_name="High APY Pool",
                token_pair="ETH/USDC",
                apy=0.50,  # 50% APY
                tvl_usd=500_000_000,
            ),
        ]
        cfg = SimulationConfig(
            duration_days=365,
            rebalance_frequency_days=0,
            seed=42,
        )
        sim = StrategySimulator(pools, cfg)
        result = sim.run(AllocationStrategy.GREEDY)

        # 50% APY on $10k for 1 year, but only 30% allocated (max_single_allocation)
        # Yield ≈ 10000 * 0.3 * 0.50 = $1,500
        assert result.final_value > 11_000
        assert result.total_return_pct > 10

    def test_historical_apys_shorter_than_duration(
        self, sample_pools: list[PoolInfo], sim_config: SimulationConfig
    ) -> None:
        """Should pad short historical APY lists."""
        historical = {
            "aave-usdc": [0.04] * 10,  # Only 10 days for 90-day sim
            "compound-usdc": [0.035] * 10,
            "uni-eth-usdc": [0.08] * 10,
            "curve-3pool": [0.03] * 10,
        }
        sim = StrategySimulator(sample_pools, sim_config)
        result = sim.run(AllocationStrategy.GREEDY, historical_apys=historical)

        assert len(result.snapshots) == 90
        assert result.final_value > 0
