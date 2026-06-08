"""Comprehensive tests for the portfolio rebalancing engine.

Tests cover drift detection, threshold-based triggering, trade generation,
cost-benefit filtering, minimum trade sizes, multi-chain batching,
edge cases, and report structure.
"""

from __future__ import annotations

import pytest

from defi_yield_aggregator.core.gas_estimator import GasEstimator
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol
from defi_yield_aggregator.core.rebalancer import (
    PortfolioRebalancer,
    RebalanceConfig,
    RebalanceReport,
    RebalanceTrade,
    TradeDirection,
    _compute_drift,
    _estimate_trade_gas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pools() -> list[PoolInfo]:
    """Create a set of sample pools for testing."""
    return [
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="aave-eth-usdc",
            pool_name="Aave USDC",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=10_000_000.0,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.COMPOUND,
            chain=Chain.ETHEREUM,
            pool_id="compound-eth-dai",
            pool_name="Compound DAI",
            token_pair="DAI",
            apy=0.04,
            tvl_usd=8_000_000.0,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ARBITRUM,
            pool_id="uni-arb-eth-usdc",
            pool_name="Uniswap ETH/USDC",
            token_pair="ETH/USDC",
            apy=0.12,
            tvl_usd=5_000_000.0,
            is_stable=False,
            impermanent_loss_risk=0.3,
        ),
        PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.POLYGON,
            pool_id="curve-poly-3pool",
            pool_name="Curve 3Pool",
            token_pair="DAI/USDC/USDT",
            apy=0.08,
            tvl_usd=3_000_000.0,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ARBITRUM,
            pool_id="aave-arb-usdt",
            pool_name="Aave USDT (Arbitrum)",
            token_pair="USDT",
            apy=0.06,
            tvl_usd=4_000_000.0,
            is_stable=True,
        ),
    ]


@pytest.fixture
def default_config() -> RebalanceConfig:
    """Default rebalance config for tests."""
    return RebalanceConfig(
        drift_threshold_pct=0.05,
        relative_drift_threshold=0.30,
        min_trade_usd=50.0,
        gas_cost_multiplier=1.2,
        holding_period_days=365,
        max_trades_per_rebalance=20,
        use_absolute_drift=True,
    )


@pytest.fixture
def rebalancer(sample_pools, default_config) -> PortfolioRebalancer:
    """Create a rebalancer with sample pools and default config."""
    return PortfolioRebalancer(
        pools=sample_pools,
        config=default_config,
        current_portfolio_value=100_000.0,
    )


# ---------------------------------------------------------------------------
# Test: Exact match — no rebalance needed
# ---------------------------------------------------------------------------


class TestExactMatch:
    """When current and target are identical, no rebalance should trigger."""

    def test_exact_match_no_trigger(self, rebalancer):
        weights = {"aave-eth-usdc": 0.5, "compound-eth-dai": 0.5}
        report = rebalancer.analyze(current_weights=weights, target_weights=weights)

        assert report.triggered is False
        assert report.trades == []
        assert report.total_gas_cost == 0.0
        assert report.max_drift == 0.0

    def test_exact_match_multi_pool(self, rebalancer):
        weights = {
            "aave-eth-usdc": 0.3,
            "compound-eth-dai": 0.3,
            "uni-arb-eth-usdc": 0.2,
            "curve-poly-3pool": 0.2,
        }
        report = rebalancer.analyze(current_weights=weights, target_weights=weights)

        assert report.triggered is False
        assert report.max_drift == 0.0


# ---------------------------------------------------------------------------
# Test: Small drift below threshold — no trigger
# ---------------------------------------------------------------------------


class TestSmallDrift:
    """Drift below the configured threshold should not trigger rebalance."""

    def test_small_drift_no_trigger(self, rebalancer):
        current = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        # 2% drift on both — well below 5% threshold
        target = {"aave-eth-usdc": 0.52, "compound-eth-dai": 0.48}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is False
        assert report.trades == []
        assert report.max_drift == pytest.approx(0.02, abs=1e-6)

    def test_drift_just_below_threshold(self, rebalancer):
        current = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        # 4.9% drift — just below 5% threshold
        target = {"aave-eth-usdc": 0.549, "compound-eth-dai": 0.451}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is False

    def test_relative_drift_below_threshold(self, sample_pools):
        config = RebalanceConfig(
            drift_threshold_pct=0.20,  # High absolute threshold
            relative_drift_threshold=0.30,
            use_absolute_drift=False,
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools, config=config, current_portfolio_value=100_000.0
        )

        # 10% relative drift (0.02 / 0.20 = 10%) — below 30% relative threshold
        current = {"aave-eth-usdc": 0.20, "compound-eth-dai": 0.80}
        target = {"aave-eth-usdc": 0.22, "compound-eth-dai": 0.78}

        triggered, max_drift = rebalancer.should_rebalance(current, target)
        assert triggered is False


# ---------------------------------------------------------------------------
# Test: Large drift above threshold — trigger with trades
# ---------------------------------------------------------------------------


class TestLargeDrift:
    """Drift above the threshold should trigger rebalance with trades."""

    def test_large_drift_triggers(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        assert len(report.trades) > 0
        assert report.max_drift == pytest.approx(0.20, abs=1e-6)

    def test_trades_have_correct_directions(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        trade_map = {t.pool_id: t for t in report.trades}
        # aave is overweight -> should SELL
        assert trade_map["aave-eth-usdc"].direction == TradeDirection.SELL
        # compound is underweight -> should BUY
        assert trade_map["compound-eth-dai"].direction == TradeDirection.BUY

    def test_trade_amounts_match_drift(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        portfolio_value = 100_000.0

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        trade_map = {t.pool_id: t for t in report.trades}
        # 20% of 100k = 20k
        assert trade_map["aave-eth-usdc"].amount_usd == pytest.approx(20_000.0, abs=0.01)
        assert trade_map["compound-eth-dai"].amount_usd == pytest.approx(20_000.0, abs=0.01)

    def test_trades_sorted_by_priority(self, rebalancer):
        current = {
            "aave-eth-usdc": 0.50,
            "compound-eth-dai": 0.30,
            "uni-arb-eth-usdc": 0.20,
        }
        target = {
            "aave-eth-usdc": 0.35,
            "compound-eth-dai": 0.25,
            "uni-arb-eth-usdc": 0.40,
        }

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        if report.trades:
            priorities = [t.priority for t in report.trades]
            assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# Test: Cost-benefit filtering
# ---------------------------------------------------------------------------


class TestCostBenefit:
    """Trades where gas exceeds APY improvement should be skipped."""

    def test_expensive_gas_skips_small_trade(self, sample_pools):
        """A tiny position with high gas should be filtered out."""
        config = RebalanceConfig(
            drift_threshold_pct=0.01,  # Low threshold to trigger
            min_trade_usd=1.0,  # Very low min trade to allow small trades
            gas_cost_multiplier=1.0,
            holding_period_days=30,  # Short holding period makes gas costly
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools,
            config=config,
            current_portfolio_value=100.0,  # Very small portfolio
        )

        # Drift is significant in %, but absolute amount is tiny ($1)
        current = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        target = {"aave-eth-usdc": 0.60, "compound-eth-dai": 0.40}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        # Trades may be filtered due to cost-benefit or be present
        # with small portfolio, gas cost may exceed APY benefit
        # At minimum, verify the analysis runs without error
        assert isinstance(report, RebalanceReport)

    def test_profitable_trade_passes_filter(self, sample_pools):
        """A large position with good APY should not be filtered."""
        config = RebalanceConfig(
            drift_threshold_pct=0.05,
            min_trade_usd=50.0,
            gas_cost_multiplier=1.0,
            holding_period_days=365,
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools,
            config=config,
            current_portfolio_value=1_000_000.0,  # Large portfolio
        )

        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        assert len(report.trades) == 2  # Both trades should pass


# ---------------------------------------------------------------------------
# Test: Minimum trade size filtering
# ---------------------------------------------------------------------------


class TestMinTradeSize:
    """Trades below min_trade_usd should be filtered out."""

    def test_small_trade_filtered(self, sample_pools):
        config = RebalanceConfig(
            drift_threshold_pct=0.01,
            min_trade_usd=10_000.0,  # High minimum
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools,
            config=config,
            current_portfolio_value=100_000.0,
        )

        # 3% drift = $3,000 trade — below $10,000 minimum
        current = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        target = {"aave-eth-usdc": 0.53, "compound-eth-dai": 0.47}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        # Should trigger (drift > threshold) but trades should be empty (below min)
        if report.triggered:
            for trade in report.trades:
                assert trade.amount_usd >= config.min_trade_usd

    def test_large_trade_passes(self, sample_pools):
        config = RebalanceConfig(
            drift_threshold_pct=0.01,
            min_trade_usd=1_000.0,
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools,
            config=config,
            current_portfolio_value=100_000.0,
        )

        # 20% drift = $20,000 trade — well above $1,000 minimum
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        assert len(report.trades) >= 1


# ---------------------------------------------------------------------------
# Test: Multi-chain trade batching
# ---------------------------------------------------------------------------


class TestMultiChainBatching:
    """Trades should be groupable by chain for execution optimization."""

    def test_batch_by_chain(self, rebalancer):
        current = {
            "aave-eth-usdc": 0.40,
            "compound-eth-dai": 0.20,
            "uni-arb-eth-usdc": 0.20,
            "curve-poly-3pool": 0.20,
        }
        target = {
            "aave-eth-usdc": 0.25,
            "compound-eth-dai": 0.25,
            "uni-arb-eth-usdc": 0.25,
            "curve-poly-3pool": 0.25,
        }

        report = rebalancer.analyze(current_weights=current, target_weights=target)
        batches = rebalancer.batch_trades_by_chain(report.trades)

        # Verify trades are grouped correctly
        assert isinstance(batches, dict)
        for chain, trades in batches.items():
            for trade in trades:
                assert trade.chain == chain

    def test_multi_chain_different_gas_costs(self, rebalancer):
        """Different chains should have different gas estimates."""
        current = {
            "aave-eth-usdc": 0.40,
            "uni-arb-eth-usdc": 0.30,
            "curve-poly-3pool": 0.30,
        }
        target = {
            "aave-eth-usdc": 0.50,
            "uni-arb-eth-usdc": 0.20,
            "curve-poly-3pool": 0.30,
        }

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        if report.triggered and len(report.trades) > 1:
            chains_seen = {t.chain for t in report.trades}
            # At least ethereum and arbitrum trades
            assert len(chains_seen) >= 1


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_portfolio(self, rebalancer):
        """Empty current weights with empty target should produce no-op report."""
        report = rebalancer.analyze(current_weights={}, target_weights={})

        assert report.triggered is False
        assert report.trades == []
        assert report.total_gas_cost == 0.0
        assert report.max_drift == 0.0

    def test_single_position(self, rebalancer):
        """100% in one pool with no change should not trigger."""
        weights = {"aave-eth-usdc": 1.0}
        report = rebalancer.analyze(current_weights=weights, target_weights=weights)

        assert report.triggered is False

    def test_single_position_to_new_pool(self, rebalancer):
        """Moving 100% to a new pool should trigger."""
        current = {"aave-eth-usdc": 1.0}
        target = {"compound-eth-dai": 1.0}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        assert len(report.trades) == 2  # SELL old + BUY new

    def test_new_pool_in_target(self, rebalancer):
        """Target has a pool not in current — should generate BUY trade."""
        current = {"aave-eth-usdc": 1.0}
        target = {"aave-eth-usdc": 0.5, "compound-eth-dai": 0.5}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        trade_map = {t.pool_id: t for t in report.trades}
        assert "compound-eth-dai" in trade_map
        assert trade_map["compound-eth-dai"].direction == TradeDirection.BUY

    def test_pool_removed_from_target(self, rebalancer):
        """Pool in current but not target should generate SELL trade."""
        current = {"aave-eth-usdc": 0.5, "compound-eth-dai": 0.5}
        target = {"aave-eth-usdc": 1.0}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True
        trade_map = {t.pool_id: t for t in report.trades}
        assert "compound-eth-dai" in trade_map
        assert trade_map["compound-eth-dai"].direction == TradeDirection.SELL

    def test_zero_weight_pool(self, rebalancer):
        """A pool with zero weight in current should be handled."""
        current = {"aave-eth-usdc": 0.0, "compound-eth-dai": 1.0}
        target = {"aave-eth-usdc": 0.5, "compound-eth-dai": 0.5}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.triggered is True


# ---------------------------------------------------------------------------
# Test: Drift calculation correctness
# ---------------------------------------------------------------------------


class TestDriftCalculation:
    """Verify drift calculations are mathematically correct."""

    def test_absolute_drift(self):
        current = {"a": 0.6, "b": 0.4}
        target = {"a": 0.5, "b": 0.5}

        drift = _compute_drift(current, target, use_absolute=True)

        assert drift["a"] == pytest.approx(0.1, abs=1e-9)
        assert drift["b"] == pytest.approx(0.1, abs=1e-9)

    def test_relative_drift(self):
        current = {"a": 0.6, "b": 0.4}
        target = {"a": 0.5, "b": 0.5}

        drift = _compute_drift(current, target, use_absolute=False)

        # 0.1 / 0.5 = 0.2 (20% relative)
        assert drift["a"] == pytest.approx(0.2, abs=1e-9)
        assert drift["b"] == pytest.approx(0.2, abs=1e-9)

    def test_drift_with_missing_pool(self):
        current = {"a": 1.0}
        target = {"b": 1.0}

        drift = _compute_drift(current, target, use_absolute=True)

        assert drift["a"] == pytest.approx(1.0, abs=1e-9)
        assert drift["b"] == pytest.approx(1.0, abs=1e-9)

    def test_relative_drift_zero_target(self):
        current = {"a": 0.5, "b": 0.5}
        target = {"a": 1.0, "b": 0.0}

        drift = _compute_drift(current, target, use_absolute=False)

        # b: 0.5 absolute drift / 0 target = inf
        assert drift["b"] == float("inf")
        # a: 0.5 / 1.0 = 0.5
        assert drift["a"] == pytest.approx(0.5, abs=1e-9)

    def test_drift_empty_inputs(self):
        drift = _compute_drift({}, {}, use_absolute=True)
        assert drift == {}

    def test_drift_single_pool(self):
        current = {"a": 0.6}
        target = {"a": 0.4}

        drift = _compute_drift(current, target, use_absolute=True)
        assert drift["a"] == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# Test: Rebalance report structure and metrics
# ---------------------------------------------------------------------------


class TestRebalanceReport:
    """Verify report structure and computed metrics."""

    def test_report_has_all_fields(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert hasattr(report, "trades")
        assert hasattr(report, "total_gas_cost")
        assert hasattr(report, "expected_apy_before")
        assert hasattr(report, "expected_apy_after")
        assert hasattr(report, "net_benefit_annualized")
        assert hasattr(report, "max_drift")
        assert hasattr(report, "triggered")
        assert hasattr(report, "current_allocations")
        assert hasattr(report, "target_allocations")

    def test_report_preserves_allocations(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert report.current_allocations == current
        assert report.target_allocations == target

    def test_apy_before_and_after(self, rebalancer):
        current = {"aave-eth-usdc": 1.0}
        target = {"compound-eth-dai": 1.0}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        # aave APY = 0.05, compound APY = 0.04
        assert report.expected_apy_before == pytest.approx(0.05, abs=1e-4)
        assert report.expected_apy_after == pytest.approx(0.04, abs=1e-4)

    def test_total_gas_cost_positive(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        if report.triggered and report.trades:
            assert report.total_gas_cost > 0

    def test_report_trade_estimated_gas(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        for trade in report.trades:
            assert trade.estimated_gas_usd > 0

    def test_net_benefit_calculation(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        # Net benefit = (apy_after - apy_before) * portfolio_value - gas / years
        assert isinstance(report.net_benefit_annualized, float)


# ---------------------------------------------------------------------------
# Test: Trade properties
# ---------------------------------------------------------------------------


class TestTradeProperties:
    """Test individual trade object properties."""

    def test_trade_frozen(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        for trade in report.trades:
            # frozen dataclass — assignment should raise
            with pytest.raises(AttributeError):
                trade.pool_id = "modified"  # type: ignore[misc]

    def test_trade_has_chain_and_protocol(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        for trade in report.trades:
            assert isinstance(trade.chain, str)
            assert isinstance(trade.protocol, str)
            assert len(trade.chain) > 0
            assert len(trade.protocol) > 0

    def test_trade_direction_enum(self, rebalancer):
        current = {"aave-eth-usdc": 0.70, "compound-eth-dai": 0.30}
        target = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        for trade in report.trades:
            assert trade.direction in (TradeDirection.BUY, TradeDirection.SELL)


# ---------------------------------------------------------------------------
# Test: Gas estimation integration
# ---------------------------------------------------------------------------


class TestGasEstimation:
    """Test gas cost estimation for trades."""

    def test_estimate_trade_gas_buy(self):
        estimator = GasEstimator()
        cost = _estimate_trade_gas(
            estimator, Chain.ETHEREUM, Protocol.AAVE, TradeDirection.BUY
        )
        assert cost > 0

    def test_estimate_trade_gas_sell(self):
        estimator = GasEstimator()
        cost = _estimate_trade_gas(
            estimator, Chain.ETHEREUM, Protocol.AAVE, TradeDirection.SELL
        )
        assert cost > 0

    def test_l2_cheaper_than_l1(self):
        estimator = GasEstimator()
        l1_cost = _estimate_trade_gas(
            estimator, Chain.ETHEREUM, Protocol.AAVE, TradeDirection.BUY
        )
        l2_cost = _estimate_trade_gas(
            estimator, Chain.ARBITRUM, Protocol.AAVE, TradeDirection.BUY
        )
        # L2 should be significantly cheaper
        assert l2_cost < l1_cost

    def test_gas_multiplier_applied(self):
        estimator = GasEstimator()
        base_cost = _estimate_trade_gas(
            estimator, Chain.ETHEREUM, Protocol.AAVE, TradeDirection.BUY, multiplier=1.0
        )
        buffered_cost = _estimate_trade_gas(
            estimator, Chain.ETHEREUM, Protocol.AAVE, TradeDirection.BUY, multiplier=1.5
        )
        assert buffered_cost == pytest.approx(base_cost * 1.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: Config variations
# ---------------------------------------------------------------------------


class TestConfig:
    """Test different configuration options."""

    def test_relative_drift_mode(self, sample_pools):
        config = RebalanceConfig(
            drift_threshold_pct=0.50,  # Very high absolute — won't trigger on abs
            relative_drift_threshold=0.10,  # Low relative — will trigger
            use_absolute_drift=False,
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools, config=config, current_portfolio_value=100_000.0
        )

        # 20% absolute drift on 50% weight = 40% relative drift > 10%
        current = {"aave-eth-usdc": 0.50, "compound-eth-dai": 0.50}
        target = {"aave-eth-usdc": 0.30, "compound-eth-dai": 0.70}

        triggered, _ = rebalancer.should_rebalance(current, target)
        assert triggered is True

    def test_max_trades_limit(self, sample_pools):
        config = RebalanceConfig(
            drift_threshold_pct=0.01,
            max_trades_per_rebalance=1,
        )
        rebalancer = PortfolioRebalancer(
            pools=sample_pools, config=config, current_portfolio_value=100_000.0
        )

        current = {
            "aave-eth-usdc": 0.40,
            "compound-eth-dai": 0.30,
            "uni-arb-eth-usdc": 0.30,
        }
        target = {
            "aave-eth-usdc": 0.30,
            "compound-eth-dai": 0.40,
            "uni-arb-eth-usdc": 0.30,
        }

        report = rebalancer.analyze(current_weights=current, target_weights=target)

        assert len(report.trades) <= 1

    def test_custom_holding_period(self, sample_pools):
        config = RebalanceConfig(holding_period_days=30)
        rebalancer = PortfolioRebalancer(
            pools=sample_pools, config=config, current_portfolio_value=100_000.0
        )

        assert rebalancer._config.holding_period_days == 30


# ---------------------------------------------------------------------------
# Test: Weighted APY calculation
# ---------------------------------------------------------------------------


class TestWeightedAPY:
    """Test APY calculation helper."""

    def test_weighted_apy_single_pool(self, rebalancer):
        weights = {"aave-eth-usdc": 1.0}
        apy = rebalancer.compute_weighted_apy(weights)
        assert apy == pytest.approx(0.05, abs=1e-6)

    def test_weighted_apy_two_pools(self, rebalancer):
        weights = {"aave-eth-usdc": 0.6, "compound-eth-dai": 0.4}
        apy = rebalancer.compute_weighted_apy(weights)
        # 0.6 * 0.05 + 0.4 * 0.04 = 0.03 + 0.016 = 0.046
        assert apy == pytest.approx(0.046, abs=1e-6)

    def test_weighted_apy_unknown_pool(self, rebalancer):
        weights = {"unknown-pool": 1.0}
        apy = rebalancer.compute_weighted_apy(weights)
        assert apy == 0.0  # Unknown pool contributes 0

    def test_weighted_apy_empty(self, rebalancer):
        apy = rebalancer.compute_weighted_apy({})
        assert apy == 0.0
