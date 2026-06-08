"""Tests for the gas cost estimator module."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from defi_yield_aggregator.core.gas_estimator import (
    GasEstimate,
    GasEstimator,
    Operation,
    _BASE_GAS_UNITS,
    _MOCK_GAS_PRICES_GWEI,
    _MOCK_NATIVE_PRICES_USD,
)
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def estimator() -> GasEstimator:
    """Default gas estimator with mock prices."""
    return GasEstimator()


@pytest.fixture
def sample_pools() -> list[PoolInfo]:
    """Sample pools across chains and protocols."""
    return [
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="aave-eth-usdc",
            pool_name="Aave V3 USDC",
            token_pair="USDC",
            apy=0.045,
            tvl_usd=500_000_000,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ETHEREUM,
            pool_id="uni-eth-usdt",
            pool_name="Uniswap ETH/USDT",
            token_pair="ETH/USDT",
            apy=0.12,
            tvl_usd=100_000_000,
            is_stable=False,
        ),
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ARBITRUM,
            pool_id="aave-arb-usdc",
            pool_name="Aave V3 USDC (Arbitrum)",
            token_pair="USDC",
            apy=0.038,
            tvl_usd=200_000_000,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="curve-3pool",
            pool_name="Curve 3Pool",
            token_pair="DAI/USDC/USDT",
            apy=0.025,
            tvl_usd=800_000_000,
            is_stable=True,
        ),
        PoolInfo(
            protocol=Protocol.LIDO,
            chain=Chain.ETHEREUM,
            pool_id="lido-steth",
            pool_name="Lido stETH",
            token_pair="stETH",
            apy=0.035,
            tvl_usd=15_000_000_000,
            is_stable=False,
        ),
    ]


# ---------------------------------------------------------------------------
# GasEstimate model tests
# ---------------------------------------------------------------------------

class TestGasEstimate:
    """Tests for the GasEstimate dataclass."""

    def test_creation(self) -> None:
        est = GasEstimate(
            operation="deposit",
            chain=Chain.ETHEREUM,
            protocol=Protocol.AAVE,
            gas_units=150_000,
            gas_price_gwei=30.0,
            native_token_price_usd=3500.0,
            total_cost_usd=15.75,
        )
        assert est.operation == "deposit"
        assert est.chain == Chain.ETHEREUM
        assert est.protocol == Protocol.AAVE
        assert est.gas_units == 150_000
        assert isinstance(est.timestamp, datetime)

    def test_gas_cost_eth_property(self) -> None:
        est = GasEstimate(
            operation="deposit",
            chain=Chain.ETHEREUM,
            protocol=Protocol.AAVE,
            gas_units=150_000,
            gas_price_gwei=30.0,
            native_token_price_usd=3500.0,
            total_cost_usd=15.75,
        )
        # 150000 * 30 / 1e9 = 0.0045 ETH
        assert abs(est.gas_cost_eth - 0.0045) < 1e-10


# ---------------------------------------------------------------------------
# GasEstimator.estimate_gas tests
# ---------------------------------------------------------------------------

class TestEstimateGas:
    """Tests for single operation gas estimation."""

    def test_ethereum_deposit(self, estimator: GasEstimator) -> None:
        est = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        assert est.chain == Chain.ETHEREUM
        assert est.protocol == Protocol.AAVE
        assert est.gas_units == 150_000  # AAVE has 1.0 multiplier
        assert est.gas_price_gwei == 30.0
        assert est.total_cost_usd > 0

    def test_ethereum_deposit_cost_calculation(self, estimator: GasEstimator) -> None:
        est = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        # 150000 * 30e-9 * 3500 = 15.75
        assert abs(est.total_cost_usd - 15.75) < 0.01

    def test_l2_much_cheaper_than_l1(self, estimator: GasEstimator) -> None:
        eth_cost = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        arb_cost = estimator.estimate_gas(Operation.DEPOSIT, Chain.ARBITRUM, Protocol.AAVE)
        opt_cost = estimator.estimate_gas(Operation.DEPOSIT, Chain.OPTIMISM, Protocol.AAVE)
        base_cost = estimator.estimate_gas(Operation.DEPOSIT, Chain.BASE, Protocol.AAVE)

        # L2s should be dramatically cheaper despite higher gas units
        assert arb_cost.total_cost_usd < eth_cost.total_cost_usd * 0.05
        assert opt_cost.total_cost_usd < eth_cost.total_cost_usd * 0.05
        assert base_cost.total_cost_usd < eth_cost.total_cost_usd * 0.05

    def test_protocol_multipliers(self, estimator: GasEstimator) -> None:
        aave_est = estimator.estimate_gas(Operation.SWAP, Chain.ETHEREUM, Protocol.AAVE)
        curve_est = estimator.estimate_gas(Operation.SWAP, Chain.ETHEREUM, Protocol.CURVE)
        lido_est = estimator.estimate_gas(Operation.SWAP, Chain.ETHEREUM, Protocol.LIDO)

        # Curve (1.3x) > Aave (1.0x) > Lido (0.85x)
        assert curve_est.gas_units > aave_est.gas_units > lido_est.gas_units
        assert curve_est.total_cost_usd > aave_est.total_cost_usd > lido_est.total_cost_usd

    def test_all_operations_supported(self, estimator: GasEstimator) -> None:
        for op in Operation:
            for chain in [Chain.ETHEREUM, Chain.ARBITRUM, Chain.POLYGON]:
                est = estimator.estimate_gas(op, chain, Protocol.AAVE)
                assert est.gas_units > 0
                assert est.total_cost_usd > 0

    def test_string_operation_input(self, estimator: GasEstimator) -> None:
        est = estimator.estimate_gas("deposit", Chain.ETHEREUM, Protocol.AAVE)
        assert est.operation == "deposit"

    def test_custom_gas_prices(self) -> None:
        custom_prices = {Chain.ETHEREUM: 100.0}  # 100 gwei
        custom = GasEstimator(gas_prices_gwei=custom_prices)
        default = GasEstimator()

        custom_est = custom.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        default_est = default.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)

        assert custom_est.total_cost_usd > default_est.total_cost_usd
        assert abs(custom_est.total_cost_usd / default_est.total_cost_usd - 100.0 / 30.0) < 0.01

    def test_custom_native_prices(self) -> None:
        custom_prices = {Chain.ETHEREUM: 7000.0}  # ETH at 7k
        custom = GasEstimator(native_prices_usd=custom_prices)
        default = GasEstimator()

        custom_est = custom.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        default_est = default.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)

        assert abs(custom_est.total_cost_usd / default_est.total_cost_usd - 2.0) < 0.01


# ---------------------------------------------------------------------------
# estimate_net_apy tests
# ---------------------------------------------------------------------------

class TestEstimateNetApy:
    """Tests for net APY calculation after gas costs."""

    def test_large_investment_high_apy(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.10,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=1_000_000, holding_period_days=365)

        assert result["profitable"] is True
        assert result["gross_apy"] == 0.10
        assert result["net_apy"] > 0
        assert result["gas_cost_pct"] < 1.0  # Gas should be tiny fraction of $1M

    def test_small_investment_gas_impact(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=100, holding_period_days=30)

        # $100 on Ethereum for 30 days at 5% APY: earnings ~$0.41, gas ~$31.5
        assert result["profitable"] is False
        assert result["breakeven_days"] > 365

    def test_l2_small_investment_profitable(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ARBITRUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=100, holding_period_days=365)

        # Arbitrum gas is very cheap — even $100 should be profitable over a year
        assert result["profitable"] is True
        assert result["gas_cost_pct"] < 1.0

    def test_breakeven_days_calculation(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=100_000, holding_period_days=365)

        # 5% of $100k = $5k/year, gas ~$31.5 -> breakeven ~2.3 days
        assert 0 < result["breakeven_days"] < 10

    def test_zero_investment(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=0, holding_period_days=365)

        assert result["profitable"] is False
        assert result["breakeven_days"] == float("inf")
        assert result["net_apy"] == 0.0

    def test_short_holding_period(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=10_000, holding_period_days=1)

        # 1 day at 5% on $10k: earnings ~$1.37, gas ~$31.5 -> not profitable
        assert result["profitable"] is False

    def test_result_has_all_keys(self, estimator: GasEstimator) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=1000)

        expected_keys = {"pool_id", "gross_apy", "gas_cost_usd", "net_apy", "breakeven_days", "profitable", "gas_cost_pct"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# batch_estimate tests
# ---------------------------------------------------------------------------

class TestBatchEstimate:
    """Tests for batch pool estimation."""

    def test_sorted_by_net_apy(self, estimator: GasEstimator, sample_pools: list[PoolInfo]) -> None:
        results = estimator.batch_estimate(sample_pools, investment_usd=10_000, holding_period_days=365)

        net_apys = [r["net_apy"] for r in results]
        assert net_apys == sorted(net_apys, reverse=True)

    def test_l2_pools_rank_higher_for_small_investment(
        self, estimator: GasEstimator, sample_pools: list[PoolInfo]
    ) -> None:
        """With small investment, L2 pools should rank higher due to lower gas."""
        results = estimator.batch_estimate(sample_pools, investment_usd=100, holding_period_days=365)

        # Arbitrum Aave should rank above Ethereum Aave for small investment
        arb_result = next(r for r in results if r["pool_id"] == "aave-arb-usdc")
        eth_result = next(r for r in results if r["pool_id"] == "aave-eth-usdc")
        assert arb_result["net_apy"] > eth_result["net_apy"]

    def test_empty_pools(self, estimator: GasEstimator) -> None:
        results = estimator.batch_estimate([], investment_usd=1000)
        assert results == []

    def test_single_pool(self, estimator: GasEstimator, sample_pools: list[PoolInfo]) -> None:
        results = estimator.batch_estimate(sample_pools[:1], investment_usd=10_000)
        assert len(results) == 1
        assert results[0]["pool_id"] == sample_pools[0].pool_id

    def test_all_pools_have_required_keys(
        self, estimator: GasEstimator, sample_pools: list[PoolInfo]
    ) -> None:
        results = estimator.batch_estimate(sample_pools, investment_usd=5_000)
        for r in results:
            assert "pool_id" in r
            assert "net_apy" in r
            assert "profitable" in r
            assert "gas_cost_usd" in r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_polygon_gas_cheap(self, estimator: GasEstimator) -> None:
        eth = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        poly = estimator.estimate_gas(Operation.DEPOSIT, Chain.POLYGON, Protocol.AAVE)

        # Polygon should be much cheaper than Ethereum
        assert poly.total_cost_usd < eth.total_cost_usd

    def test_withdraw_cheaper_than_deposit(self, estimator: GasEstimator) -> None:
        deposit = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        withdraw = estimator.estimate_gas(Operation.WITHDRAW, Chain.ETHEREUM, Protocol.AAVE)

        assert withdraw.gas_units < deposit.gas_units

    def test_swap_most_expensive(self, estimator: GasEstimator) -> None:
        deposit = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        withdraw = estimator.estimate_gas(Operation.WITHDRAW, Chain.ETHEREUM, Protocol.AAVE)
        swap = estimator.estimate_gas(Operation.SWAP, Chain.ETHEREUM, Protocol.AAVE)

        assert swap.gas_units > deposit.gas_units
        assert swap.gas_units > withdraw.gas_units

    def test_net_apy_never_negative(self, estimator: GasEstimator) -> None:
        """Even with tiny investment on expensive chain, net_apy floors at 0."""
        pool = PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.001,  # Very low APY
            tvl_usd=100_000_000,
        )
        result = estimator.estimate_net_apy(pool, investment_usd=10, holding_period_days=7)

        assert result["net_apy"] >= 0.0
