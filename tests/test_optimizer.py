"""Tests for the portfolio optimizer."""

import pytest

from defi_yield_aggregator.core.models import Chain, Config, PoolInfo, Protocol
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer


@pytest.fixture
def sample_pools() -> list[PoolInfo]:
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
        ),
        PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="curve-3pool",
            pool_name="Curve 3Pool",
            token_pair="DAI/USDC/USDT",
            apy=0.032,
            tvl_usd=1_500_000_000,
            is_stable=True,
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
            impermanent_loss_risk=0.3,
        ),
        PoolInfo(
            protocol=Protocol.YEARN,
            chain=Chain.ETHEREUM,
            pool_id="yearn-usdc",
            pool_name="Yearn USDC",
            token_pair="USDC",
            apy=0.055,
            tvl_usd=500_000_000,
            is_stable=True,
        ),
    ]


class TestPortfolioOptimizer:
    """Tests for portfolio optimization."""

    def test_basic_optimization(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions > 0
        assert portfolio.total_expected_apy > 0
        assert portfolio.total_investment_usd == 100_000

    def test_allocation_sum_le_one(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(sample_pools, 50_000)
        total_pct = sum(a.allocation_pct for a in portfolio.allocations)
        assert total_pct <= 1.001  # floating point tolerance

    def test_max_positions_respected(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_positions=2)
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions <= 2

    def test_max_allocation_respected(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_single_allocation_pct=0.20)
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            assert alloc.allocation_pct <= 0.201

    def test_stablecoins_only(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(stablecoins_only=True)
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(sample_pools, 50_000)
        for alloc in portfolio.allocations:
            assert alloc.token_pair in ("USDC", "USDT", "DAI/USDC/USDT")

    def test_zero_investment_raises(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer()
        with pytest.raises(ValueError, match="positive"):
            optimizer.optimize(sample_pools, 0)

    def test_negative_investment_raises(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer()
        with pytest.raises(ValueError, match="positive"):
            optimizer.optimize(sample_pools, -1000)

    def test_no_eligible_pools(self) -> None:
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="tiny",
                pool_name="Tiny",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=100,  # Below min_tvl
            )
        ]
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(pools, 10_000)
        assert portfolio.num_positions == 0
        assert portfolio.total_expected_apy == 0.0

    def test_amount_usd_consistent(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            expected = round(100_000 * alloc.allocation_pct, 2)
            assert abs(alloc.amount_usd - expected) < 0.1

    def test_risk_constraint(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_risk_score=25.0)  # Very strict
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            assert alloc.risk_score <= 25.0
