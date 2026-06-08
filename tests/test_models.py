"""Tests for data models."""

import pytest
from pydantic import ValidationError

from defi_yield_aggregator.core.models import (
    Chain,
    Config,
    OptimizedPortfolio,
    PoolInfo,
    PortfolioAllocation,
    Protocol,
    RiskLevel,
    RiskScore,
)


class TestPoolInfo:
    """Tests for PoolInfo model."""

    def test_create_valid_pool(self) -> None:
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test-pool",
            pool_name="Test Pool",
            token_pair="USDC",
            apy=0.05,
            tvl_usd=1_000_000,
        )
        assert pool.protocol == Protocol.AAVE
        assert pool.apy == 0.05
        assert pool.is_stable is False

    def test_reject_negative_apy(self) -> None:
        with pytest.raises(ValidationError, match="apy"):
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="test",
                pool_name="Test",
                token_pair="USDC",
                apy=-0.01,
                tvl_usd=1_000_000,
            )

    def test_reject_excessive_apy(self) -> None:
        with pytest.raises(ValidationError):
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="test",
                pool_name="Test",
                token_pair="USDC",
                apy=200.0,  # 20000%
                tvl_usd=1_000_000,
            )

    def test_reject_negative_tvl(self) -> None:
        with pytest.raises(ValidationError, match="tvl"):
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="test",
                pool_name="Test",
                token_pair="USDC",
                apy=0.05,
                tvl_usd=-100,
            )

    def test_stable_pool_defaults(self) -> None:
        pool = PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="stable",
            pool_name="Stable",
            token_pair="USDC/USDT",
            apy=0.03,
            tvl_usd=500_000_000,
            is_stable=True,
        )
        assert pool.is_stable is True
        assert pool.impermanent_loss_risk == 0.0
        assert pool.deposit_fee == 0.0


class TestRiskScore:
    """Tests for RiskScore model."""

    def test_create_risk_score(self) -> None:
        score = RiskScore(
            pool_id="test",
            protocol=Protocol.AAVE,
            overall_score=25.0,
            risk_level=RiskLevel.LOW,
        )
        assert score.overall_score == 25.0

    def test_boundary_risk_levels(self) -> None:
        score = RiskScore(
            pool_id="test",
            protocol=Protocol.AAVE,
            overall_score=55.0,
            risk_level=RiskLevel.MEDIUM,
        )
        assert score.risk_level == RiskLevel.MEDIUM


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation model."""

    def test_valid_allocation(self) -> None:
        alloc = PortfolioAllocation(
            pool_id="test",
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            token_pair="USDC",
            allocation_pct=0.25,
            expected_apy=0.05,
            risk_score=20.0,
            amount_usd=2500.0,
        )
        assert alloc.allocation_pct == 0.25

    def test_reject_over_100_pct(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioAllocation(
                pool_id="test",
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                token_pair="USDC",
                allocation_pct=1.5,
                expected_apy=0.05,
                risk_score=20.0,
            )


class TestOptimizedPortfolio:
    """Tests for OptimizedPortfolio model."""

    def test_diversification_score_empty(self) -> None:
        portfolio = OptimizedPortfolio(
            allocations=[],
            total_expected_apy=0.0,
            weighted_risk_score=0.0,
            total_investment_usd=0.0,
        )
        assert portfolio.diversification_score == 0.0
        assert portfolio.num_positions == 0

    def test_diversification_score_single(self) -> None:
        alloc = PortfolioAllocation(
            pool_id="test",
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            token_pair="USDC",
            allocation_pct=1.0,
            expected_apy=0.05,
            risk_score=20.0,
        )
        portfolio = OptimizedPortfolio(
            allocations=[alloc],
            total_expected_apy=0.05,
            weighted_risk_score=20.0,
            total_investment_usd=10000.0,
        )
        # Single position = 0 diversification
        assert portfolio.diversification_score == 0.0
        assert portfolio.num_positions == 1


class TestConfig:
    """Tests for Config model."""

    def test_default_config(self) -> None:
        config = Config()
        assert config.max_risk_score == 70.0
        assert config.min_positions == 2
        assert config.stablecoins_only is False

    def test_custom_config(self) -> None:
        config = Config(max_risk_score=50.0, stablecoins_only=True)
        assert config.max_risk_score == 50.0
        assert config.stablecoins_only is True
