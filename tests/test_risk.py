"""Tests for the risk engine."""

import pytest

from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol, RiskLevel
from defi_yield_aggregator.core.risk_engine import RiskEngine


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine()


@pytest.fixture
def high_tvl_pool() -> PoolInfo:
    return PoolInfo(
        protocol=Protocol.AAVE,
        chain=Chain.ETHEREUM,
        pool_id="aave-usdc",
        pool_name="Aave USDC",
        token_pair="USDC",
        apy=0.04,
        tvl_usd=5_000_000_000,
        is_stable=True,
    )


@pytest.fixture
def low_tvl_pool() -> PoolInfo:
    return PoolInfo(
        protocol=Protocol.YEARN,
        chain=Chain.ETHEREUM,
        pool_id="yearn-new",
        pool_name="Yearn New Vault",
        token_pair="ETH",
        apy=0.15,
        tvl_usd=500_000,
        is_stable=False,
        impermanent_loss_risk=0.3,
    )


class TestRiskEngine:
    """Tests for the risk scoring engine."""

    def test_high_tvl_low_risk(self, risk_engine: RiskEngine, high_tvl_pool: PoolInfo) -> None:
        score = risk_engine.score_pool(high_tvl_pool)
        assert score.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
        assert score.overall_score < 50

    def test_low_tvl_high_risk(self, risk_engine: RiskEngine, low_tvl_pool: PoolInfo) -> None:
        score = risk_engine.score_pool(low_tvl_pool)
        assert score.overall_score > 40

    def test_stable_bonus(self, risk_engine: RiskEngine) -> None:
        pool_stable = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="stable",
            pool_name="Stable",
            token_pair="USDC",
            apy=0.03,
            tvl_usd=1_000_000_000,
            is_stable=True,
        )
        pool_volatile = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="volatile",
            pool_name="Volatile",
            token_pair="ETH/BTC",
            apy=0.03,
            tvl_usd=1_000_000_000,
            is_stable=False,
        )
        stable_score = risk_engine.score_pool(pool_stable)
        volatile_score = risk_engine.score_pool(pool_volatile)
        assert stable_score.overall_score < volatile_score.overall_score

    def test_custom_weights(self) -> None:
        engine = RiskEngine(tvl_weight=0.5, age_weight=0.2, audit_weight=0.2, chain_weight=0.1)
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="test",
            pool_name="Test",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=500_000_000,
        )
        score = engine.score_pool(pool)
        assert 0 <= score.overall_score <= 100

    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            RiskEngine(tvl_weight=0.5, age_weight=0.5, audit_weight=0.5, chain_weight=0.5)

    def test_filter_by_risk(self, risk_engine: RiskEngine) -> None:
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id=f"pool-{i}",
                pool_name=f"Pool {i}",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=tv,
            )
            for i, tv in enumerate([10_000_000_000, 1_000_000, 500_000])
        ]
        results = risk_engine.filter_by_risk(pools, max_risk=50)
        assert len(results) <= len(pools)
        for _, score in results:
            assert score.overall_score <= 50

    def test_score_all_fields_populated(self, risk_engine: RiskEngine, high_tvl_pool: PoolInfo) -> None:
        score = risk_engine.score_pool(high_tvl_pool)
        assert score.tvl_score > 0
        assert score.age_score > 0
        assert score.audit_score > 0
        assert score.chain_diversity_score > 0

    def test_score_pools_batch(self, risk_engine: RiskEngine) -> None:
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id=f"p{i}",
                pool_name=f"P{i}",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=1_000_000_000,
            )
            for i in range(3)
        ]
        scores = risk_engine.score_pools(pools)
        assert len(scores) == 3
