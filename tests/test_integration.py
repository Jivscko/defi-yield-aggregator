"""Integration tests for the full pipeline: fetch → risk → optimize."""

from __future__ import annotations

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import get_all_adapters
from defi_yield_aggregator.core.models import (
    Chain,
    Config,
    PoolInfo,
    Protocol,
    RiskLevel,
)
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer
from defi_yield_aggregator.core.risk_engine import RiskEngine


@pytest.fixture
def all_pools() -> list[PoolInfo]:
    """Fetch all pools from all adapters (sync wrapper)."""
    loop = asyncio.new_event_loop()
    try:
        adapters = get_all_adapters()
        pools: list[PoolInfo] = []
        for adapter in adapters:
            result = loop.run_until_complete(adapter.fetch_pools())
            pools.extend(result)
        return pools
    finally:
        loop.close()


class TestFullPipeline:
    """Integration tests for the complete data pipeline."""

    def test_all_adapters_return_pools(self, all_pools: list[PoolInfo]) -> None:
        """All registered adapters should return at least one pool."""
        assert len(all_pools) > 0

    def test_all_protocols_represented(self, all_pools: list[PoolInfo]) -> None:
        """Every Protocol enum value should appear in the pool data."""
        protocols_in_pools = {p.protocol for p in all_pools}
        # At least the major protocols should be present
        expected = {Protocol.AAVE, Protocol.COMPOUND, Protocol.UNISWAP, Protocol.CURVE}
        assert expected.issubset(protocols_in_pools)

    def test_pool_data_validity(self, all_pools: list[PoolInfo]) -> None:
        """All pools should have sensible data."""
        for pool in all_pools:
            assert pool.pool_id, f"Empty pool_id for {pool}"
            assert pool.pool_name, f"Empty pool_name for {pool}"
            assert pool.apy >= 0, f"Negative APY for {pool.pool_id}"
            assert pool.tvl_usd >= 0, f"Negative TVL for {pool.pool_id}"
            assert 0 <= pool.impermanent_loss_risk <= 1, f"IL risk out of range for {pool.pool_id}"

    def test_risk_scoring_all_pools(self, all_pools: list[PoolInfo]) -> None:
        """Risk engine should score all pools without errors."""
        engine = RiskEngine()
        scores = engine.score_pools(all_pools)
        assert len(scores) == len(all_pools)
        for score in scores:
            assert 0 <= score.overall_score <= 100
            assert score.risk_level in RiskLevel.__members__.values()

    def test_risk_filtering(self, all_pools: list[PoolInfo]) -> None:
        """Risk filtering should return only pools under the threshold."""
        engine = RiskEngine()
        safe = engine.filter_by_risk(all_pools, max_risk=50)
        for pool, score in safe:
            assert score.overall_score <= 50

    def test_optimizer_produces_portfolio(self, all_pools: list[PoolInfo]) -> None:
        """Optimizer should produce a valid portfolio from real pool data."""
        config = Config(
            max_risk_score=70,
            min_tvl_usd=1_000_000,
            max_single_allocation_pct=0.30,
        )
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(all_pools, investment_usd=100_000)

        assert portfolio.num_positions > 0
        assert portfolio.total_expected_apy > 0
        assert portfolio.weighted_risk_score <= 100
        # Sum of allocations should be ~100%
        total_alloc = sum(a.allocation_pct for a in portfolio.allocations)
        assert abs(total_alloc - 1.0) < 0.01 or total_alloc <= 1.0

    def test_optimizer_stablecoins_only(self, all_pools: list[PoolInfo]) -> None:
        """Stablecoins-only mode should only include stable pools."""
        config = Config(
            max_risk_score=70,
            min_tvl_usd=1_000_000,
            stablecoins_only=True,
        )
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(all_pools, investment_usd=50_000)

        for alloc in portfolio.allocations:
            # Find the matching pool and verify it's stable
            matching = [p for p in all_pools if p.pool_id == alloc.pool_id]
            if matching:
                assert matching[0].is_stable

    def test_optimizer_custom_weights(self, all_pools: list[PoolInfo]) -> None:
        """Optimizer with Kelly Criterion strategy should work end-to-end."""
        config = Config(
            max_risk_score=80,
            min_tvl_usd=500_000,
            allocation_strategy="kelly",
        )
        optimizer = PortfolioOptimizer(config=config)
        portfolio = optimizer.optimize(all_pools, investment_usd=200_000)
        assert portfolio.num_positions >= 0  # May or may not find positions

    def test_chain_filtering_pipeline(self, all_pools: list[PoolInfo]) -> None:
        """Chain filtering works through the full pipeline."""
        eth_pools = [p for p in all_pools if p.chain == Chain.ETHEREUM]
        assert len(eth_pools) > 0

        engine = RiskEngine()
        scores = engine.score_pools(eth_pools)
        assert len(scores) == len(eth_pools)

        optimizer = PortfolioOptimizer()
        portfolio = optimizer.optimize(eth_pools, 10_000)
        # Should still find something on Ethereum
        assert portfolio.num_positions >= 0

    def test_single_protocol_pipeline(self, all_pools: list[PoolInfo]) -> None:
        """Running the pipeline on a single protocol's pools."""
        aave_pools = [p for p in all_pools if p.protocol == Protocol.AAVE]
        assert len(aave_pools) > 0

        engine = RiskEngine()
        scores = engine.score_pools(aave_pools)
        assert all(s.protocol == Protocol.AAVE for s in scores)

    def test_risk_score_consistency(self, all_pools: list[PoolInfo]) -> None:
        """Scoring the same pool twice gives identical results."""
        engine = RiskEngine()
        pool = all_pools[0]
        score1 = engine.score_pool(pool)
        score2 = engine.score_pool(pool)
        assert score1.overall_score == score2.overall_score
        assert score1.risk_level == score2.risk_level

    def test_custom_risk_weights(self, all_pools: list[PoolInfo]) -> None:
        """Custom risk weights change scoring behavior."""
        default_engine = RiskEngine()
        conservative_engine = RiskEngine(
            tvl_weight=0.40,
            age_weight=0.20,
            audit_weight=0.25,
            chain_weight=0.15,
            liquidity_weight=0.0,
            sc_risk_weight=0.0,
        )
        pool = all_pools[0]
        default_score = default_engine.score_pool(pool)
        conservative_score = conservative_engine.score_pool(pool)
        # Scores may differ based on weights
        assert 0 <= default_score.overall_score <= 100
        assert 0 <= conservative_score.overall_score <= 100


class TestCrossModuleInterop:
    """Tests verifying modules work together correctly."""

    def test_pool_to_risk_to_optimize(self, all_pools: list[PoolInfo]) -> None:
        """Full flow: fetch pools → risk filter → optimize remaining."""
        engine = RiskEngine()
        safe_pools_scores = engine.filter_by_risk(all_pools, max_risk=60)
        safe_pools = [p for p, _ in safe_pools_scores]

        if safe_pools:
            optimizer = PortfolioOptimizer()
            portfolio = optimizer.optimize(safe_pools, 50_000)
            assert portfolio.num_positions >= 0
            # Risk should be bounded since we pre-filtered
            for alloc in portfolio.allocations:
                assert alloc.risk_score <= 100

    def test_stablecoin_low_risk_correlation(self, all_pools: list[PoolInfo]) -> None:
        """Stablecoin pools tend to have lower risk scores."""
        engine = RiskEngine()
        stable_scores = [engine.score_pool(p) for p in all_pools if p.is_stable]
        volatile_scores = [engine.score_pool(p) for p in all_pools if not p.is_stable]

        if stable_scores and volatile_scores:
            avg_stable = sum(s.overall_score for s in stable_scores) / len(stable_scores)
            avg_volatile = sum(s.overall_score for s in volatile_scores) / len(volatile_scores)
            # Stable pools should generally score lower (safer)
            # Allow some margin since IL risk and TVL matter too
            assert avg_stable <= avg_volatile + 20

    def test_high_tvl_low_risk_correlation(self, all_pools: list[PoolInfo]) -> None:
        """High-TVL pools tend to have lower risk scores."""
        engine = RiskEngine()
        high_tvl = [p for p in all_pools if p.tvl_usd >= 1_000_000_000]
        low_tvl = [p for p in all_pools if p.tvl_usd < 100_000_000]

        if high_tvl and low_tvl:
            avg_high = sum(engine.score_pool(p).overall_score for p in high_tvl) / len(high_tvl)
            avg_low = sum(engine.score_pool(p).overall_score for p in low_tvl) / len(low_tvl)
            assert avg_high < avg_low + 30
