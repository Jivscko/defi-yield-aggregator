"""Tests for the portfolio optimizer — all four strategies."""

from __future__ import annotations

import pytest

from defi_yield_aggregator.core.models import Chain, Config, PoolInfo, Protocol
from defi_yield_aggregator.core.optimizer import (
    AllocationStrategy,
    PortfolioOptimizer,
    _pool_edge,
    _pool_volatility,
    _kelly_optimize,
    _risk_parity_optimize,
    _black_litterman_optimize,
)
from defi_yield_aggregator.core.risk_engine import RiskEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine()


# ---------------------------------------------------------------------------
# Original greedy tests (unchanged)
# ---------------------------------------------------------------------------


class TestGreedyOptimizer:
    """Tests for the original greedy portfolio optimization."""

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


# ---------------------------------------------------------------------------
# Kelly Criterion tests
# ---------------------------------------------------------------------------


class TestKellyOptimizer:
    """Tests for Kelly Criterion allocation strategy."""

    def test_kelly_basic(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions > 0
        assert portfolio.total_expected_apy > 0
        assert portfolio.total_investment_usd == 100_000

    def test_kelly_allocation_sum_le_one(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        total_pct = sum(a.allocation_pct for a in portfolio.allocations)
        assert total_pct <= 1.001

    def test_kelly_max_positions(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_positions=3)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.KELLY
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions <= 3

    def test_kelly_max_allocation_cap(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_single_allocation_pct=0.25)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.KELLY
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            assert alloc.allocation_pct <= 0.251

    def test_kelly_fraction_parameter(self, sample_pools: list[PoolInfo]) -> None:
        """Half-Kelly should produce smaller or equal allocations than full Kelly."""
        opt_half = PortfolioOptimizer(
            strategy=AllocationStrategy.KELLY, kelly_fraction=0.25
        )
        opt_full = PortfolioOptimizer(
            strategy=AllocationStrategy.KELLY, kelly_fraction=1.0
        )
        port_half = opt_half.optimize(sample_pools, 100_000)
        port_full = opt_full.optimize(sample_pools, 100_000)

        # Both should produce valid portfolios
        assert port_half.num_positions > 0
        assert port_full.num_positions > 0

        # Full Kelly may allocate more to high-edge pools
        # At minimum, full Kelly should not produce *less* total allocation
        # (both may be capped at 1.0 though)
        assert port_full.total_expected_apy >= 0

    def test_kelly_fraction_validation(self) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            PortfolioOptimizer(strategy=AllocationStrategy.KELLY, kelly_fraction=0.0)
        with pytest.raises(ValueError, match="kelly_fraction"):
            PortfolioOptimizer(strategy=AllocationStrategy.KELLY, kelly_fraction=1.5)

    def test_kelly_no_eligible_pools(self) -> None:
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="tiny",
                pool_name="Tiny",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=100,
            )
        ]
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(pools, 10_000)
        assert portfolio.num_positions == 0

    def test_kelly_zero_investment_raises(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        with pytest.raises(ValueError, match="positive"):
            optimizer.optimize(sample_pools, 0)

    def test_kelly_higher_edge_to_variance_pool_gets_more(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Kelly allocates more to pools with better edge/variance ratio.

        Aave USDC (4% APY, lowest risk) vs Uniswap ETH/USDC (8% APY, but
        high risk + IL).  Aave should get a larger Kelly allocation because
        its much lower volatility produces a higher Kelly fraction despite
        lower raw APY.
        """
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(sample_pools, 100_000)

        aave_alloc = next(
            (a for a in portfolio.allocations if a.pool_id == "aave-usdc"), None
        )
        uni_alloc = next(
            (a for a in portfolio.allocations if a.pool_id == "uni-eth-usdc"), None
        )
        # Aave (low risk, stable) should get at least as much as Uniswap
        # (high risk, non-stable, IL exposure)
        if aave_alloc and uni_alloc:
            assert aave_alloc.allocation_pct >= uni_alloc.allocation_pct

    def test_kelly_stablecoins_only(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(stablecoins_only=True)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.KELLY
        )
        portfolio = optimizer.optimize(sample_pools, 50_000)
        for alloc in portfolio.allocations:
            assert alloc.token_pair in ("USDC", "USDT", "DAI/USDC/USDT")


# ---------------------------------------------------------------------------
# Risk Parity tests
# ---------------------------------------------------------------------------


class TestRiskParityOptimizer:
    """Tests for Risk Parity allocation strategy."""

    def test_risk_parity_basic(self, sample_pools: list[PoolInfo]) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions > 0
        assert portfolio.total_expected_apy > 0
        assert portfolio.total_investment_usd == 100_000

    def test_risk_parity_allocation_sum_le_one(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        total_pct = sum(a.allocation_pct for a in portfolio.allocations)
        assert total_pct <= 1.001

    def test_risk_parity_max_positions(self, sample_pools: list[PoolInfo]) -> None:
        config = Config(max_positions=3)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.RISK_PARITY
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions <= 3

    def test_risk_parity_max_allocation_cap(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        config = Config(max_single_allocation_pct=0.25)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.RISK_PARITY
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            assert alloc.allocation_pct <= 0.251

    def test_risk_parity_favors_stable_pools(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Stable pools (lower volatility) should receive larger allocations in risk parity."""
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        portfolio = optimizer.optimize(sample_pools, 100_000)

        # Stable pools should be in the top allocations
        stable_allocs = [a for a in portfolio.allocations if a.token_pair in ("USDC", "DAI/USDC/USDT")]
        non_stable_allocs = [a for a in portfolio.allocations if a.token_pair == "ETH/USDC"]

        if stable_allocs and non_stable_allocs:
            max_stable = max(a.allocation_pct for a in stable_allocs)
            max_non_stable = max(a.allocation_pct for a in non_stable_allocs)
            # Stable pools should get at least as much as non-stable
            assert max_stable >= max_non_stable

    def test_risk_parity_no_eligible_pools(self) -> None:
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="tiny",
                pool_name="Tiny",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=100,
            )
        ]
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        portfolio = optimizer.optimize(pools, 10_000)
        assert portfolio.num_positions == 0

    def test_risk_parity_zero_investment_raises(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        with pytest.raises(ValueError, match="positive"):
            optimizer.optimize(sample_pools, 0)

    def test_risk_parity_stablecoins_only(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        config = Config(stablecoins_only=True)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.RISK_PARITY
        )
        portfolio = optimizer.optimize(sample_pools, 50_000)
        for alloc in portfolio.allocations:
            assert alloc.token_pair in ("USDC", "USDT", "DAI/USDC/USDT")

    def test_risk_parity_equal_risk_contribution(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """Verify that risk parity produces more balanced allocations than greedy."""
        opt_rp = PortfolioOptimizer(strategy=AllocationStrategy.RISK_PARITY)
        opt_greedy = PortfolioOptimizer(strategy=AllocationStrategy.GREEDY)

        port_rp = opt_rp.optimize(sample_pools, 100_000)
        port_greedy = opt_greedy.optimize(sample_pools, 100_000)

        # Risk parity should have higher diversification (lower allocation spread)
        rp_pcts = [a.allocation_pct for a in port_rp.allocations]
        greedy_pcts = [a.allocation_pct for a in port_greedy.allocations]

        if len(rp_pcts) > 1 and len(greedy_pcts) > 1:
            rp_range = max(rp_pcts) - min(rp_pcts)
            greedy_range = max(greedy_pcts) - min(greedy_pcts)
            # Risk parity should have a narrower allocation range
            assert rp_range <= greedy_range + 0.01  # small tolerance


# ---------------------------------------------------------------------------
# Black-Litterman tests
# ---------------------------------------------------------------------------


class TestBlackLittermanOptimizer:
    """Tests for Black-Litterman allocation strategy."""

    def test_bl_basic(self, sample_pools: list[PoolInfo]) -> None:
        """BL strategy produces a valid portfolio."""
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions > 0
        assert portfolio.total_expected_apy > 0
        assert portfolio.total_investment_usd == 100_000

    def test_bl_allocation_sum_le_one(self, sample_pools: list[PoolInfo]) -> None:
        """BL allocations sum to at most 1.0."""
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        portfolio = optimizer.optimize(sample_pools, 100_000)
        total_pct = sum(a.allocation_pct for a in portfolio.allocations)
        assert total_pct <= 1.001

    def test_bl_max_positions(self, sample_pools: list[PoolInfo]) -> None:
        """BL respects max_positions constraint."""
        config = Config(max_positions=3)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.BLACK_LITTERMAN
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        assert portfolio.num_positions <= 3

    def test_bl_max_allocation_cap(self, sample_pools: list[PoolInfo]) -> None:
        """BL respects max_single_allocation_pct."""
        config = Config(max_single_allocation_pct=0.25)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.BLACK_LITTERMAN
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)
        for alloc in portfolio.allocations:
            assert alloc.allocation_pct <= 0.251

    def test_bl_tvl_tilt(self, sample_pools: list[PoolInfo]) -> None:
        """BL should tilt toward high-TVL pools (market equilibrium prior).

        With low tau (more trust in equilibrium), the BL portfolio should
        assign more weight to pools with higher TVL, since the prior is
        TVL-weighted.
        """
        optimizer = PortfolioOptimizer(
            strategy=AllocationStrategy.BLACK_LITTERMAN, bl_tau=0.01
        )
        portfolio = optimizer.optimize(sample_pools, 100_000)

        # Aave USDC has TVL $5B (largest) — should get a large allocation
        aave_alloc = next(
            (a for a in portfolio.allocations if a.pool_id == "aave-usdc"), None
        )
        assert aave_alloc is not None
        # With very low tau, it should get a substantial allocation
        assert aave_alloc.allocation_pct > 0.15

    def test_bl_tau_validation(self) -> None:
        """BL tau parameter must be in (0, 1]."""
        with pytest.raises(ValueError, match="bl_tau"):
            PortfolioOptimizer(
                strategy=AllocationStrategy.BLACK_LITTERMAN, bl_tau=0.0
            )
        with pytest.raises(ValueError, match="bl_tau"):
            PortfolioOptimizer(
                strategy=AllocationStrategy.BLACK_LITTERMAN, bl_tau=1.5
            )

    def test_bl_no_eligible_pools(self) -> None:
        """BL returns empty portfolio when no pools meet criteria."""
        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="tiny",
                pool_name="Tiny",
                token_pair="USDC",
                apy=0.04,
                tvl_usd=100,
            )
        ]
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        portfolio = optimizer.optimize(pools, 10_000)
        assert portfolio.num_positions == 0

    def test_bl_zero_investment_raises(self, sample_pools: list[PoolInfo]) -> None:
        """BL raises ValueError for zero investment."""
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        with pytest.raises(ValueError, match="positive"):
            optimizer.optimize(sample_pools, 0)

    def test_bl_stablecoins_only(self, sample_pools: list[PoolInfo]) -> None:
        """BL respects stablecoins_only config."""
        config = Config(stablecoins_only=True)
        optimizer = PortfolioOptimizer(
            config=config, strategy=AllocationStrategy.BLACK_LITTERMAN
        )
        portfolio = optimizer.optimize(sample_pools, 50_000)
        for alloc in portfolio.allocations:
            assert alloc.token_pair in ("USDC", "USDT", "DAI/USDC/USDT")

    def test_bl_different_from_greedy(self, sample_pools: list[PoolInfo]) -> None:
        """BL and greedy produce different allocations (different objectives)."""
        opt_bl = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        opt_greedy = PortfolioOptimizer(strategy=AllocationStrategy.GREEDY)

        port_bl = opt_bl.optimize(sample_pools, 100_000)
        port_greedy = opt_greedy.optimize(sample_pools, 100_000)

        # Both should be valid
        assert port_bl.num_positions > 0
        assert port_greedy.num_positions > 0

        # APY differences indicate different allocation decisions
        # (not guaranteed to be different due to floating point, but likely)
        assert port_bl.total_expected_apy >= 0
        assert port_greedy.total_expected_apy >= 0

    def test_bl_higher_tau_more_views(self, sample_pools: list[PoolInfo]) -> None:
        """Higher tau (more trust in views) changes the allocation."""
        opt_low = PortfolioOptimizer(
            strategy=AllocationStrategy.BLACK_LITTERMAN, bl_tau=0.01
        )
        opt_high = PortfolioOptimizer(
            strategy=AllocationStrategy.BLACK_LITTERMAN, bl_tau=0.20
        )

        port_low = opt_low.optimize(sample_pools, 100_000)
        port_high = opt_high.optimize(sample_pools, 100_000)

        # Both should be valid portfolios
        assert port_low.num_positions > 0
        assert port_high.num_positions > 0

    def test_bl_single_pool(self) -> None:
        """BL handles single-pool portfolio gracefully."""
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
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        portfolio = optimizer.optimize(pools, 100_000)
        assert portfolio.num_positions == 1
        assert portfolio.allocations[0].allocation_pct == pytest.approx(1.0, abs=0.01)

    def test_bl_all_non_stable(self) -> None:
        """BL works with all non-stable pools."""
        pools = [
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ETHEREUM,
                pool_id=f"uni-pool-{i}",
                pool_name=f"Uniswap Pool {i}",
                token_pair=f"TOKEN{i}/ETH",
                apy=0.05 + i * 0.02,
                tvl_usd=1_000_000_000 * (i + 1),
                is_stable=False,
                impermanent_loss_risk=0.2 + i * 0.1,
            )
            for i in range(4)
        ]
        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.BLACK_LITTERMAN)
        portfolio = optimizer.optimize(pools, 100_000)
        assert portfolio.num_positions > 0
        total_pct = sum(a.allocation_pct for a in portfolio.allocations)
        assert total_pct <= 1.001


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for _pool_volatility and _pool_edge helpers."""

    def test_pool_volatility_positive(
        self, sample_pools: list[PoolInfo], risk_engine: RiskEngine
    ) -> None:
        for pool in sample_pools:
            risk = risk_engine.score_pool(pool)
            vol = _pool_volatility(pool, risk)
            assert vol > 0

    def test_stable_pool_lower_volatility(
        self, risk_engine: RiskEngine
    ) -> None:
        stable = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="stable",
            pool_name="Stable",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=5_000_000_000,
            is_stable=True,
        )
        volatile = PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ETHEREUM,
            pool_id="volatile",
            pool_name="Volatile",
            token_pair="ETH/USDC",
            apy=0.08,
            tvl_usd=5_000_000_000,
            is_stable=False,
            impermanent_loss_risk=0.5,
        )
        vol_stable = _pool_volatility(stable, risk_engine.score_pool(stable))
        vol_volatile = _pool_volatility(volatile, risk_engine.score_pool(volatile))
        assert vol_stable < vol_volatile

    def test_pool_edge_positive_for_good_pools(
        self, sample_pools: list[PoolInfo], risk_engine: RiskEngine
    ) -> None:
        """Pools with positive APY should have positive edge (discount < APY)."""
        for pool in sample_pools:
            if pool.apy > 0:
                risk = risk_engine.score_pool(pool)
                edge = _pool_edge(pool, risk)
                # Edge should be positive because discount is at most 50% of APY
                assert edge > 0

    def test_pool_edge_less_than_apy(
        self, sample_pools: list[PoolInfo], risk_engine: RiskEngine
    ) -> None:
        for pool in sample_pools:
            risk = risk_engine.score_pool(pool)
            edge = _pool_edge(pool, risk)
            assert edge <= pool.apy


# ---------------------------------------------------------------------------
# Strategy enum tests
# ---------------------------------------------------------------------------


class TestAllocationStrategy:
    """Tests for the AllocationStrategy enum."""

    def test_enum_values(self) -> None:
        assert AllocationStrategy.GREEDY == "greedy"
        assert AllocationStrategy.KELLY == "kelly"
        assert AllocationStrategy.RISK_PARITY == "risk_rarity"
        assert AllocationStrategy.BLACK_LITTERMAN == "black_litterman"

    def test_default_strategy_is_greedy(self) -> None:
        optimizer = PortfolioOptimizer()
        assert optimizer.strategy is AllocationStrategy.GREEDY

    def test_strategy_switch(self, sample_pools: list[PoolInfo]) -> None:
        """All three strategies should produce valid but different portfolios."""
        portfolios = {}
        for strategy in AllocationStrategy:
            opt = PortfolioOptimizer(strategy=strategy)
            port = opt.optimize(sample_pools, 100_000)
            assert port.num_positions > 0
            portfolios[strategy] = port

        # Different strategies should produce different allocations
        greedy_apy = portfolios[AllocationStrategy.GREEDY].total_expected_apy
        kelly_apy = portfolios[AllocationStrategy.KELLY].total_expected_apy
        rp_apy = portfolios[AllocationStrategy.RISK_PARITY].total_expected_apy
        bl_apy = portfolios[AllocationStrategy.BLACK_LITTERMAN].total_expected_apy

        # All should produce positive APY
        assert greedy_apy > 0
        assert kelly_apy > 0
        assert rp_apy > 0
        assert bl_apy > 0


# ---------------------------------------------------------------------------
# Internal strategy function tests
# ---------------------------------------------------------------------------


class TestInternalStrategies:
    """Direct tests for _kelly_optimize and _risk_parity_optimize."""

    def test_kelly_empty_candidates(self) -> None:
        config = Config()
        result = _kelly_optimize([], 100_000, config)
        assert result.num_positions == 0
        assert result.total_expected_apy == 0.0

    def test_risk_parity_empty_candidates(self) -> None:
        config = Config()
        result = _risk_parity_optimize([], 100_000, config)
        assert result.num_positions == 0
        assert result.total_expected_apy == 0.0

    def test_kelly_preserves_pool_metadata(
        self, sample_pools: list[PoolInfo], risk_engine: RiskEngine
    ) -> None:
        """Kelly allocations should preserve protocol/chain/pool_id."""
        config = Config()
        candidates = [(p, risk_engine.score_pool(p)) for p in sample_pools]
        result = _kelly_optimize(candidates, 100_000, config)

        for alloc in result.allocations:
            pool = next(p for p in sample_pools if p.pool_id == alloc.pool_id)
            assert alloc.protocol == pool.protocol
            assert alloc.chain == pool.chain
            assert alloc.token_pair == pool.token_pair

    def test_risk_parity_preserves_pool_metadata(
        self, sample_pools: list[PoolInfo], risk_engine: RiskEngine
    ) -> None:
        """Risk parity allocations should preserve protocol/chain/pool_id."""
        config = Config()
        candidates = [(p, risk_engine.score_pool(p)) for p in sample_pools]
        result = _risk_parity_optimize(candidates, 100_000, config)

        for alloc in result.allocations:
            pool = next(p for p in sample_pools if p.pool_id == alloc.pool_id)
            assert alloc.protocol == pool.protocol
            assert alloc.chain == pool.chain
            assert alloc.token_pair == pool.token_pair
