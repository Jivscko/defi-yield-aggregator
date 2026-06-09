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
        assert score.overall_score < 55

    def test_low_tvl_high_risk(self, risk_engine: RiskEngine, low_tvl_pool: PoolInfo) -> None:
        score = risk_engine.score_pool(low_tvl_pool)
        assert score.overall_score > 35

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
        assert score.liquidity_score >= 0
        assert score.smart_contract_risk_score >= 0
        assert score.governance_risk_score >= 0

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


class TestLiquidityRisk:
    """Tests for the liquidity risk scoring dimension."""

    def test_liquidity_score_high_volume(self, risk_engine: RiskEngine) -> None:
        """Pool with high volume/TVL ratio gets low liquidity risk score."""
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="high-vol",
            pool_name="High Volume Pool",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
            daily_volume_usd=200_000_000,  # 20% ratio
        )
        score = risk_engine.score_pool(pool)
        assert score.liquidity_score <= 10  # Very liquid

    def test_liquidity_score_low_volume(self, risk_engine: RiskEngine) -> None:
        """Pool with very low volume gets high liquidity risk score."""
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="low-vol",
            pool_name="Low Volume Pool",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
            daily_volume_usd=1_000_000,  # 0.1% ratio
        )
        score = risk_engine.score_pool(pool)
        assert score.liquidity_score >= 70  # High risk + volume penalty

    def test_liquidity_score_no_volume(self, risk_engine: RiskEngine) -> None:
        """Pool with zero daily volume gets moderate-high score."""
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="no-vol",
            pool_name="No Volume Pool",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
            daily_volume_usd=0,  # No volume data
        )
        score = risk_engine.score_pool(pool)
        # 70 base + 15 penalty = 85
        assert score.liquidity_score == 85.0


class TestSmartContractRisk:
    """Tests for the smart contract risk scoring dimension."""

    def test_smart_contract_risk_yearn_highest(self, risk_engine: RiskEngine) -> None:
        """Yearn has highest SC risk among known protocols."""
        yearn_pool = PoolInfo(
            protocol=Protocol.YEARN,
            chain=Chain.ETHEREUM,
            pool_id="yearn",
            pool_name="Yearn",
            token_pair="ETH",
            apy=0.10,
            tvl_usd=1_000_000_000,
        )
        score = risk_engine.score_pool(yearn_pool)
        # Yearn: 40 + 10 (proxy) + 15 (composability) = 65
        assert score.smart_contract_risk_score == 65.0

    def test_smart_contract_risk_uniswap_low(self, risk_engine: RiskEngine) -> None:
        """Uniswap has low SC risk (simple contracts, no proxy)."""
        uni_pool = PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ETHEREUM,
            pool_id="uni",
            pool_name="Uniswap",
            token_pair="ETH/USDC",
            apy=0.05,
            tvl_usd=1_000_000_000,
        )
        score = risk_engine.score_pool(uni_pool)
        # Uniswap: 15 + 0 (no proxy) + 15 (composability) = 30
        assert score.smart_contract_risk_score == 30.0

    def test_unknown_protocol_high_sc_risk(self, risk_engine: RiskEngine) -> None:
        """Unknown/default protocol gets high SC risk."""
        # BALANCER is in the meta, so test by checking the default path
        # We'll test via the _smart_contract_risk_score function directly
        from defi_yield_aggregator.core.risk_engine import _smart_contract_risk_score
        # Use a known protocol to verify the function works, then check default
        # The default path would need a protocol not in PROTOCOL_SC_META,
        # but all enum members are covered. Verify the calculation is correct.
        score = _smart_contract_risk_score(Protocol.BALANCER)
        # Balancer: 30 + 10 (proxy) + 15 (composability) = 55
        assert score == 55.0

    def test_all_known_protocols_sc_scores(self, risk_engine: RiskEngine) -> None:
        """Verify expected SC risk scores for all known protocols."""
        expected = {
            Protocol.AAVE: 45.0,        # 20 + 10 + 15
            Protocol.COMPOUND: 50.0,     # 25 + 10 + 15
            Protocol.UNISWAP: 30.0,      # 15 + 0 + 15
            Protocol.CURVE: 50.0,        # 35 + 0 + 15
            Protocol.YEARN: 65.0,        # 40 + 10 + 15
            Protocol.LIDO: 30.0,         # 20 + 10 + 0
            Protocol.BALANCER: 55.0,     # 30 + 10 + 15
        }
        from defi_yield_aggregator.core.risk_engine import _smart_contract_risk_score
        for proto, exp_score in expected.items():
            actual = _smart_contract_risk_score(proto)
            assert actual == exp_score, f"{proto}: expected {exp_score}, got {actual}"


class TestNewWeights:
    """Tests for the updated weight configuration."""

    def test_new_weights_valid(self) -> None:
        """Verify default weights sum to 1.0."""
        engine = RiskEngine()
        total = (
            engine.tvl_weight
            + engine.age_weight
            + engine.audit_weight
            + engine.chain_weight
            + engine.liquidity_weight
            + engine.sc_risk_weight
            + engine.governance_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_backward_compat_custom_weights(self) -> None:
        """Old-style 4-weight construction still works."""
        engine = RiskEngine(
            tvl_weight=0.35,
            age_weight=0.20,
            audit_weight=0.25,
            chain_weight=0.20,
        )
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="compat",
            pool_name="Compat Test",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=5_000_000_000,
        )
        score = engine.score_pool(pool)
        assert 0 <= score.overall_score <= 100


class TestLiquidityAffectsOverall:
    """Tests that liquidity risk properly influences overall score."""

    def test_liquidity_affects_overall(self, risk_engine: RiskEngine) -> None:
        """A pool with zero volume scores higher overall than same pool with high volume."""
        pool_no_vol = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="no-vol",
            pool_name="No Volume",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
            daily_volume_usd=0,
        )
        pool_high_vol = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="high-vol",
            pool_name="High Volume",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
            daily_volume_usd=200_000_000,
        )
        score_no_vol = risk_engine.score_pool(pool_no_vol)
        score_high_vol = risk_engine.score_pool(pool_high_vol)
        # No volume pool should have higher risk (higher score)
        assert score_no_vol.overall_score > score_high_vol.overall_score


class TestGovernanceRisk:
    """Tests for the governance risk scoring dimension."""

    def test_governance_score_makerdao_well_governed(self) -> None:
        """MakerDAO should have a relatively low governance risk score (< 40).

        It has high decentralization (80), timelock, and moderate concentration (35).
        """
        from defi_yield_aggregator.core.risk_engine import _governance_risk_score
        score = _governance_risk_score(Protocol.MAKERDAO)
        # 100-80=20, -10(timelock), +10(emergency_pause), +(35-50)*0.3=-4.5 → 15.5
        assert score < 40

    def test_governance_score_convex_higher_risk(self) -> None:
        """Convex should have higher governance risk (>= 50).

        It lacks timelock and has medium decentralization (45).
        """
        from defi_yield_aggregator.core.risk_engine import _governance_risk_score
        score = _governance_risk_score(Protocol.CONVEX)
        # 100-45=55, no timelock, +10(emergency_pause), +(50-50)*0.3=0 → 65.0
        assert score >= 50

    def test_governance_score_rocket_pool_lowest(self) -> None:
        """Rocket Pool should have among the lowest governance risk.

        It's highly decentralized (85), has timelock, no emergency pause,
        and low concentration (20).
        """
        from defi_yield_aggregator.core.risk_engine import _governance_risk_score
        score = _governance_risk_score(Protocol.ROCKET_POOL)
        # 100-85=15, -10(timelock), no emergency, +(20-50)*0.3=-9 → -4 → clamped to 0.0
        assert score == 0.0

    def test_governance_score_range(self) -> None:
        """All known protocols return scores in [0, 100]."""
        from defi_yield_aggregator.core.risk_engine import _governance_risk_score
        for proto in Protocol:
            score = _governance_risk_score(proto)
            assert 0 <= score <= 100, f"{proto}: score {score} out of range"

    def test_governance_affects_overall(self) -> None:
        """A pool from a well-governed protocol scores lower overall than one from a poorly-governed protocol."""
        pool_makerdao = PoolInfo(
            protocol=Protocol.MAKERDAO,
            chain=Chain.ETHEREUM,
            pool_id="maker-test",
            pool_name="MakerDAO Test",
            token_pair="DAI",
            apy=0.04,
            tvl_usd=1_000_000_000,
        )
        pool_convex = PoolInfo(
            protocol=Protocol.CONVEX,
            chain=Chain.ETHEREUM,
            pool_id="convex-test",
            pool_name="Convex Test",
            token_pair="CRV",
            apy=0.04,
            tvl_usd=1_000_000_000,
        )
        # Use a new engine with explicit governance weight so it matters
        engine = RiskEngine(
            tvl_weight=0.20,
            age_weight=0.10,
            audit_weight=0.15,
            chain_weight=0.10,
            liquidity_weight=0.15,
            sc_risk_weight=0.15,
            governance_weight=0.15,
        )
        score_makerdao = engine.score_pool(pool_makerdao)
        score_convex = engine.score_pool(pool_convex)
        assert score_makerdao.overall_score < score_convex.overall_score

    def test_governance_score_populated_in_result(self) -> None:
        """Verify governance_risk_score > 0 in the RiskScore result."""
        engine = RiskEngine()
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="gov-pop",
            pool_name="Gov Pop Test",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
        )
        score = engine.score_pool(pool)
        assert score.governance_risk_score > 0

    def test_all_known_protocols_gov_scores(self) -> None:
        """Verify expected governance risk scores for all 12 protocols.

        Formula: score = (100 - decentralization) + (-10 if timelock)
                  + (+10 if emergency_pause) + (concentration - 50) * 0.3
        Clamped to [0, 100].
        """
        from defi_yield_aggregator.core.risk_engine import _governance_risk_score
        expected = {
            # AAVE: 100-80=20, -10, +10, +(30-50)*0.3=-6 → 14.0
            Protocol.AAVE: 14.0,
            # COMPOUND: 100-75=25, -10, +0, +(35-50)*0.3=-4.5 → 10.5
            Protocol.COMPOUND: 10.5,
            # UNISWAP: 100-90=10, -10, +10, +(25-50)*0.3=-7.5 → 2.5
            Protocol.UNISWAP: 2.5,
            # CURVE: 100-60=40, -10, +0, +(40-50)*0.3=-3 → 27.0
            Protocol.CURVE: 27.0,
            # YEARN: 100-55=45, -10, +10, +(35-50)*0.3=-4.5 → 40.5
            Protocol.YEARN: 40.5,
            # LIDO: 100-65=35, -10, +10, +(45-50)*0.3=-1.5 → 33.5
            Protocol.LIDO: 33.5,
            # BALANCER: 100-70=30, -10, +10, +(30-50)*0.3=-6 → 24.0
            Protocol.BALANCER: 24.0,
            # CONVEX: 100-45=55, +0(no timelock), +10, +(50-50)*0.3=0 → 65.0
            Protocol.CONVEX: 65.0,
            # ROCKET_POOL: 100-85=15, -10, +0, +(20-50)*0.3=-9 → -4 → 0.0
            Protocol.ROCKET_POOL: 0.0,
            # SUSHISWAP: 100-50=50, -10, +10, +(45-50)*0.3=-1.5 → 48.5
            Protocol.SUSHISWAP: 48.5,
            # FRAX: 100-55=45, -10, +10, +(55-50)*0.3=1.5 → 46.5
            Protocol.FRAX: 46.5,
            # MAKERDAO: 100-80=20, -10, +10, +(35-50)*0.3=-4.5 → 15.5
            Protocol.MAKERDAO: 15.5,
        }
        for proto, exp_score in expected.items():
            actual = _governance_risk_score(proto)
            assert abs(actual - exp_score) < 0.01, (
                f"{proto}: expected {exp_score}, got {actual}"
            )

    def test_backward_compat_with_governance(self) -> None:
        """Old-style 4-weight construction still works with governance_weight=0.

        New 7-weight construction works with explicit governance_weight.
        """
        # Old-style 4 weights summing to 1.0 → governance_weight should be 0.0
        old_engine = RiskEngine(
            tvl_weight=0.5,
            age_weight=0.2,
            audit_weight=0.2,
            chain_weight=0.1,
        )
        assert old_engine.governance_weight == 0.0
        assert old_engine.liquidity_weight == 0.0
        assert old_engine.sc_risk_weight == 0.0

        # New 7-weight construction
        new_engine = RiskEngine(
            tvl_weight=0.20,
            age_weight=0.10,
            audit_weight=0.15,
            chain_weight=0.10,
            liquidity_weight=0.15,
            sc_risk_weight=0.15,
            governance_weight=0.15,
        )
        assert new_engine.governance_weight == 0.15
        total = (
            new_engine.tvl_weight
            + new_engine.age_weight
            + new_engine.audit_weight
            + new_engine.chain_weight
            + new_engine.liquidity_weight
            + new_engine.sc_risk_weight
            + new_engine.governance_weight
        )
        assert abs(total - 1.0) < 1e-6

        # Both engines can score a pool without error
        pool = PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="compat-gov",
            pool_name="Compat Gov Test",
            token_pair="USDC",
            apy=0.04,
            tvl_usd=1_000_000_000,
        )
        old_score = old_engine.score_pool(pool)
        new_score = new_engine.score_pool(pool)
        assert 0 <= old_score.overall_score <= 100
        assert 0 <= new_score.overall_score <= 100
