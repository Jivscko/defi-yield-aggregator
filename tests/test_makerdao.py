"""Comprehensive tests for the MakerDAO DSR / SparkLend protocol adapter."""

from __future__ import annotations

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import MakerDAOAdapter, get_all_adapters
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol
from defi_yield_aggregator.core.risk_engine import (
    PROTOCOL_META,
    PROTOCOL_SC_META,
    RiskEngine,
)


@pytest.fixture
def adapter() -> MakerDAOAdapter:
    return MakerDAOAdapter()


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


class TestMakerDAOAdapter:
    """Tests for MakerDAO adapter fetch_pools and fetch_pool_detail."""

    @pytest.mark.asyncio
    async def test_fetch_pools_returns_non_empty(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_all_pools_have_makerdao_protocol(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        assert all(p.protocol == Protocol.MAKERDAO for p in pools)

    @pytest.mark.asyncio
    async def test_pool_ids_unique(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        ids = [p.pool_id for p in pools]
        assert len(ids) == len(set(ids)), f"Duplicate pool IDs: {ids}"

    @pytest.mark.asyncio
    async def test_sdai_eth_exists(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        sdai = next((p for p in pools if p.pool_id == "makerdao-sdai-eth"), None)
        assert sdai is not None
        assert sdai.pool_name == "MakerDAO sDAI Savings (ERC-4626)"
        assert sdai.token_pair == "sDAI"
        assert sdai.is_stable is True
        assert sdai.impermanent_loss_risk == 0.0
        assert sdai.tvl_usd == 3_200_000_000
        assert sdai.apy == pytest.approx(0.05)
        assert sdai.deposit_fee == 0.0
        assert sdai.withdrawal_fee == 0.0

    @pytest.mark.asyncio
    async def test_sparklend_dai_eth(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        spark = next((p for p in pools if p.pool_id == "makerdao-spark-dai-eth"), None)
        assert spark is not None
        assert "SparkLend" in spark.pool_name
        assert spark.is_stable is True
        assert spark.tvl_usd == 1_800_000_000

    @pytest.mark.asyncio
    async def test_sparklend_weth_is_not_stable(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        weth = next((p for p in pools if p.pool_id == "makerdao-spark-weth-eth"), None)
        assert weth is not None
        assert weth.is_stable is False

    @pytest.mark.asyncio
    async def test_chain_filter_ethereum(self, adapter: MakerDAOAdapter) -> None:
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)
        assert len(eth_pools) > 0

    @pytest.mark.asyncio
    async def test_chain_filter_arbitrum(self, adapter: MakerDAOAdapter) -> None:
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert all(p.chain == Chain.ARBITRUM for p in arb_pools)
        assert len(arb_pools) >= 1

    @pytest.mark.asyncio
    async def test_chain_filter_returns_empty_for_unsupported(self, adapter: MakerDAOAdapter) -> None:
        # BASE has no MakerDAO pools in mock data
        base_pools = await adapter.fetch_pools(chain=Chain.BASE)
        assert base_pools == []

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_found(self, adapter: MakerDAOAdapter) -> None:
        detail = await adapter.fetch_pool_detail("makerdao-sdai-eth")
        assert detail is not None
        assert detail.pool_id == "makerdao-sdai-eth"
        assert detail.protocol == Protocol.MAKERDAO

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_not_found(self, adapter: MakerDAOAdapter) -> None:
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_all_pools(self, adapter: MakerDAOAdapter) -> None:
        """Every pool returned by fetch_pools should be retrievable by ID."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            detail = await adapter.fetch_pool_detail(pool.pool_id)
            assert detail is not None
            assert detail.pool_id == pool.pool_id

    @pytest.mark.asyncio
    async def test_apy_values_reasonable(self, adapter: MakerDAOAdapter) -> None:
        """All APY values should be between 0% and 10%."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert 0 < pool.apy < 0.10, f"{pool.pool_id} has unreasonable APY: {pool.apy}"

    @pytest.mark.asyncio
    async def test_tvl_positive(self, adapter: MakerDAOAdapter) -> None:
        pools = await adapter.fetch_pools()
        assert all(p.tvl_usd > 0 for p in pools)

    @pytest.mark.asyncio
    async def test_sdai_pools_zero_il_risk(self, adapter: MakerDAOAdapter) -> None:
        """sDAI and lending pools have no impermanent loss."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert pool.impermanent_loss_risk == 0.0, (
                f"{pool.pool_id} should have zero IL risk"
            )

    @pytest.mark.asyncio
    async def test_zero_fees(self, adapter: MakerDAOAdapter) -> None:
        """All MakerDAO/SparkLend pools should have zero deposit/withdrawal fees."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert pool.deposit_fee == 0.0
            assert pool.withdrawal_fee == 0.0

    def test_repr(self, adapter: MakerDAOAdapter) -> None:
        assert "MakerDAOAdapter" in repr(adapter)
        assert "makerdao" in repr(adapter)

    def test_base_url(self, adapter: MakerDAOAdapter) -> None:
        assert adapter.base_url == "https://api.makerdao.com"

    def test_protocol_attribute(self, adapter: MakerDAOAdapter) -> None:
        assert adapter.protocol == Protocol.MAKERDAO


class TestMakerDAOInRegistry:
    """Verify MakerDAO is properly registered in the adapter system."""

    def test_in_adapter_registry(self) -> None:
        from defi_yield_aggregator.adapters.protocols import ADAPTERS

        assert Protocol.MAKERDAO in ADAPTERS

    @pytest.mark.asyncio
    async def test_get_all_adapters_includes_makerdao(self) -> None:
        adapters = get_all_adapters()
        protocols = [a.protocol for a in adapters]
        assert Protocol.MAKERDAO in protocols

    def test_protocol_enum_member(self) -> None:
        assert Protocol.MAKERDAO.value == "makerdao"


class TestMakerDAORiskScoring:
    """Verify risk engine correctly scores MakerDAO pools."""

    @pytest.mark.asyncio
    async def test_sdai_risk_score_low(self, engine: RiskEngine) -> None:
        """sDAI vault should have LOW risk given MakerDAO's maturity."""
        pool = PoolInfo(
            protocol=Protocol.MAKERDAO,
            chain=Chain.ETHEREUM,
            pool_id="makerdao-sdai-eth",
            pool_name="MakerDAO sDAI Savings",
            token_pair="sDAI",
            apy=0.05,
            tvl_usd=3_200_000_000,
            is_stable=True,
            impermanent_loss_risk=0.0,
        )
        score = engine.score_pool(pool)
        assert score.risk_level.value == "low"
        assert score.overall_score < 30

    @pytest.mark.asyncio
    async def test_sparklend_weth_risk_higher_than_sdai(self, engine: RiskEngine) -> None:
        """WETH lending should be riskier than sDAI (not stable, lower TVL)."""
        sdai = PoolInfo(
            protocol=Protocol.MAKERDAO,
            chain=Chain.ETHEREUM,
            pool_id="makerdao-sdai-eth",
            pool_name="sDAI",
            token_pair="sDAI",
            apy=0.05,
            tvl_usd=3_200_000_000,
            is_stable=True,
            impermanent_loss_risk=0.0,
        )
        weth = PoolInfo(
            protocol=Protocol.MAKERDAO,
            chain=Chain.ETHEREUM,
            pool_id="makerdao-spark-weth-eth",
            pool_name="SparkLend WETH",
            token_pair="WETH",
            apy=0.0195,
            tvl_usd=480_000_000,
            is_stable=False,
            impermanent_loss_risk=0.0,
        )
        sdai_score = engine.score_pool(sdai)
        weth_score = engine.score_pool(weth)
        assert weth_score.overall_score > sdai_score.overall_score

    def test_makerdao_in_protocol_meta(self) -> None:
        assert Protocol.MAKERDAO in PROTOCOL_META
        year, audits, chains = PROTOCOL_META[Protocol.MAKERDAO]
        assert year == 2017
        assert audits == 10
        assert chains == 3

    def test_makerdao_in_sc_meta(self) -> None:
        assert Protocol.MAKERDAO in PROTOCOL_SC_META
        complexity, has_proxy, has_composability = PROTOCOL_SC_META[Protocol.MAKERDAO]
        assert complexity == 20
        assert has_proxy is True
        assert has_composability is False


class TestMakerDAOIntegration:
    """Integration-level tests combining adapter with risk engine."""

    @pytest.mark.asyncio
    async def test_all_pools_scoring_succeeds(self) -> None:
        """Risk engine should handle all MakerDAO pools without errors."""
        adapter = MakerDAOAdapter()
        engine = RiskEngine()
        pools = await adapter.fetch_pools()
        for pool in pools:
            score = engine.score_pool(pool)
            assert 0 <= score.overall_score <= 100
            assert score.risk_level.value in ("low", "medium", "high", "critical")

    @pytest.mark.asyncio
    async def test_filter_by_risk_includes_sdai(self) -> None:
        """sDAI should pass a conservative risk filter."""
        adapter = MakerDAOAdapter()
        engine = RiskEngine()
        pools = await adapter.fetch_pools()
        filtered = engine.filter_by_risk(pools, max_risk=40)
        filtered_ids = [p.pool_id for p, _ in filtered]
        assert "makerdao-sdai-eth" in filtered_ids

    @pytest.mark.asyncio
    async def test_pool_info_model_validation(self) -> None:
        """All pools should validate against the PoolInfo Pydantic model."""
        adapter = MakerDAOAdapter()
        pools = await adapter.fetch_pools()
        for pool in pools:
            # Pydantic validation happens at construction — if we got here, it passed
            assert isinstance(pool, PoolInfo)
            assert pool.last_updated is not None
