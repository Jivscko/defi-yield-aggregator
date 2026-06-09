"""Comprehensive tests for the Pendle Finance protocol adapter."""

from __future__ import annotations

import pytest

from defi_yield_aggregator.adapters.protocols import PendleAdapter, get_all_adapters
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


@pytest.fixture
def adapter() -> PendleAdapter:
    return PendleAdapter()


class TestPendleAdapter:
    """Tests for Pendle adapter fetch_pools and fetch_pool_detail."""

    def test_adapter_instantiation(self, adapter: PendleAdapter) -> None:
        """Adapter should instantiate with correct protocol and default URL."""
        assert adapter.protocol == Protocol.PENDLE
        assert adapter.base_url == "https://api-v2.pendle.finance"

    def test_repr(self, adapter: PendleAdapter) -> None:
        assert "PendleAdapter" in repr(adapter)
        assert "pendle" in repr(adapter)

    def test_protocol_attribute(self, adapter: PendleAdapter) -> None:
        assert adapter.protocol == Protocol.PENDLE

    def test_base_url(self, adapter: PendleAdapter) -> None:
        assert adapter.base_url == "https://api-v2.pendle.finance"

    @pytest.mark.asyncio
    async def test_fetch_pools_returns_correct_count(self, adapter: PendleAdapter) -> None:
        """Should return 7 pools (4 Ethereum + 3 Arbitrum)."""
        pools = await adapter.fetch_pools()
        assert len(pools) == 7

    @pytest.mark.asyncio
    async def test_fetch_pools_returns_non_empty(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_all_pools_have_pendle_protocol(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        assert all(p.protocol == Protocol.PENDLE for p in pools)

    @pytest.mark.asyncio
    async def test_all_pools_have_positive_apy(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert pool.apy > 0, f"{pool.pool_id} has non-positive APY: {pool.apy}"

    @pytest.mark.asyncio
    async def test_all_pools_have_positive_tvl(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert pool.tvl_usd > 0, f"{pool.pool_id} has non-positive TVL: {pool.tvl_usd}"

    @pytest.mark.asyncio
    async def test_pool_ids_unique(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        ids = [p.pool_id for p in pools]
        assert len(ids) == len(set(ids)), f"Duplicate pool IDs: {ids}"

    @pytest.mark.asyncio
    async def test_chain_filter_ethereum(self, adapter: PendleAdapter) -> None:
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)
        assert len(eth_pools) == 4

    @pytest.mark.asyncio
    async def test_chain_filter_arbitrum(self, adapter: PendleAdapter) -> None:
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert all(p.chain == Chain.ARBITRUM for p in arb_pools)
        assert len(arb_pools) == 3

    @pytest.mark.asyncio
    async def test_chain_filter_returns_empty_for_unsupported(self, adapter: PendleAdapter) -> None:
        """No Pendle pools on Polygon in mock data."""
        poly_pools = await adapter.fetch_pools(chain=Chain.POLYGON)
        assert poly_pools == []

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_found(self, adapter: PendleAdapter) -> None:
        detail = await adapter.fetch_pool_detail("pendle-eth-staking-pt-eth")
        assert detail is not None
        assert detail.pool_id == "pendle-eth-staking-pt-eth"
        assert detail.protocol == Protocol.PENDLE

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_not_found(self, adapter: PendleAdapter) -> None:
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_all_pools(self, adapter: PendleAdapter) -> None:
        """Every pool returned by fetch_pools should be retrievable by ID."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            detail = await adapter.fetch_pool_detail(pool.pool_id)
            assert detail is not None
            assert detail.pool_id == pool.pool_id

    @pytest.mark.asyncio
    async def test_eth_staking_pt_exists(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        pool = next((p for p in pools if p.pool_id == "pendle-eth-staking-pt-eth"), None)
        assert pool is not None
        assert pool.chain == Chain.ETHEREUM
        assert pool.apy == pytest.approx(0.0980)
        assert pool.tvl_usd == 420_000_000

    @pytest.mark.asyncio
    async def test_steth_pt_exists(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        pool = next((p for p in pools if p.pool_id == "pendle-steth-pt-eth"), None)
        assert pool is not None
        assert pool.chain == Chain.ETHEREUM
        assert pool.apy == pytest.approx(0.0640)

    @pytest.mark.asyncio
    async def test_usdc_lp_ethereum_exists(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        pool = next((p for p in pools if p.pool_id == "pendle-usdc-lp-eth"), None)
        assert pool is not None
        assert pool.is_stable is True

    @pytest.mark.asyncio
    async def test_reth_yt_exists(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        pool = next((p for p in pools if p.pool_id == "pendle-reth-yt-eth"), None)
        assert pool is not None
        assert pool.chain == Chain.ETHEREUM
        assert pool.apy == pytest.approx(0.1450)

    @pytest.mark.asyncio
    async def test_glp_pt_arbitrum_exists(self, adapter: PendleAdapter) -> None:
        pools = await adapter.fetch_pools()
        pool = next((p for p in pools if p.pool_id == "pendle-glp-pt-arb"), None)
        assert pool is not None
        assert pool.chain == Chain.ARBITRUM
        assert pool.apy == pytest.approx(0.1920)

    @pytest.mark.asyncio
    async def test_lp_pools_have_higher_il_risk_than_pt_yt(self, adapter: PendleAdapter) -> None:
        """LP pools should have higher impermanent loss risk than PT/YT pools."""
        pools = await adapter.fetch_pools()
        lp_pools = [p for p in pools if "LP" in p.pool_name]
        pt_yt_pools = [p for p in pools if "LP" not in p.pool_name]
        avg_lp_il = sum(p.impermanent_loss_risk for p in lp_pools) / len(lp_pools)
        avg_pt_yt_il = sum(p.impermanent_loss_risk for p in pt_yt_pools) / len(pt_yt_pools)
        assert avg_lp_il > avg_pt_yt_il

    @pytest.mark.asyncio
    async def test_apy_values_reasonable(self, adapter: PendleAdapter) -> None:
        """All APY values should be between 0% and 25%."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert 0 < pool.apy < 0.25, f"{pool.pool_id} has unreasonable APY: {pool.apy}"

    @pytest.mark.asyncio
    async def test_pool_info_model_validation(self, adapter: PendleAdapter) -> None:
        """All pools should validate against the PoolInfo Pydantic model."""
        pools = await adapter.fetch_pools()
        for pool in pools:
            assert isinstance(pool, PoolInfo)
            assert pool.last_updated is not None


class TestPendleInRegistry:
    """Verify Pendle is properly registered in the adapter system."""

    def test_in_adapter_registry(self) -> None:
        from defi_yield_aggregator.adapters.protocols import ADAPTERS
        assert Protocol.PENDLE in ADAPTERS

    @pytest.mark.asyncio
    async def test_get_all_adapters_includes_pendle(self) -> None:
        adapters = get_all_adapters()
        protocols = [a.protocol for a in adapters]
        assert Protocol.PENDLE in protocols

    def test_protocol_enum_member(self) -> None:
        assert Protocol.PENDLE.value == "pendle"

    def test_pendle_in_init_exports(self) -> None:
        from defi_yield_aggregator.adapters import PendleAdapter as InitPendle
        assert InitPendle is PendleAdapter
