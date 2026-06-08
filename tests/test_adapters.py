"""Tests for protocol adapters."""

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import (
    AaveAdapter,
    CompoundAdapter,
    CurveAdapter,
    LidoAdapter,
    UniswapAdapter,
    YearnAdapter,
    get_all_adapters,
)
from defi_yield_aggregator.core.models import Chain, Protocol


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestAdapters:
    """Tests for protocol adapters."""

    @pytest.mark.asyncio
    async def test_aave_fetch_pools(self) -> None:
        adapter = AaveAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) > 0
        assert all(p.protocol == Protocol.AAVE for p in pools)

    @pytest.mark.asyncio
    async def test_compound_fetch_pools(self) -> None:
        adapter = CompoundAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) > 0
        assert all(p.protocol == Protocol.COMPOUND for p in pools)

    @pytest.mark.asyncio
    async def test_uniswap_fetch_pools(self) -> None:
        adapter = UniswapAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_curve_fetch_pools(self) -> None:
        adapter = CurveAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_yearn_fetch_pools(self) -> None:
        adapter = YearnAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_chain_filter(self) -> None:
        adapter = AaveAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_fetch_pool_detail(self) -> None:
        adapter = AaveAdapter()
        pools = await adapter.fetch_pools()
        detail = await adapter.fetch_pool_detail(pools[0].pool_id)
        assert detail is not None
        assert detail.pool_id == pools[0].pool_id

    @pytest.mark.asyncio
    async def test_fetch_pool_detail_not_found(self) -> None:
        adapter = AaveAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    def test_get_all_adapters(self) -> None:
        adapters = get_all_adapters()
        assert len(adapters) == 6

    def test_adapter_repr(self) -> None:
        adapter = AaveAdapter()
        assert "AaveAdapter" in repr(adapter)
        assert "aave" in repr(adapter)

    @pytest.mark.asyncio
    async def test_lido_fetch_pools(self) -> None:
        adapter = LidoAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 3
        assert all(p.protocol == Protocol.LIDO for p in pools)

    @pytest.mark.asyncio
    async def test_lido_fetch_pools_chain_filter(self) -> None:
        adapter = LidoAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 1
        assert eth_pools[0].pool_id == "lido-steth-eth"
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_lido_fetch_pool_detail(self) -> None:
        adapter = LidoAdapter()
        detail = await adapter.fetch_pool_detail("lido-steth-eth")
        assert detail is not None
        assert detail.pool_id == "lido-steth-eth"
        assert detail.tvl_usd == 14_000_000_000
        assert detail.apy == pytest.approx(0.032)

    @pytest.mark.asyncio
    async def test_lido_fetch_pool_detail_not_found(self) -> None:
        adapter = LidoAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_lido_pools_have_zero_il_risk(self) -> None:
        adapter = LidoAdapter()
        pools = await adapter.fetch_pools()
        assert all(p.impermanent_loss_risk == 0.0 for p in pools)
