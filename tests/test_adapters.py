"""Tests for protocol adapters."""

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import (
    AaveAdapter,
    CompoundAdapter,
    CurveAdapter,
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
        assert len(adapters) == 5

    def test_adapter_repr(self) -> None:
        adapter = AaveAdapter()
        assert "AaveAdapter" in repr(adapter)
        assert "aave" in repr(adapter)
