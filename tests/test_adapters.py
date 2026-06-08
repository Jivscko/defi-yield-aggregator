"""Tests for protocol adapters."""

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import (
    AaveAdapter,
    BalancerAdapter,
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
        assert len(adapters) == 7

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

    # --- Balancer tests ---

    @pytest.mark.asyncio
    async def test_balancer_fetch_pools(self) -> None:
        adapter = BalancerAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 5
        assert all(p.protocol == Protocol.BALANCER for p in pools)

    @pytest.mark.asyncio
    async def test_balancer_fetch_pools_chain_filter_eth(self) -> None:
        adapter = BalancerAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 3
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_balancer_fetch_pools_chain_filter_arb(self) -> None:
        adapter = BalancerAdapter()
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert len(arb_pools) == 1
        assert arb_pools[0].pool_id == "balancer-weth-usdc-arb"

    @pytest.mark.asyncio
    async def test_balancer_fetch_pools_chain_filter_poly(self) -> None:
        adapter = BalancerAdapter()
        poly_pools = await adapter.fetch_pools(chain=Chain.POLYGON)
        assert len(poly_pools) == 1
        assert poly_pools[0].pool_id == "balancer-boosted-aave-poly"

    @pytest.mark.asyncio
    async def test_balancer_fetch_pool_detail(self) -> None:
        adapter = BalancerAdapter()
        detail = await adapter.fetch_pool_detail("balancer-wsteth-weth-5050")
        assert detail is not None
        assert detail.pool_id == "balancer-wsteth-weth-5050"
        assert detail.tvl_usd == 1_800_000_000
        assert detail.apy == pytest.approx(0.034)

    @pytest.mark.asyncio
    async def test_balancer_fetch_pool_detail_not_found(self) -> None:
        adapter = BalancerAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_balancer_stable_pools(self) -> None:
        """Stable pools should have low IL risk."""
        adapter = BalancerAdapter()
        stable_pools = [p for p in await adapter.fetch_pools() if p.is_stable]
        assert len(stable_pools) >= 2
        assert all(p.impermanent_loss_risk < 0.01 for p in stable_pools)

    @pytest.mark.asyncio
    async def test_balancer_weighted_pools_higher_il(self) -> None:
        """Weighted pools (non-stable) should have higher IL risk."""
        adapter = BalancerAdapter()
        weighted = [p for p in await adapter.fetch_pools() if not p.is_stable]
        assert len(weighted) >= 2
        assert any(p.impermanent_loss_risk > 0.1 for p in weighted)

    def test_balancer_in_registry(self) -> None:
        """Balancer adapter should be registered in get_all_adapters."""
        adapters = get_all_adapters()
        assert len(adapters) == 7
        assert any(a.protocol == Protocol.BALANCER for a in adapters)

    def test_balancer_adapter_repr(self) -> None:
        adapter = BalancerAdapter()
        assert "BalancerAdapter" in repr(adapter)
        assert "balancer" in repr(adapter)
