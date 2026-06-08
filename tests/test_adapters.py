"""Tests for protocol adapters."""

import asyncio

import pytest

from defi_yield_aggregator.adapters.protocols import (
    AaveAdapter,
    BalancerAdapter,
    CompoundAdapter,
    ConvexAdapter,
    CurveAdapter,
    FraxAdapter,
    LidoAdapter,
    RocketPoolAdapter,
    SushiSwapAdapter,
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
        assert len(adapters) == 11

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
        assert len(adapters) == 11
        assert any(a.protocol == Protocol.BALANCER for a in adapters)

    def test_balancer_adapter_repr(self) -> None:
        adapter = BalancerAdapter()
        assert "BalancerAdapter" in repr(adapter)
        assert "balancer" in repr(adapter)

    # --- Convex tests ---

    @pytest.mark.asyncio
    async def test_convex_fetch_pools(self) -> None:
        adapter = ConvexAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 5
        assert all(p.protocol == Protocol.CONVEX for p in pools)

    @pytest.mark.asyncio
    async def test_convex_fetch_pools_chain_filter(self) -> None:
        adapter = ConvexAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 5
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_convex_fetch_pool_detail(self) -> None:
        adapter = ConvexAdapter()
        detail = await adapter.fetch_pool_detail("convex-steth")
        assert detail is not None
        assert detail.pool_id == "convex-steth"
        assert detail.tvl_usd == 1_600_000_000
        assert detail.apy == pytest.approx(0.065)

    @pytest.mark.asyncio
    async def test_convex_fetch_pool_detail_not_found(self) -> None:
        adapter = ConvexAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_convex_stable_pools_low_il(self) -> None:
        """Stable pools should have low IL risk."""
        adapter = ConvexAdapter()
        stable_pools = [p for p in await adapter.fetch_pools() if p.is_stable]
        assert len(stable_pools) >= 2
        assert all(p.impermanent_loss_risk < 0.01 for p in stable_pools)

    @pytest.mark.asyncio
    async def test_convex_non_stable_pools(self) -> None:
        """Non-stable pools should be retrievable."""
        adapter = ConvexAdapter()
        non_stable = [p for p in await adapter.fetch_pools() if not p.is_stable]
        assert len(non_stable) >= 2
        assert all(p.protocol == Protocol.CONVEX for p in non_stable)

    def test_convex_in_registry(self) -> None:
        """Convex adapter should be registered in get_all_adapters."""
        adapters = get_all_adapters()
        assert len(adapters) == 11
        assert any(a.protocol == Protocol.CONVEX for a in adapters)

    def test_convex_adapter_repr(self) -> None:
        adapter = ConvexAdapter()
        assert "ConvexAdapter" in repr(adapter)
        assert "convex" in repr(adapter)

    # --- SushiSwap tests ---

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pools(self) -> None:
        adapter = SushiSwapAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 8
        assert all(p.protocol == Protocol.SUSHISWAP for p in pools)

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pools_chain_filter_eth(self) -> None:
        adapter = SushiSwapAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 3
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pools_chain_filter_arb(self) -> None:
        adapter = SushiSwapAdapter()
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert len(arb_pools) == 2
        assert all(p.chain == Chain.ARBITRUM for p in arb_pools)

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pools_chain_filter_base(self) -> None:
        adapter = SushiSwapAdapter()
        base_pools = await adapter.fetch_pools(chain=Chain.BASE)
        assert len(base_pools) == 1
        assert base_pools[0].pool_id == "sushi-eth-usdc-base"

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pools_chain_filter_optimism(self) -> None:
        adapter = SushiSwapAdapter()
        op_pools = await adapter.fetch_pools(chain=Chain.OPTIMISM)
        assert len(op_pools) == 1
        assert op_pools[0].pool_id == "sushi-eth-usdt-op"

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pool_detail(self) -> None:
        adapter = SushiSwapAdapter()
        detail = await adapter.fetch_pool_detail("sushi-eth-usdc-eth")
        assert detail is not None
        assert detail.pool_id == "sushi-eth-usdc-eth"
        assert detail.tvl_usd == 180_000_000
        assert detail.apy == pytest.approx(0.054)

    @pytest.mark.asyncio
    async def test_sushiswap_fetch_pool_detail_not_found(self) -> None:
        adapter = SushiSwapAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_sushiswap_stable_pools_low_il(self) -> None:
        """Stable pools (Trident, Kashi) should have low IL risk."""
        adapter = SushiSwapAdapter()
        stable_pools = [p for p in await adapter.fetch_pools() if p.is_stable]
        assert len(stable_pools) == 2
        assert all(p.impermanent_loss_risk < 0.01 for p in stable_pools)

    @pytest.mark.asyncio
    async def test_sushiswap_amm_pools_have_il_risk(self) -> None:
        """AMM pools (non-stable) should have meaningful IL risk."""
        adapter = SushiSwapAdapter()
        amm_pools = [p for p in await adapter.fetch_pools() if not p.is_stable]
        assert len(amm_pools) == 6
        assert all(p.impermanent_loss_risk > 0.1 for p in amm_pools)

    @pytest.mark.asyncio
    async def test_sushiswap_multi_chain_coverage(self) -> None:
        """SushiSwap should cover 5 chains."""
        adapter = SushiSwapAdapter()
        pools = await adapter.fetch_pools()
        chains = {p.chain for p in pools}
        assert chains == {Chain.ETHEREUM, Chain.ARBITRUM, Chain.POLYGON, Chain.BASE, Chain.OPTIMISM}

    def test_sushiswap_in_registry(self) -> None:
        """SushiSwap adapter should be registered in get_all_adapters."""
        adapters = get_all_adapters()
        assert len(adapters) == 11
        assert any(a.protocol == Protocol.SUSHISWAP for a in adapters)

    def test_sushiswap_adapter_repr(self) -> None:
        adapter = SushiSwapAdapter()
        assert "SushiSwapAdapter" in repr(adapter)
        assert "sushiswap" in repr(adapter)

    @pytest.mark.asyncio
    async def test_sushiswap_kashi_lending_zero_il(self) -> None:
        """Kashi lending pools should have zero IL risk (single-asset)."""
        adapter = SushiSwapAdapter()
        detail = await adapter.fetch_pool_detail("sushi-kashi-usdc-lend")
        assert detail is not None
        assert detail.impermanent_loss_risk == 0.0
        assert detail.is_stable is True

    # --- Rocket Pool tests ---

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pools(self) -> None:
        adapter = RocketPoolAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 5
        assert all(p.protocol == Protocol.ROCKET_POOL for p in pools)

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pools_chain_filter_eth(self) -> None:
        adapter = RocketPoolAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 2
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pools_chain_filter_arb(self) -> None:
        adapter = RocketPoolAdapter()
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert len(arb_pools) == 1
        assert arb_pools[0].pool_id == "rocket-pool-reth-arb"

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pools_chain_filter_op(self) -> None:
        adapter = RocketPoolAdapter()
        op_pools = await adapter.fetch_pools(chain=Chain.OPTIMISM)
        assert len(op_pools) == 1
        assert op_pools[0].pool_id == "rocket-pool-reth-op"

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pools_chain_filter_base(self) -> None:
        adapter = RocketPoolAdapter()
        base_pools = await adapter.fetch_pools(chain=Chain.BASE)
        assert len(base_pools) == 1
        assert base_pools[0].pool_id == "rocket-pool-reth-base"

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pool_detail(self) -> None:
        adapter = RocketPoolAdapter()
        detail = await adapter.fetch_pool_detail("rocket-pool-reth-eth")
        assert detail is not None
        assert detail.pool_id == "rocket-pool-reth-eth"
        assert detail.tvl_usd == 5_800_000_000
        assert detail.apy == pytest.approx(0.0335)

    @pytest.mark.asyncio
    async def test_rocket_pool_fetch_pool_detail_not_found(self) -> None:
        adapter = RocketPoolAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_rocket_pool_reth_staking_zero_il(self) -> None:
        """Pure rETH staking (single-asset) should have zero IL risk."""
        adapter = RocketPoolAdapter()
        detail = await adapter.fetch_pool_detail("rocket-pool-reth-eth")
        assert detail is not None
        assert detail.impermanent_loss_risk == 0.0

    @pytest.mark.asyncio
    async def test_rocket_pool_curve_lp_has_il_risk(self) -> None:
        """Curve LP position (rETH/ETH) should have non-zero IL risk."""
        adapter = RocketPoolAdapter()
        detail = await adapter.fetch_pool_detail("rocket-pool-reth-eth-curve")
        assert detail is not None
        assert detail.impermanent_loss_risk > 0.0

    @pytest.mark.asyncio
    async def test_rocket_pool_multi_chain_coverage(self) -> None:
        """Rocket Pool should cover Ethereum + L2s."""
        adapter = RocketPoolAdapter()
        pools = await adapter.fetch_pools()
        chains = {p.chain for p in pools}
        assert chains == {Chain.ETHEREUM, Chain.ARBITRUM, Chain.OPTIMISM, Chain.BASE}

    def test_rocket_pool_in_registry(self) -> None:
        """Rocket Pool adapter should be registered in get_all_adapters."""
        adapters = get_all_adapters()
        assert len(adapters) == 11
        assert any(a.protocol == Protocol.ROCKET_POOL for a in adapters)

    def test_rocket_pool_adapter_repr(self) -> None:
        adapter = RocketPoolAdapter()
        assert "RocketPoolAdapter" in repr(adapter)
        assert "rocket_pool" in repr(adapter)

    # --- Frax tests ---

    @pytest.mark.asyncio
    async def test_frax_fetch_pools(self) -> None:
        adapter = FraxAdapter()
        pools = await adapter.fetch_pools()
        assert len(pools) == 5
        assert all(p.protocol == Protocol.FRAX for p in pools)

    @pytest.mark.asyncio
    async def test_frax_fetch_pools_chain_filter_eth(self) -> None:
        adapter = FraxAdapter()
        eth_pools = await adapter.fetch_pools(chain=Chain.ETHEREUM)
        assert len(eth_pools) == 4
        assert all(p.chain == Chain.ETHEREUM for p in eth_pools)

    @pytest.mark.asyncio
    async def test_frax_fetch_pools_chain_filter_arb(self) -> None:
        adapter = FraxAdapter()
        arb_pools = await adapter.fetch_pools(chain=Chain.ARBITRUM)
        assert len(arb_pools) == 1
        assert arb_pools[0].pool_id == "frax-sfrax-arb"

    @pytest.mark.asyncio
    async def test_frax_fetch_pool_detail(self) -> None:
        adapter = FraxAdapter()
        detail = await adapter.fetch_pool_detail("frax-sfrax-eth")
        assert detail is not None
        assert detail.pool_id == "frax-sfrax-eth"
        assert detail.tvl_usd == 820_000_000
        assert detail.apy == pytest.approx(0.0475)

    @pytest.mark.asyncio
    async def test_frax_fetch_pool_detail_not_found(self) -> None:
        adapter = FraxAdapter()
        detail = await adapter.fetch_pool_detail("nonexistent-pool")
        assert detail is None

    @pytest.mark.asyncio
    async def test_frax_sfrax_staking_zero_il(self) -> None:
        """sFRAX staking (single-asset) should have zero IL risk."""
        adapter = FraxAdapter()
        detail = await adapter.fetch_pool_detail("frax-sfrax-eth")
        assert detail is not None
        assert detail.impermanent_loss_risk == 0.0
        assert detail.is_stable is True

    @pytest.mark.asyncio
    async def test_frax_stable_pools_low_il(self) -> None:
        """Stable pools should have low IL risk."""
        adapter = FraxAdapter()
        stable_pools = [p for p in await adapter.fetch_pools() if p.is_stable]
        assert len(stable_pools) == 4
        assert all(p.impermanent_loss_risk < 0.01 for p in stable_pools)

    @pytest.mark.asyncio
    async def test_frax_non_stable_pool(self) -> None:
        """Fraxlend WETH pool is non-stable with zero IL risk (lending)."""
        adapter = FraxAdapter()
        detail = await adapter.fetch_pool_detail("frax-fraxlend-frax-weth")
        assert detail is not None
        assert detail.is_stable is False
        assert detail.impermanent_loss_risk == 0.0
        assert detail.apy == pytest.approx(0.082)

    @pytest.mark.asyncio
    async def test_frax_fraxbp_curve_lp_has_il_risk(self) -> None:
        """FraxBP Curve LP should have small but non-zero IL risk."""
        adapter = FraxAdapter()
        detail = await adapter.fetch_pool_detail("frax-fraxbp-curve")
        assert detail is not None
        assert detail.impermanent_loss_risk > 0.0
        assert detail.daily_volume_usd == 35_000_000

    @pytest.mark.asyncio
    async def test_frax_multi_chain_coverage(self) -> None:
        """Frax should cover Ethereum and Arbitrum."""
        adapter = FraxAdapter()
        pools = await adapter.fetch_pools()
        chains = {p.chain for p in pools}
        assert chains == {Chain.ETHEREUM, Chain.ARBITRUM}

    def test_frax_in_registry(self) -> None:
        """Frax adapter should be registered in get_all_adapters."""
        adapters = get_all_adapters()
        assert len(adapters) == 11
        assert any(a.protocol == Protocol.FRAX for a in adapters)

    def test_frax_adapter_repr(self) -> None:
        adapter = FraxAdapter()
        assert "FraxAdapter" in repr(adapter)
        assert "frax" in repr(adapter)
