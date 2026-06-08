"""Protocol adapters - fetch pool data from DeFi protocols."""

from __future__ import annotations

from datetime import datetime

from defi_yield_aggregator.adapters.base import BaseAdapter
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


class AaveAdapter(BaseAdapter):
    """Adapter for Aave V3 lending protocol."""

    def __init__(self, base_url: str = "https://aave-api-v2.aave.com") -> None:
        super().__init__(Protocol.AAVE, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Aave lending markets. Currently returns mock data."""
        mock_pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-v3-usdc-eth",
                pool_name="Aave V3 USDC",
                token_pair="USDC",
                apy=0.0385,
                tvl_usd=6_200_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-v3-weth-eth",
                pool_name="Aave V3 WETH",
                token_pair="WETH",
                apy=0.0215,
                tvl_usd=4_800_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ARBITRUM,
                pool_id="aave-v3-usdc-arb",
                pool_name="Aave V3 USDC (Arbitrum)",
                token_pair="USDC",
                apy=0.0420,
                tvl_usd=1_100_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.POLYGON,
                pool_id="aave-v3-usdt-poly",
                pool_name="Aave V3 USDT (Polygon)",
                token_pair="USDT",
                apy=0.0510,
                tvl_usd=850_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class CompoundAdapter(BaseAdapter):
    """Adapter for Compound V3 lending protocol."""

    def __init__(self, base_url: str = "https://api.compound.finance") -> None:
        super().__init__(Protocol.COMPOUND, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        mock_pools = [
            PoolInfo(
                protocol=Protocol.COMPOUND,
                chain=Chain.ETHEREUM,
                pool_id="compound-v3-usdc-eth",
                pool_name="Compound V3 USDC",
                token_pair="USDC",
                apy=0.0340,
                tvl_usd=3_200_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.COMPOUND,
                chain=Chain.ETHEREUM,
                pool_id="compound-v3-weth",
                pool_name="Compound V3 WETH",
                token_pair="WETH",
                apy=0.0180,
                tvl_usd=2_100_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.COMPOUND,
                chain=Chain.POLYGON,
                pool_id="compound-v3-usdc-poly",
                pool_name="Compound V3 USDC (Polygon)",
                token_pair="USDC",
                apy=0.0390,
                tvl_usd=520_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class UniswapAdapter(BaseAdapter):
    """Adapter for Uniswap V3 concentrated liquidity."""

    def __init__(self, base_url: str = "https://api.thegraph.com/subgraphs/name/uniswap") -> None:
        super().__init__(Protocol.UNISWAP, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        mock_pools = [
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ETHEREUM,
                pool_id="uni-v3-usdc-eth-03",
                pool_name="Uniswap V3 USDC/ETH 0.3%",
                token_pair="USDC/ETH",
                apy=0.0680,
                tvl_usd=2_400_000_000,
                daily_volume_usd=180_000_000,
                is_stable=False,
                impermanent_loss_risk=0.35,
            ),
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ETHEREUM,
                pool_id="uni-v3-usdc-usdt-001",
                pool_name="Uniswap V3 USDC/USDT 0.01%",
                token_pair="USDC/USDT",
                apy=0.0250,
                tvl_usd=1_800_000_000,
                daily_volume_usd=350_000_000,
                is_stable=True,
                impermanent_loss_risk=0.01,
            ),
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ARBITRUM,
                pool_id="uni-v3-eth-usdc-arb-03",
                pool_name="Uniswap V3 ETH/USDC (Arbitrum) 0.3%",
                token_pair="ETH/USDC",
                apy=0.0820,
                tvl_usd=680_000_000,
                daily_volume_usd=95_000_000,
                is_stable=False,
                impermanent_loss_risk=0.40,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class CurveAdapter(BaseAdapter):
    """Adapter for Curve Finance stableswap pools."""

    def __init__(self, base_url: str = "https://api.curve.fi") -> None:
        super().__init__(Protocol.CURVE, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        mock_pools = [
            PoolInfo(
                protocol=Protocol.CURVE,
                chain=Chain.ETHEREUM,
                pool_id="curve-3pool",
                pool_name="Curve 3Pool (DAI/USDC/USDT)",
                token_pair="DAI/USDC/USDT",
                apy=0.0320,
                tvl_usd=1_500_000_000,
                daily_volume_usd=120_000_000,
                is_stable=True,
                impermanent_loss_risk=0.005,
            ),
            PoolInfo(
                protocol=Protocol.CURVE,
                chain=Chain.ETHEREUM,
                pool_id="curve-steth",
                pool_name="Curve stETH/ETH",
                token_pair="stETH/ETH",
                apy=0.0450,
                tvl_usd=2_800_000_000,
                daily_volume_usd=45_000_000,
                is_stable=False,
                impermanent_loss_risk=0.05,
            ),
            PoolInfo(
                protocol=Protocol.CURVE,
                chain=Chain.ARBITRUM,
                pool_id="curve-2pool-arb",
                pool_name="Curve 2Pool (Arbitrum)",
                token_pair="USDC/USDT",
                apy=0.0380,
                tvl_usd=340_000_000,
                is_stable=True,
                impermanent_loss_risk=0.002,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class YearnAdapter(BaseAdapter):
    """Adapter for Yearn Finance vaults."""

    def __init__(self, base_url: str = "https://api.yearn.fi") -> None:
        super().__init__(Protocol.YEARN, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        mock_pools = [
            PoolInfo(
                protocol=Protocol.YEARN,
                chain=Chain.ETHEREUM,
                pool_id="yearn-yvusdc",
                pool_name="Yearn V3 yvUSDC",
                token_pair="USDC",
                apy=0.0520,
                tvl_usd=780_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.YEARN,
                chain=Chain.ETHEREUM,
                pool_id="yearn-yvweth",
                pool_name="Yearn V3 yvWETH",
                token_pair="WETH",
                apy=0.0310,
                tvl_usd=450_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.YEARN,
                chain=Chain.ARBITRUM,
                pool_id="yearn-yvusdc-arb",
                pool_name="Yearn V3 yvUSDC (Arbitrum)",
                token_pair="USDC",
                apy=0.0580,
                tvl_usd=210_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


# Registry of all adapters
ADAPTERS: dict[Protocol, type[BaseAdapter]] = {
    Protocol.AAVE: AaveAdapter,
    Protocol.COMPOUND: CompoundAdapter,
    Protocol.UNISWAP: UniswapAdapter,
    Protocol.CURVE: CurveAdapter,
    Protocol.YEARN: YearnAdapter,
}


def get_all_adapters(base_urls: dict[str, str] | None = None) -> list[BaseAdapter]:
    """Instantiate all protocol adapters.

    Args:
        base_urls: Optional mapping of protocol name to base URL override.

    Returns:
        List of initialized adapters.
    """
    urls = base_urls or {}
    return [
        adapter_cls(base_url=urls.get(proto.value, ""))
        for proto, adapter_cls in ADAPTERS.items()
    ]
