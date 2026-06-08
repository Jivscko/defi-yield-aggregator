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


class LidoAdapter(BaseAdapter):
    """Adapter for Lido liquid staking protocol (stETH, stMATIC)."""

    def __init__(self, base_url: str = "https://stake.lido.fi/api") -> None:
        super().__init__(Protocol.LIDO, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Lido liquid staking pools. Currently returns mock data."""
        mock_pools = [
            PoolInfo(
                protocol=Protocol.LIDO,
                chain=Chain.ETHEREUM,
                pool_id="lido-steth-eth",
                pool_name="Lido stETH",
                token_pair="stETH",
                apy=0.032,
                tvl_usd=14_000_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.LIDO,
                chain=Chain.POLYGON,
                pool_id="lido-stmatic-poly",
                pool_name="Lido stMATIC (Polygon)",
                token_pair="stMATIC",
                apy=0.045,
                tvl_usd=800_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.LIDO,
                chain=Chain.ARBITRUM,
                pool_id="lido-steth-arb",
                pool_name="Lido stETH (Arbitrum via bridged)",
                token_pair="stETH",
                apy=0.030,
                tvl_usd=400_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class BalancerAdapter(BaseAdapter):
    """Adapter for Balancer V2 weighted and boosted pools."""

    def __init__(self, base_url: str = "https://api.balancer.fi") -> None:
        super().__init__(Protocol.BALANCER, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Balancer V2 pools. Currently returns mock data.

        Balancer supports weighted pools (e.g. 80/20 token allocations),
        stable pools (similar to Curve for pegged assets), and
        boosted pools (composable with Aave/Yearn for extra yield).

        Args:
            chain: Optional chain filter.

        Returns:
            List of pool information.
        """
        mock_pools = [
            PoolInfo(
                protocol=Protocol.BALANCER,
                chain=Chain.ETHEREUM,
                pool_id="balancer-wsteth-weth-5050",
                pool_name="Balancer wstETH/WETH 50/50",
                token_pair="wstETH/WETH",
                apy=0.0340,
                tvl_usd=1_800_000_000,
                daily_volume_usd=42_000_000,
                is_stable=False,
                impermanent_loss_risk=0.02,
            ),
            PoolInfo(
                protocol=Protocol.BALANCER,
                chain=Chain.ETHEREUM,
                pool_id="balancer-stable-usdc-dai-usdt",
                pool_name="Balancer Stable Pool (USDC/DAI/USDT)",
                token_pair="USDC/DAI/USDT",
                apy=0.0280,
                tvl_usd=620_000_000,
                daily_volume_usd=28_000_000,
                is_stable=True,
                impermanent_loss_risk=0.003,
            ),
            PoolInfo(
                protocol=Protocol.BALANCER,
                chain=Chain.ETHEREUM,
                pool_id="balancer-80bal-20weth",
                pool_name="Balancer 80BAL/20WETH",
                token_pair="BAL/WETH",
                apy=0.0890,
                tvl_usd=185_000_000,
                daily_volume_usd=5_500_000,
                is_stable=False,
                impermanent_loss_risk=0.45,
            ),
            PoolInfo(
                protocol=Protocol.BALANCER,
                chain=Chain.ARBITRUM,
                pool_id="balancer-weth-usdc-arb",
                pool_name="Balancer WETH/USDC 50/50 (Arbitrum)",
                token_pair="WETH/USDC",
                apy=0.0720,
                tvl_usd=340_000_000,
                daily_volume_usd=18_000_000,
                is_stable=False,
                impermanent_loss_risk=0.28,
            ),
            PoolInfo(
                protocol=Protocol.BALANCER,
                chain=Chain.POLYGON,
                pool_id="balancer-boosted-aave-poly",
                pool_name="Balancer Boosted Aave USDC/USDT (Polygon)",
                token_pair="USDC/USDT",
                apy=0.0480,
                tvl_usd=275_000_000,
                daily_volume_usd=8_000_000,
                is_stable=True,
                impermanent_loss_risk=0.005,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific Balancer pool.

        Args:
            pool_id: Pool identifier (e.g. ``balancer-wsteth-weth-5050``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class ConvexAdapter(BaseAdapter):
    """Adapter for Convex Finance boosted Curve pools."""

    def __init__(self, base_url: str = "https://www.convexfinance.com/api") -> None:
        super().__init__(Protocol.CONVEX, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Convex Finance pools. Currently returns mock data.

        Convex Finance boosts Curve LP yields by allowing users to stake CRV
        tokens without locking them directly on Curve.

        Args:
            chain: Optional chain filter.

        Returns:
            List of pool information.
        """
        mock_pools = [
            PoolInfo(
                protocol=Protocol.CONVEX,
                chain=Chain.ETHEREUM,
                pool_id="convex-cvxcrv-staking",
                pool_name="Convex cvxCRV Staking",
                token_pair="cvxCRV",
                apy=0.048,
                tvl_usd=1_200_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.CONVEX,
                chain=Chain.ETHEREUM,
                pool_id="convex-3pool",
                pool_name="Convex Curve 3Pool Staking",
                token_pair="DAI/USDC/USDT",
                apy=0.052,
                tvl_usd=850_000_000,
                daily_volume_usd=45_000_000,
                is_stable=True,
                impermanent_loss_risk=0.005,
            ),
            PoolInfo(
                protocol=Protocol.CONVEX,
                chain=Chain.ETHEREUM,
                pool_id="convex-steth",
                pool_name="Convex Curve stETH/ETH",
                token_pair="stETH/ETH",
                apy=0.065,
                tvl_usd=1_600_000_000,
                daily_volume_usd=30_000_000,
                is_stable=False,
                impermanent_loss_risk=0.04,
            ),
            PoolInfo(
                protocol=Protocol.CONVEX,
                chain=Chain.ETHEREUM,
                pool_id="convex-frax-usdc",
                pool_name="Convex Curve FRAX/USDC",
                token_pair="FRAX/USDC",
                apy=0.058,
                tvl_usd=420_000_000,
                is_stable=True,
                impermanent_loss_risk=0.003,
            ),
            PoolInfo(
                protocol=Protocol.CONVEX,
                chain=Chain.ETHEREUM,
                pool_id="convex-cvx-staking",
                pool_name="Convex CVX Staking",
                token_pair="CVX",
                apy=0.072,
                tvl_usd=380_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific Convex pool.

        Args:
            pool_id: Pool identifier (e.g. ``convex-steth``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class SushiSwapAdapter(BaseAdapter):
    """Adapter for SushiSwap multi-chain DEX with AMM, Trident, and Kashi pools."""

    def __init__(self, base_url: str = "https://api.sushi.com") -> None:
        super().__init__(Protocol.SUSHISWAP, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch SushiSwap pools across multiple chains.

        SushiSwap offers constant product AMM pools (x*y=k), Trident concentrated
        liquidity stable pools, and Kashi isolated lending markets. The Onsen menu
        provides additional SUSHI rewards for select pairs.

        Args:
            chain: Optional chain filter.

        Returns:
            List of pool information.
        """
        mock_pools = [
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.ETHEREUM,
                pool_id="sushi-eth-usdc-eth",
                pool_name="SushiSwap ETH/USDC (Ethereum)",
                token_pair="ETH/USDC",
                apy=0.0540,
                tvl_usd=180_000_000,
                daily_volume_usd=35_000_000,
                is_stable=False,
                impermanent_loss_risk=0.30,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.ETHEREUM,
                pool_id="sushi-wbtc-eth-eth",
                pool_name="SushiSwap WBTC/ETH (Ethereum)",
                token_pair="WBTC/ETH",
                apy=0.0420,
                tvl_usd=95_000_000,
                daily_volume_usd=12_000_000,
                is_stable=False,
                impermanent_loss_risk=0.15,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.ETHEREUM,
                pool_id="sushi-kashi-usdc-lend",
                pool_name="SushiSwap Kashi USDC Lending",
                token_pair="USDC",
                apy=0.0380,
                tvl_usd=28_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.ARBITRUM,
                pool_id="sushi-usdc-usdt-arb",
                pool_name="SushiSwap Trident USDC/USDT (Arbitrum)",
                token_pair="USDC/USDT",
                apy=0.0290,
                tvl_usd=62_000_000,
                daily_volume_usd=18_000_000,
                is_stable=True,
                impermanent_loss_risk=0.002,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.ARBITRUM,
                pool_id="sushi-eth-usdc-arb",
                pool_name="SushiSwap ETH/USDC (Arbitrum)",
                token_pair="ETH/USDC",
                apy=0.0680,
                tvl_usd=45_000_000,
                daily_volume_usd=8_500_000,
                is_stable=False,
                impermanent_loss_risk=0.32,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.POLYGON,
                pool_id="sushi-matic-usdc-poly",
                pool_name="SushiSwap MATIC/USDC (Polygon)",
                token_pair="MATIC/USDC",
                apy=0.0750,
                tvl_usd=22_000_000,
                daily_volume_usd=4_200_000,
                is_stable=False,
                impermanent_loss_risk=0.38,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.BASE,
                pool_id="sushi-eth-usdc-base",
                pool_name="SushiSwap ETH/USDC (Base)",
                token_pair="ETH/USDC",
                apy=0.0890,
                tvl_usd=15_000_000,
                daily_volume_usd=3_800_000,
                is_stable=False,
                impermanent_loss_risk=0.35,
            ),
            PoolInfo(
                protocol=Protocol.SUSHISWAP,
                chain=Chain.OPTIMISM,
                pool_id="sushi-eth-usdt-op",
                pool_name="SushiSwap ETH/USDT (Optimism)",
                token_pair="ETH/USDT",
                apy=0.0620,
                tvl_usd=18_000_000,
                daily_volume_usd=2_100_000,
                is_stable=False,
                impermanent_loss_risk=0.28,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific SushiSwap pool.

        Args:
            pool_id: Pool identifier (e.g. ``sushi-eth-usdc-eth``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


# Registry of all adapters
ADAPTERS: dict[Protocol, type[BaseAdapter]] = {
    Protocol.AAVE: AaveAdapter,
    Protocol.COMPOUND: CompoundAdapter,
    Protocol.UNISWAP: UniswapAdapter,
    Protocol.CURVE: CurveAdapter,
    Protocol.YEARN: YearnAdapter,
    Protocol.LIDO: LidoAdapter,
    Protocol.BALANCER: BalancerAdapter,
    Protocol.CONVEX: ConvexAdapter,
    Protocol.SUSHISWAP: SushiSwapAdapter,
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
