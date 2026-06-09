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


class RocketPoolAdapter(BaseAdapter):
    """Adapter for Rocket Pool decentralized liquid staking protocol.

    Rocket Pool is the leading decentralized ETH staking protocol. Users deposit
    ETH and receive rETH, a liquid staking token that accrues staking rewards
    automatically. Unlike Lido, Rocket Pool uses a decentralized network of
    node operators (minipool operators) with 8/16 ETH bonds, making it more
    censorship-resistant.

    Key features:
    - rETH: rebasing-free liquid staking token (value accrual model)
    - Minipools: node operators run validators with reduced capital (8 ETH)
    - NO RPL requirement for stakers (only for node operators)
    - Deployed on Ethereum mainnet with bridged rETH on Arbitrum, Optimism, Base
    """

    def __init__(self, base_url: str = "https://api.rocketpool.net") -> None:
        super().__init__(Protocol.ROCKET_POOL, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Rocket Pool staking pools across chains.

        Returns mock data representing the rETH staking vault and
        rETH DeFi opportunities on L2s. The base rETH staking rate
        is determined by Ethereum consensus + execution layer rewards,
        typically running 0.5-1.5% higher than solo staking due to
        the pooling efficiency.

        Args:
            chain: Optional chain filter.

        Returns:
            List of pool information.
        """
        mock_pools = [
            PoolInfo(
                protocol=Protocol.ROCKET_POOL,
                chain=Chain.ETHEREUM,
                pool_id="rocket-pool-reth-eth",
                pool_name="Rocket Pool rETH Staking",
                token_pair="rETH",
                apy=0.0335,
                tvl_usd=5_800_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.ROCKET_POOL,
                chain=Chain.ETHEREUM,
                pool_id="rocket-pool-reth-eth-curve",
                pool_name="Rocket Pool rETH/ETH (Curve LP)",
                token_pair="rETH/ETH",
                apy=0.0410,
                tvl_usd=320_000_000,
                daily_volume_usd=18_000_000,
                is_stable=False,
                impermanent_loss_risk=0.03,
            ),
            PoolInfo(
                protocol=Protocol.ROCKET_POOL,
                chain=Chain.ARBITRUM,
                pool_id="rocket-pool-reth-arb",
                pool_name="Rocket Pool rETH (Arbitrum)",
                token_pair="rETH",
                apy=0.0320,
                tvl_usd=450_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.ROCKET_POOL,
                chain=Chain.OPTIMISM,
                pool_id="rocket-pool-reth-op",
                pool_name="Rocket Pool rETH (Optimism)",
                token_pair="rETH",
                apy=0.0318,
                tvl_usd=280_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
            PoolInfo(
                protocol=Protocol.ROCKET_POOL,
                chain=Chain.BASE,
                pool_id="rocket-pool-reth-base",
                pool_name="Rocket Pool rETH (Base)",
                token_pair="rETH",
                apy=0.0315,
                tvl_usd=120_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific Rocket Pool pool.

        Args:
            pool_id: Pool identifier (e.g. ``rocket-pool-reth-eth``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class FraxAdapter(BaseAdapter):
    """Adapter for Frax Finance ecosystem.

    Covers sFRAX staked stablecoin yields, Fraxlend lending markets,
    and FraxBP Curve-based liquidity pools.  All data is mock / indicative.
    """

    def __init__(self, base_url: str = "https://api.frax.finance") -> None:
        super().__init__(Protocol.FRAX, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Frax Finance yield opportunities.

        Returns pools across Ethereum and Arbitrum:
        - **sFRAX** – ERC-4626 vault backed by Finres T-bill yields.
        - **Fraxlend** – over-collateralised lending markets.
        - **FraxBP** – Curve-native FRAX/USDC stable LP.

        Args:
            chain: Optional chain filter.

        Returns:
            List of :class:`PoolInfo` objects.
        """
        mock_pools: list[PoolInfo] = [
            # ── sFRAX staking (Ethereum) ──────────────────────────────
            PoolInfo(
                protocol=Protocol.FRAX,
                chain=Chain.ETHEREUM,
                pool_id="frax-sfrax-eth",
                pool_name="Frax sFRAX Staking",
                token_pair="sFRAX",
                apy=0.0475,
                tvl_usd=820_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── sFRAX staking (Arbitrum) ──────────────────────────────
            PoolInfo(
                protocol=Protocol.FRAX,
                chain=Chain.ARBITRUM,
                pool_id="frax-sfrax-arb",
                pool_name="Frax sFRAX Staking (Arbitrum)",
                token_pair="sFRAX",
                apy=0.0460,
                tvl_usd=95_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── Fraxlend: FRAX/USDC lending (Ethereum) ───────────────
            PoolInfo(
                protocol=Protocol.FRAX,
                chain=Chain.ETHEREUM,
                pool_id="frax-fraxlend-frax-usdc",
                pool_name="Fraxlend FRAX/USDC",
                token_pair="FRAX/USDC",
                apy=0.0560,
                tvl_usd=180_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── Fraxlend: FRAX/WETH lending (Ethereum) ───────────────
            PoolInfo(
                protocol=Protocol.FRAX,
                chain=Chain.ETHEREUM,
                pool_id="frax-fraxlend-frax-weth",
                pool_name="Fraxlend FRAX/WETH",
                token_pair="FRAX/WETH",
                apy=0.0820,
                tvl_usd=62_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0001,
            ),
            # ── FraxBP Curve LP (Ethereum) ────────────────────────────
            PoolInfo(
                protocol=Protocol.FRAX,
                chain=Chain.ETHEREUM,
                pool_id="frax-fraxbp-curve",
                pool_name="FraxBP (FRAX/USDC) Curve LP",
                token_pair="FRAX/USDC",
                apy=0.0380,
                tvl_usd=420_000_000,
                daily_volume_usd=35_000_000,
                is_stable=True,
                impermanent_loss_risk=0.003,
                deposit_fee=0.0,
                withdrawal_fee=0.0004,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific Frax pool.

        Args:
            pool_id: Pool identifier (e.g. ``frax-sfrax-eth``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class MakerDAOAdapter(BaseAdapter):
    """Adapter for MakerDAO DAI Savings Rate (DSR) and sDAI vaults.

    MakerDAO's DAI Savings Rate (DSR) is one of the most battle-tested
    yield primitives in DeFi.  Users deposit DAI into the DSR module
    (or the sDAI ERC-4626 wrapper on SparkLend) and earn a governance-
    determined rate funded by stability fees on outstanding DAI debt.

    Key characteristics:
    - Single-asset (DAI) — zero impermanent loss.
    - Rate is set by MKR governance; has ranged 1 %–8 % historically.
    - sDAI is an ERC-4626 vault, composable across DeFi.
    - Also covers SparkLend (MakerDAO-affiliated lending market).

    All data is mock / indicative.
    """

    def __init__(self, base_url: str = "https://api.makerdao.com") -> None:
        super().__init__(Protocol.MAKERDAO, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch MakerDAO DSR and SparkLend pools.

        Returns pools across Ethereum and Gnosis Chain:
        - **sDAI** – ERC-4626 wrapper around the DSR module.
        - **DSR direct** – raw DSR deposit (included for completeness).
        - **SparkLend** – MakerDAO-affiliated lending markets.

        Args:
            chain: Optional chain filter.

        Returns:
            List of :class:`PoolInfo` objects.
        """
        mock_pools: list[PoolInfo] = [
            # ── sDAI vault (Ethereum) ─────────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ETHEREUM,
                pool_id="makerdao-sdai-eth",
                pool_name="MakerDAO sDAI Savings (ERC-4626)",
                token_pair="sDAI",
                apy=0.0500,
                tvl_usd=3_200_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── sDAI vault (Gnosis Chain) ─────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ETHEREUM,  # Gnosis not in Chain enum; label as Ethereum
                pool_id="makerdao-sdai-gnosis",
                pool_name="MakerDAO sDAI Savings (Gnosis)",
                token_pair="sDAI",
                apy=0.0500,
                tvl_usd=145_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── SparkLend DAI (Ethereum) ──────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ETHEREUM,
                pool_id="makerdao-spark-dai-eth",
                pool_name="SparkLend DAI (Ethereum)",
                token_pair="DAI",
                apy=0.0430,
                tvl_usd=1_800_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── SparkLend USDC (Ethereum) ─────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ETHEREUM,
                pool_id="makerdao-spark-usdc-eth",
                pool_name="SparkLend USDC (Ethereum)",
                token_pair="USDC",
                apy=0.0380,
                tvl_usd=620_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── SparkLend WETH (Ethereum) ─────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ETHEREUM,
                pool_id="makerdao-spark-weth-eth",
                pool_name="SparkLend WETH (Ethereum)",
                token_pair="WETH",
                apy=0.0195,
                tvl_usd=480_000_000,
                is_stable=False,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
            # ── SparkLend DAI (Arbitrum) ──────────────────────────────
            PoolInfo(
                protocol=Protocol.MAKERDAO,
                chain=Chain.ARBITRUM,
                pool_id="makerdao-spark-dai-arb",
                pool_name="SparkLend DAI (Arbitrum)",
                token_pair="DAI",
                apy=0.0460,
                tvl_usd=210_000_000,
                is_stable=True,
                impermanent_loss_risk=0.0,
                deposit_fee=0.0,
                withdrawal_fee=0.0,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific MakerDAO pool.

        Args:
            pool_id: Pool identifier (e.g. ``makerdao-sdai-eth``).

        Returns:
            Pool info or None if not found.
        """
        pools = await self.fetch_pools()
        return next((p for p in pools if p.pool_id == pool_id), None)


class PendleAdapter(BaseAdapter):
    """Adapter for Pendle Finance yield tokenization protocol."""

    def __init__(self, base_url: str = "https://api-v2.pendle.finance") -> None:
        super().__init__(Protocol.PENDLE, base_url)

    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch Pendle Finance pools. Currently returns mock data.

        Pendle Finance enables yield tokenization — users can trade yield
        tokens (YT), provide liquidity in PT/YT AMM pools, lock PT for
        fixed yield at maturity, or stake PENDLE for vePENDLE boosted yields.

        Args:
            chain: Optional chain filter.

        Returns:
            List of pool information.
        """
        mock_pools = [
            # ── ETH Staking PT (Ethereum) ────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ETHEREUM,
                pool_id="pendle-eth-staking-pt-eth",
                pool_name="Pendle ETH Staking PT (Fixed Yield)",
                token_pair="PT-ETH",
                apy=0.0980,
                tvl_usd=420_000_000,
                daily_volume_usd=12_000_000,
                is_stable=False,
                impermanent_loss_risk=0.02,
            ),
            # ── stETH PT (Ethereum) ──────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ETHEREUM,
                pool_id="pendle-steth-pt-eth",
                pool_name="Pendle stETH PT (Fixed Yield)",
                token_pair="PT-stETH",
                apy=0.0640,
                tvl_usd=310_000_000,
                daily_volume_usd=8_500_000,
                is_stable=False,
                impermanent_loss_risk=0.01,
            ),
            # ── USDC LP (Ethereum) ───────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ETHEREUM,
                pool_id="pendle-usdc-lp-eth",
                pool_name="Pendle USDC LP (Ethereum)",
                token_pair="PT-USDC/USDC",
                apy=0.0820,
                tvl_usd=185_000_000,
                daily_volume_usd=6_200_000,
                is_stable=True,
                impermanent_loss_risk=0.08,
            ),
            # ── rETH YT (Ethereum) ───────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ETHEREUM,
                pool_id="pendle-reth-yt-eth",
                pool_name="Pendle rETH YT (Yield Speculation)",
                token_pair="YT-rETH",
                apy=0.1450,
                tvl_usd=95_000_000,
                daily_volume_usd=3_800_000,
                is_stable=False,
                impermanent_loss_risk=0.03,
            ),
            # ── GLP PT (Arbitrum) ────────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ARBITRUM,
                pool_id="pendle-glp-pt-arb",
                pool_name="Pendle GLP PT (Fixed Yield, Arbitrum)",
                token_pair="PT-GLP",
                apy=0.1920,
                tvl_usd=120_000_000,
                daily_volume_usd=4_500_000,
                is_stable=False,
                impermanent_loss_risk=0.03,
            ),
            # ── USDC LP (Arbitrum) ───────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ARBITRUM,
                pool_id="pendle-usdc-lp-arb",
                pool_name="Pendle USDC LP (Arbitrum)",
                token_pair="PT-USDC/USDC",
                apy=0.0950,
                tvl_usd=85_000_000,
                daily_volume_usd=3_200_000,
                is_stable=True,
                impermanent_loss_risk=0.07,
            ),
            # ── wstETH LP (Arbitrum) ─────────────────────────────────
            PoolInfo(
                protocol=Protocol.PENDLE,
                chain=Chain.ARBITRUM,
                pool_id="pendle-wsteth-lp-arb",
                pool_name="Pendle wstETH LP (Arbitrum)",
                token_pair="PT-wstETH/wstETH",
                apy=0.0680,
                tvl_usd=72_000_000,
                daily_volume_usd=2_100_000,
                is_stable=False,
                impermanent_loss_risk=0.10,
            ),
        ]
        if chain:
            return [p for p in mock_pools if p.chain == chain]
        return mock_pools

    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch detail for a specific Pendle pool.

        Args:
            pool_id: Pool identifier (e.g. ``pendle-eth-staking-pt-eth``).

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
    Protocol.ROCKET_POOL: RocketPoolAdapter,
    Protocol.FRAX: FraxAdapter,
    Protocol.MAKERDAO: MakerDAOAdapter,
    Protocol.PENDLE: PendleAdapter,
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
