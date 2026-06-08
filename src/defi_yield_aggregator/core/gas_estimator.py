"""Gas cost estimator for DeFi operations.

Estimates gas costs for deposit, withdraw, and swap operations across
different chains and protocols, then calculates net APY after accounting
for transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol


class Operation(str, Enum):
    """Supported DeFi operations."""

    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"
    SWAP = "swap"


# Gas units by (chain, operation) — realistic estimates from mainnet/L2 data
_BASE_GAS_UNITS: dict[tuple[Chain, Operation], int] = {
    # Ethereum L1 — standard EVM gas
    (Chain.ETHEREUM, Operation.DEPOSIT): 150_000,
    (Chain.ETHEREUM, Operation.WITHDRAW): 120_000,
    (Chain.ETHEREUM, Operation.SWAP): 180_000,
    # Arbitrum — L2 compute units (higher count, much lower price)
    (Chain.ARBITRUM, Operation.DEPOSIT): 800_000,
    (Chain.ARBITRUM, Operation.WITHDRAW): 600_000,
    (Chain.ARBITRUM, Operation.SWAP): 1_000_000,
    # Optimism — similar L2 profile
    (Chain.OPTIMISM, Operation.DEPOSIT): 700_000,
    (Chain.OPTIMISM, Operation.WITHDRAW): 550_000,
    (Chain.OPTIMISM, Operation.SWAP): 900_000,
    # Polygon — PoS chain, moderate gas
    (Chain.POLYGON, Operation.DEPOSIT): 200_000,
    (Chain.POLYGON, Operation.WITHDRAW): 160_000,
    (Chain.POLYGON, Operation.SWAP): 250_000,
    # Base — OP-stack L2
    (Chain.BASE, Operation.DEPOSIT): 700_000,
    (Chain.BASE, Operation.WITHDRAW): 550_000,
    (Chain.BASE, Operation.SWAP): 900_000,
    # Avalanche — C-chain
    (Chain.AVALANCHE, Operation.DEPOSIT): 180_000,
    (Chain.AVALANCHE, Operation.WITHDRAW): 140_000,
    (Chain.AVALANCHE, Operation.SWAP): 220_000,
}

# Protocol complexity multipliers (1.0 = baseline)
_PROTOCOL_MULTIPLIERS: dict[Protocol, float] = {
    Protocol.AAVE: 1.0,
    Protocol.COMPOUND: 1.05,
    Protocol.UNISWAP: 0.9,
    Protocol.CURVE: 1.3,  # Curve pools are more complex
    Protocol.YEARN: 1.2,
    Protocol.LIDO: 0.85,  # Liquid staking is simpler
    Protocol.BALANCER: 1.15,
}

# Mock gas prices in gwei by chain
_MOCK_GAS_PRICES_GWEI: dict[Chain, float] = {
    Chain.ETHEREUM: 30.0,
    Chain.ARBITRUM: 0.1,
    Chain.OPTIMISM: 0.08,
    Chain.POLYGON: 30.0,  # gwei on Polygon (MATIC-denominated)
    Chain.BASE: 0.05,
    Chain.AVALANCHE: 25.0,
}

# Native token prices in USD (for gas cost conversion)
_MOCK_NATIVE_PRICES_USD: dict[Chain, float] = {
    Chain.ETHEREUM: 3500.0,
    Chain.ARBITRUM: 3500.0,  # Uses ETH
    Chain.OPTIMISM: 3500.0,  # Uses ETH
    Chain.POLYGON: 0.75,
    Chain.BASE: 3500.0,  # Uses ETH
    Chain.AVALANCHE: 35.0,
}


@dataclass
class GasEstimate:
    """Gas cost estimate for a single DeFi operation."""

    operation: str
    chain: Chain
    protocol: Protocol
    gas_units: int
    gas_price_gwei: float
    native_token_price_usd: float
    total_cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def gas_cost_eth(self) -> float:
        """Gas cost in native token units."""
        return (self.gas_units * self.gas_price_gwei) / 1e9


class GasEstimator:
    """Estimates gas costs for DeFi operations across chains and protocols.

    Uses mock data with realistic gas unit counts and price assumptions.
    In production these would be replaced with on-chain estimates.

    Example::

        estimator = GasEstimator()
        est = estimator.estimate_gas(Operation.DEPOSIT, Chain.ETHEREUM, Protocol.AAVE)
        print(f"Deposit costs ~${est.total_cost_usd:.2f}")

        net = estimator.estimate_net_apy(pool, investment_usd=10_000, holding_period_days=30)
        print(f"Net APY: {net['net_apy']:.2%}")
    """

    def __init__(
        self,
        gas_prices_gwei: dict[Chain, float] | None = None,
        native_prices_usd: dict[Chain, float] | None = None,
        protocol_multipliers: dict[Protocol, float] | None = None,
    ) -> None:
        """Initialize the gas estimator.

        Args:
            gas_prices_gwei: Override default gas prices per chain (gwei).
            native_prices_usd: Override native token USD prices per chain.
            protocol_multipliers: Override protocol gas complexity multipliers.
        """
        self._gas_prices = gas_prices_gwei or _MOCK_GAS_PRICES_GWEI
        self._native_prices = native_prices_usd or _MOCK_NATIVE_PRICES_USD
        self._protocol_multipliers = protocol_multipliers or _PROTOCOL_MULTIPLIERS

    def estimate_gas(
        self,
        operation: Operation | str,
        chain: Chain,
        protocol: Protocol,
    ) -> GasEstimate:
        """Estimate gas cost for a DeFi operation.

        Args:
            operation: The operation type (deposit, withdraw, swap).
            chain: The target blockchain.
            protocol: The DeFi protocol.

        Returns:
            GasEstimate with cost breakdown.

        Raises:
            ValueError: If chain/operation combination is not supported.
        """
        if isinstance(operation, str):
            operation = Operation(operation)

        key = (chain, operation)
        if key not in _BASE_GAS_UNITS:
            raise ValueError(
                f"Unsupported chain/operation combination: {chain.value}/{operation.value}"
            )

        base_units = _BASE_GAS_UNITS[key]
        multiplier = self._protocol_multipliers.get(protocol, 1.0)
        gas_units = int(base_units * multiplier)

        gas_price = self._gas_prices.get(chain, 30.0)
        native_price = self._native_prices.get(chain, 3500.0)

        # Cost in USD = gas_units * gas_price_gwei * 1e-9 (to ETH) * native_price_usd
        total_cost_usd = (gas_units * gas_price / 1e9) * native_price

        return GasEstimate(
            operation=operation.value,
            chain=chain,
            protocol=protocol,
            gas_units=gas_units,
            gas_price_gwei=gas_price,
            native_token_price_usd=native_price,
            total_cost_usd=total_cost_usd,
        )

    def estimate_net_apy(
        self,
        pool: PoolInfo,
        investment_usd: float,
        holding_period_days: int = 365,
    ) -> dict[str, Any]:
        """Calculate net APY after gas costs for a pool.

        Args:
            pool: The pool to evaluate.
            investment_usd: Investment amount in USD.
            holding_period_days: How long the position will be held.

        Returns:
            Dict with keys: gross_apy, gas_cost_usd, net_apy,
            breakeven_days, profitable, gas_cost_pct.
        """
        if investment_usd <= 0:
            return {
                "pool_id": pool.pool_id,
                "gross_apy": pool.apy,
                "gas_cost_usd": 0.0,
                "net_apy": 0.0,
                "breakeven_days": float("inf"),
                "profitable": False,
                "gas_cost_pct": 0.0,
            }

        # Total gas = deposit + withdraw
        deposit_gas = self.estimate_gas(Operation.DEPOSIT, pool.chain, pool.protocol)
        withdraw_gas = self.estimate_gas(Operation.WITHDRAW, pool.chain, pool.protocol)
        total_gas_usd = deposit_gas.total_cost_usd + withdraw_gas.total_cost_usd

        # Gross earnings for the holding period
        # APY is annualized, so pro-rate by holding period
        annual_earnings = investment_usd * pool.apy
        period_earnings = annual_earnings * (holding_period_days / 365.0)

        # Net earnings after gas
        net_earnings = period_earnings - total_gas_usd
        net_apy = net_earnings / investment_usd * (365.0 / holding_period_days) if holding_period_days > 0 else 0.0

        # Breakeven: days until earnings cover gas
        daily_earnings = annual_earnings / 365.0
        breakeven_days = total_gas_usd / daily_earnings if daily_earnings > 0 else float("inf")

        return {
            "pool_id": pool.pool_id,
            "gross_apy": pool.apy,
            "gas_cost_usd": total_gas_usd,
            "net_apy": max(net_apy, 0.0),
            "breakeven_days": breakeven_days,
            "profitable": net_earnings > 0,
            "gas_cost_pct": (total_gas_usd / investment_usd * 100) if investment_usd > 0 else 0.0,
        }

    def batch_estimate(
        self,
        pools: list[PoolInfo],
        investment_usd: float,
        holding_period_days: int = 365,
    ) -> list[dict[str, Any]]:
        """Estimate net APY for multiple pools, sorted by net APY descending.

        Args:
            pools: List of pools to evaluate.
            investment_usd: Investment amount in USD.
            holding_period_days: Holding period in days.

        Returns:
            List of estimate dicts sorted by net_apy descending.
        """
        results = [
            self.estimate_net_apy(pool, investment_usd, holding_period_days)
            for pool in pools
        ]
        results.sort(key=lambda x: x["net_apy"], reverse=True)
        return results
