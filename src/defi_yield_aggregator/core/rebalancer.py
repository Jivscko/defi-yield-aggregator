"""Portfolio Rebalancing Engine.

Compares current portfolio allocations to optimized target allocations,
calculates drift, recommends rebalancing trades, and estimates costs.
Supports threshold-based triggering, cost-benefit analysis, minimum trade
sizes, and multi-chain trade batching.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from defi_yield_aggregator.core.gas_estimator import (
    GasEstimator,
    Operation,
)
from defi_yield_aggregator.core.models import (
    Chain,
    PoolInfo,
    Protocol,
)


class TradeDirection(str, Enum):
    """Direction of a rebalance trade."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class RebalanceTrade:
    """A single recommended rebalancing trade.

    Attributes:
        pool_id: Identifier of the pool to trade.
        direction: BUY to increase allocation, SELL to decrease.
        amount_usd: Trade size in USD.
        estimated_gas_usd: Estimated gas cost for this trade.
        chain: Blockchain where the pool resides.
        protocol: DeFi protocol of the pool.
        current_weight: Current allocation weight (0-1).
        target_weight: Target allocation weight (0-1).
        priority: Drift magnitude used for sorting (higher = more urgent).
    """

    pool_id: str
    direction: TradeDirection
    amount_usd: float
    estimated_gas_usd: float
    chain: str
    protocol: str
    current_weight: float
    target_weight: float
    priority: float


@dataclass
class RebalanceReport:
    """Comprehensive report from a rebalancing analysis.

    Attributes:
        trades: Recommended trades sorted by priority (descending).
        total_gas_cost: Total estimated gas cost for all trades.
        expected_apy_before: Weighted APY before rebalancing.
        expected_apy_after: Weighted APY after rebalancing.
        net_benefit_annualized: Net annualized benefit (APY gain minus
            amortized gas costs).
        max_drift: Maximum drift detected across all positions.
        triggered: Whether the rebalance threshold was exceeded.
        current_allocations: Current pool_id -> weight mapping.
        target_allocations: Target pool_id -> weight mapping.
    """

    trades: list[RebalanceTrade]
    total_gas_cost: float
    expected_apy_before: float
    expected_apy_after: float
    net_benefit_annualized: float
    max_drift: float
    triggered: bool
    current_allocations: dict[str, float]
    target_allocations: dict[str, float]


@dataclass(frozen=True)
class RebalanceConfig:
    """Configuration for the rebalancing engine.

    Attributes:
        drift_threshold_pct: Absolute drift threshold to trigger rebalance
            (e.g., 0.05 = 5%).
        relative_drift_threshold: Relative drift threshold (e.g., 0.30 = 30%
            of target weight).
        min_trade_usd: Minimum trade size in USD to avoid dust trades.
        gas_cost_multiplier: Safety buffer multiplier on gas estimates.
        holding_period_days: Holding period for cost-benefit analysis.
        max_trades_per_rebalance: Maximum number of trades per rebalance.
        use_absolute_drift: If True, use absolute drift for triggering;
            if False, use relative drift.
    """

    drift_threshold_pct: float = 0.05
    relative_drift_threshold: float = 0.30
    min_trade_usd: float = 50.0
    gas_cost_multiplier: float = 1.2
    holding_period_days: int = 365
    max_trades_per_rebalance: int = 20
    use_absolute_drift: bool = True


def _get_pool_info_map(pools: list[PoolInfo]) -> dict[str, PoolInfo]:
    """Build a lookup dict from pool_id to PoolInfo.

    Args:
        pools: List of pool information objects.

    Returns:
        Dict mapping pool_id to PoolInfo.
    """
    return {p.pool_id: p for p in pools}


def _compute_drift(
    current: dict[str, float],
    target: dict[str, float],
    use_absolute: bool = True,
) -> dict[str, float]:
    """Compute per-pool drift between current and target allocations.

    Args:
        current: Current allocation weights {pool_id: weight}.
        target: Target allocation weights {pool_id: weight}.
        use_absolute: If True, return absolute drift; otherwise relative.

    Returns:
        Dict mapping pool_id to drift value.
    """
    all_pools = set(current.keys()) | set(target.keys())
    drift: dict[str, float] = {}

    for pool_id in all_pools:
        cur = current.get(pool_id, 0.0)
        tgt = target.get(pool_id, 0.0)
        abs_drift = abs(cur - tgt)

        if use_absolute:
            drift[pool_id] = abs_drift
        else:
            # Relative drift: absolute drift / target weight
            if tgt > 1e-9:
                drift[pool_id] = abs_drift / tgt
            else:
                # Target is zero — any current position is infinite relative drift
                drift[pool_id] = float("inf") if abs_drift > 1e-9 else 0.0

    return drift


def _estimate_trade_gas(
    gas_estimator: GasEstimator,
    chain: Chain,
    protocol: Protocol,
    direction: TradeDirection,
    multiplier: float = 1.2,
) -> float:
    """Estimate gas cost for a single trade direction.

    Args:
        gas_estimator: The gas estimator instance.
        chain: Target blockchain.
        protocol: Target DeFi protocol.
        direction: BUY maps to DEPOSIT, SELL maps to WITHDRAW.
        multiplier: Safety buffer on the estimate.

    Returns:
        Estimated gas cost in USD.
    """
    operation = Operation.DEPOSIT if direction == TradeDirection.BUY else Operation.WITHDRAW
    estimate = gas_estimator.estimate_gas(operation, chain, protocol)
    return estimate.total_cost_usd * multiplier


class PortfolioRebalancer:
    """Rebalancing engine that compares current vs target allocations.

    Analyzes drift, generates trade recommendations with cost-benefit
    filtering, and produces comprehensive rebalance reports.

    Example::

        rebalancer = PortfolioRebalancer(
            pools=pool_list,
            config=RebalanceConfig(drift_threshold_pct=0.05),
            current_portfolio_value=100_000.0,
        )
        report = rebalancer.analyze(
            current_weights={"pool_a": 0.6, "pool_b": 0.4},
            target_weights={"pool_a": 0.5, "pool_b": 0.3, "pool_c": 0.2},
        )
        if report.triggered:
            for trade in report.trades:
                print(f"{trade.direction.value} ${trade.amount_usd:.0f} in {trade.pool_id}")
    """

    def __init__(
        self,
        pools: list[PoolInfo],
        config: RebalanceConfig | None = None,
        current_portfolio_value: float = 10_000.0,
        gas_estimator: GasEstimator | None = None,
    ) -> None:
        """Initialize the portfolio rebalancer.

        Args:
            pools: List of available pool information.
            config: Rebalancing configuration. Uses defaults if None.
            current_portfolio_value: Total portfolio value in USD.
            gas_estimator: Gas estimator instance. Creates a default if None.
        """
        self._pools = pools
        self._pool_map = _get_pool_info_map(pools)
        self._config = config or RebalanceConfig()
        self._portfolio_value = current_portfolio_value
        self._gas_estimator = gas_estimator or GasEstimator()

    def compute_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute per-pool drift between current and target allocations.

        Args:
            current_weights: Current allocation weights {pool_id: weight}.
            target_weights: Target allocation weights {pool_id: weight}.

        Returns:
            Dict mapping pool_id to drift magnitude.
        """
        return _compute_drift(
            current_weights,
            target_weights,
            use_absolute=self._config.use_absolute_drift,
        )

    def should_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> tuple[bool, float]:
        """Determine if rebalancing should be triggered.

        Args:
            current_weights: Current allocation weights.
            target_weights: Target allocation weights.

        Returns:
            Tuple of (should_rebalance, max_drift).
        """
        drift = self.compute_drift(current_weights, target_weights)

        if not drift:
            return False, 0.0

        max_drift = max(drift.values())
        threshold = (
            self._config.drift_threshold_pct
            if self._config.use_absolute_drift
            else self._config.relative_drift_threshold
        )

        return max_drift > threshold, max_drift

    def generate_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> list[RebalanceTrade]:
        """Generate rebalancing trades based on drift analysis.

        Filters trades by minimum size and cost-benefit analysis, then
        sorts by priority (drift magnitude) descending.

        Args:
            current_weights: Current allocation weights.
            target_weights: Target allocation weights.

        Returns:
            List of recommended trades, sorted by priority.
        """
        all_pools = set(current_weights.keys()) | set(target_weights.keys())
        trades: list[RebalanceTrade] = []

        for pool_id in all_pools:
            cur = current_weights.get(pool_id, 0.0)
            tgt = target_weights.get(pool_id, 0.0)
            diff = tgt - cur  # positive = need to buy, negative = need to sell

            if abs(diff) < 1e-9:
                continue

            direction = TradeDirection.BUY if diff > 0 else TradeDirection.SELL
            amount_usd = abs(diff) * self._portfolio_value

            # Filter by minimum trade size
            if amount_usd < self._config.min_trade_usd:
                continue

            # Look up pool info for chain/protocol
            pool_info = self._pool_map.get(pool_id)
            if pool_info is not None:
                chain = pool_info.chain
                protocol = pool_info.protocol
                chain_enum = chain
                protocol_enum = protocol
                chain_str = chain.value if hasattr(chain, "value") else str(chain)
                protocol_str = protocol.value if hasattr(protocol, "value") else str(protocol)
            else:
                # Fallback: try to parse from pool_id or use defaults
                chain_str = "ethereum"
                protocol_str = "aave"
                chain_enum = Chain.ETHEREUM
                protocol_enum = Protocol.AAVE

            # Estimate gas
            gas_usd = _estimate_trade_gas(
                self._gas_estimator,
                chain_enum,
                protocol_enum,
                direction,
                self._config.gas_cost_multiplier,
            )

            # Cost-benefit: skip if gas > expected APY improvement over holding period
            if pool_info is not None:
                apy_improvement = self._compute_apy_improvement(
                    pool_id, cur, tgt, gas_usd
                )
                if apy_improvement < 0:
                    # Gas cost exceeds benefit — skip this trade
                    continue

            drift = abs(cur - tgt)
            trades.append(
                RebalanceTrade(
                    pool_id=pool_id,
                    direction=direction,
                    amount_usd=round(amount_usd, 2),
                    estimated_gas_usd=round(gas_usd, 4),
                    chain=chain_str,
                    protocol=protocol_str,
                    current_weight=cur,
                    target_weight=tgt,
                    priority=drift,
                )
            )

        # Sort by priority descending (largest drift first)
        trades.sort(key=lambda t: t.priority, reverse=True)

        # Limit number of trades
        trades = trades[: self._config.max_trades_per_rebalance]

        return trades

    def _compute_apy_improvement(
        self,
        pool_id: str,
        current_weight: float,
        target_weight: float,
        gas_cost_usd: float,
    ) -> float:
        """Compute net APY improvement for a trade after gas costs.

        For a BUY trade: the new position earns APY on the incremental amount.
        For a SELL trade: the freed capital can be redeployed.

        Args:
            pool_id: Pool identifier.
            current_weight: Current allocation weight.
            target_weight: Target allocation weight.
            gas_cost_usd: Estimated gas cost.

        Returns:
            Net APY improvement (in USD per year). Negative means gas exceeds benefit.
        """
        pool_info = self._pool_map.get(pool_id)
        if pool_info is None:
            return 0.0  # No pool info — can't evaluate, assume ok

        incremental_usd = abs(target_weight - current_weight) * self._portfolio_value
        if incremental_usd < 1e-9:
            return -1.0  # No meaningful trade

        # Annual APY gain from the incremental position
        annual_apy_gain = incremental_usd * pool_info.apy

        # Amortize gas over the holding period
        holding_years = self._config.holding_period_days / 365.0
        gas_per_year = gas_cost_usd / holding_years if holding_years > 0 else gas_cost_usd

        net = annual_apy_gain - gas_per_year
        return net

    def compute_weighted_apy(
        self,
        weights: dict[str, float],
    ) -> float:
        """Compute weighted average APY for a set of allocations.

        Args:
            weights: Pool allocation weights.

        Returns:
            Weighted APY as a decimal.
        """
        total_apy = 0.0
        for pool_id, weight in weights.items():
            pool_info = self._pool_map.get(pool_id)
            if pool_info is not None:
                total_apy += weight * pool_info.apy
        return total_apy

    def batch_trades_by_chain(
        self,
        trades: list[RebalanceTrade],
    ) -> dict[str, list[RebalanceTrade]]:
        """Group trades by chain for optimal execution order.

        Args:
            trades: List of trades to group.

        Returns:
            Dict mapping chain name to list of trades on that chain.
        """
        batches: dict[str, list[RebalanceTrade]] = defaultdict(list)
        for trade in trades:
            batches[trade.chain].append(trade)
        return dict(batches)

    def analyze(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> RebalanceReport:
        """Run full rebalancing analysis and produce a report.

        Performs drift detection, threshold checking, trade generation
        with cost-benefit filtering, and assembles the final report.

        Args:
            current_weights: Current allocation weights {pool_id: weight}.
            target_weights: Target allocation weights {pool_id: weight}.

        Returns:
            RebalanceReport with all analysis results.
        """
        # Handle empty inputs
        if not current_weights and not target_weights:
            return RebalanceReport(
                trades=[],
                total_gas_cost=0.0,
                expected_apy_before=0.0,
                expected_apy_after=0.0,
                net_benefit_annualized=0.0,
                max_drift=0.0,
                triggered=False,
                current_allocations={},
                target_allocations={},
            )

        # Check if rebalance is needed
        triggered, max_drift = self.should_rebalance(current_weights, target_weights)

        # Compute APY before and after
        apy_before = self.compute_weighted_apy(current_weights)
        apy_after = self.compute_weighted_apy(target_weights)

        if not triggered:
            return RebalanceReport(
                trades=[],
                total_gas_cost=0.0,
                expected_apy_before=round(apy_before, 6),
                expected_apy_after=round(apy_after, 6),
                net_benefit_annualized=0.0,
                max_drift=round(max_drift, 6),
                triggered=False,
                current_allocations=dict(current_weights),
                target_allocations=dict(target_weights),
            )

        # Generate trades
        trades = self.generate_trades(current_weights, target_weights)

        # Compute totals
        total_gas = sum(t.estimated_gas_usd for t in trades)

        # Net benefit: APY improvement in USD - gas costs amortized over holding period
        apy_diff = apy_after - apy_before
        annual_apy_improvement_usd = apy_diff * self._portfolio_value
        holding_years = self._config.holding_period_days / 365.0
        gas_per_year = total_gas / holding_years if holding_years > 0 else total_gas
        net_benefit = annual_apy_improvement_usd - gas_per_year

        return RebalanceReport(
            trades=trades,
            total_gas_cost=round(total_gas, 4),
            expected_apy_before=round(apy_before, 6),
            expected_apy_after=round(apy_after, 6),
            net_benefit_annualized=round(net_benefit, 2),
            max_drift=round(max_drift, 6),
            triggered=True,
            current_allocations=dict(current_weights),
            target_allocations=dict(target_weights),
        )
