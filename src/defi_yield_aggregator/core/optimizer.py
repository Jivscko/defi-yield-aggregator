"""Portfolio optimizer - maximize yield given risk constraints.

Supports three allocation strategies:

1. **Greedy** (default): Sort by yield/risk ratio and allocate top pools.
2. **Kelly Criterion**: Use Kelly formula to size positions based on expected
   edge and variance, maximizing long-term geometric growth rate.
3. **Risk Parity**: Allocate so each position contributes equally to total
   portfolio risk, using inverse-volatility weighting.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from defi_yield_aggregator.core.models import (
    Config,
    OptimizedPortfolio,
    PoolInfo,
    PortfolioAllocation,
    RiskScore,
)
from defi_yield_aggregator.core.risk_engine import RiskEngine


class AllocationStrategy(str, Enum):
    """Available portfolio allocation strategies."""

    GREEDY = "greedy"
    KELLY = "kelly"
    RISK_PARITY = "risk_rarity"
    BLACK_LITTERMAN = "black_litterman"


def _pool_volatility(pool: PoolInfo, risk: RiskScore) -> float:
    """Estimate volatility for a pool from available risk signals.

    Uses the risk score as a proxy for return uncertainty.  Higher risk pools
    are assumed to have wider return distributions.  The result is a per-period
    (daily-equivalent) standard deviation expressed as a decimal.

    Args:
        pool: Pool information.
        risk: Risk score for the pool.

    Returns:
        Estimated volatility (>= 0.001 to avoid division by zero).
    """
    # Base volatility from risk score (0-100 mapped to 0.01 - 0.40)
    base_vol = 0.01 + (risk.overall_score / 100.0) * 0.39

    # Impermanent loss amplifies volatility
    il_boost = pool.impermanent_loss_risk * 0.15

    # Stable pools are inherently less volatile
    stable_factor = 0.5 if pool.is_stable else 1.0

    vol = (base_vol + il_boost) * stable_factor
    return max(vol, 0.001)


def _pool_edge(pool: PoolInfo, risk: RiskScore) -> float:
    """Estimate the expected edge (net excess return) for a pool.

    Edge = APY minus a risk-adjusted discount.  This serves as the "win rate"
    analogue for Kelly sizing.

    Args:
        pool: Pool information.
        risk: Risk score for the pool.

    Returns:
        Estimated edge (can be negative for unprofitable pools).
    """
    # Risk-adjusted discount: higher risk -> larger haircut
    risk_discount = (risk.overall_score / 100.0) * pool.apy * 0.5
    return pool.apy - risk_discount


def _greedy_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
) -> OptimizedPortfolio:
    """Greedy allocation: sort by risk-adjusted yield and allocate.

    Strategy:
    1. Score each pool by yield / risk ratio
    2. Allocate up to max_single_allocation_pct per pool
    3. Stop when budget exhausted or max_positions reached
    """
    # Sort by yield-to-risk ratio (higher = better)
    scored = []
    for pool, risk in candidates:
        ratio = pool.apy / max(risk.overall_score, 1.0)
        scored.append((pool, risk, ratio))

    scored.sort(key=lambda x: x[2], reverse=True)

    allocations: list[PortfolioAllocation] = []
    remaining_pct = 1.0

    for pool, risk, ratio in scored:
        if len(allocations) >= config.max_positions:
            break
        if remaining_pct <= 0.001:
            break

        alloc_pct = min(config.max_single_allocation_pct, remaining_pct)
        amount = investment_usd * alloc_pct

        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(alloc_pct, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )
        remaining_pct -= alloc_pct

    if not allocations:
        return OptimizedPortfolio(
            allocations=[],
            total_expected_apy=0.0,
            weighted_risk_score=0.0,
            total_investment_usd=0.0,
        )

    # Distribute remaining allocation proportionally
    if remaining_pct > 0.001 and allocations:
        bonus = remaining_pct / len(allocations)
        for a in allocations:
            a.allocation_pct = min(config.max_single_allocation_pct, a.allocation_pct + bonus)
            a.amount_usd = round(investment_usd * a.allocation_pct, 2)

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


def _kelly_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
    kelly_fraction: float = 0.5,
) -> OptimizedPortfolio:
    """Kelly Criterion allocation: size positions for maximum geometric growth.

    The Kelly Criterion determines the optimal fraction of wealth to allocate
    to each bet (pool) to maximize the expected logarithm of terminal wealth.
    We use a fractional Kelly (default 0.5) to reduce volatility.

    For each pool the Kelly fraction is::

        f* = edge / variance

    where *edge* is the risk-adjusted expected excess return and *variance* is
    the squared estimated volatility.  Positions are then capped at
    ``max_single_allocation_pct`` and re-normalized.

    Args:
        candidates: Pools with their risk scores.
        investment_usd: Total capital to allocate.
        config: Portfolio constraints.
        kelly_fraction: Fractional Kelly multiplier (0-1).  Lower is more
            conservative.  Default 0.5 (half-Kelly) balances growth and
            drawdown risk.

    Returns:
        Optimized portfolio with Kelly-sized allocations.
    """
    if not candidates:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Compute raw Kelly fractions
    kelly_fracs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk in candidates:
        edge = _pool_edge(pool, risk)
        vol = _pool_volatility(pool, risk)
        variance = vol ** 2

        if variance <= 0 or edge <= 0:
            raw_kelly = 0.0
        else:
            raw_kelly = edge / variance

        # Apply fractional Kelly
        frac = raw_kelly * kelly_fraction
        kelly_fracs.append((pool, risk, frac))

    # Filter out zero-allocation pools
    kelly_fracs = [(p, r, f) for p, r, f in kelly_fracs if f > 1e-6]

    if not kelly_fracs:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Sort by Kelly fraction descending, take top N
    kelly_fracs.sort(key=lambda x: x[2], reverse=True)
    kelly_fracs = kelly_fracs[: config.max_positions]

    # Normalize fractions to sum to at most 1.0
    total_raw = sum(f for _, _, f in kelly_fracs)
    if total_raw > 1.0:
        kelly_fracs = [(p, r, f / total_raw) for p, r, f in kelly_fracs]

    # Cap at max_single_allocation_pct and re-normalize
    allocations: list[PortfolioAllocation] = []
    capped_fracs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk, frac in kelly_fracs:
        capped = min(frac, config.max_single_allocation_pct)
        capped_fracs.append((pool, risk, capped))

    # Re-normalize if capping caused total < 1.0
    capped_total = sum(f for _, _, f in capped_fracs)
    if capped_total > 0 and capped_total < 0.999:
        scale = min(1.0 / capped_total, 1.0)
        capped_fracs = [(p, r, f * scale) for p, r, f in capped_fracs]

    for pool, risk, frac in capped_fracs:
        if frac < 0.001:
            continue
        amount = investment_usd * frac
        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(frac, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )

    if not allocations:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


def _risk_parity_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
) -> OptimizedPortfolio:
    """Risk Parity allocation: equal risk contribution from each position.

    Allocates inversely proportional to volatility so that every pool
    contributes the same amount of portfolio risk.  This is the
    "inverse-volatility weighting" variant of risk parity — simple,
    effective, and widely used.

    Weight_i = (1 / vol_i) / sum(1 / vol_j)

    Args:
        candidates: Pools with their risk scores.
        investment_usd: Total capital to allocate.
        config: Portfolio constraints.

    Returns:
        Optimized portfolio with risk-parity allocations.
    """
    if not candidates:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Compute inverse-volatility weights
    inv_vol_pairs: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk in candidates:
        vol = _pool_volatility(pool, risk)
        inv_vol = 1.0 / vol
        inv_vol_pairs.append((pool, risk, inv_vol))

    # Sort by inverse-volatility descending (most stable first)
    inv_vol_pairs.sort(key=lambda x: x[2], reverse=True)

    # Take top max_positions
    inv_vol_pairs = inv_vol_pairs[: config.max_positions]

    total_inv_vol = sum(iv for _, _, iv in inv_vol_pairs)
    if total_inv_vol <= 0:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    # Raw risk-parity weights
    rp_weights = [(p, r, iv / total_inv_vol) for p, r, iv in inv_vol_pairs]

    # Cap at max_single_allocation_pct and re-normalize
    allocations: list[PortfolioAllocation] = []
    capped_weights: list[tuple[PoolInfo, RiskScore, float]] = []
    for pool, risk, w in rp_weights:
        capped = min(w, config.max_single_allocation_pct)
        capped_weights.append((pool, risk, capped))

    capped_total = sum(w for _, _, w in capped_weights)
    if capped_total > 0 and capped_total < 0.999:
        scale = min(1.0 / capped_total, 1.0)
        capped_weights = [(p, r, w * scale) for p, r, w in capped_weights]

    for pool, risk, w in capped_weights:
        if w < 0.001:
            continue
        amount = investment_usd * w
        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(w, 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )

    if not allocations:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


def _black_litterman_optimize(
    candidates: list[tuple[PoolInfo, RiskScore]],
    investment_usd: float,
    config: Config,
    tau: float = 0.05,
) -> OptimizedPortfolio:
    """Black-Litterman allocation: blend market equilibrium with investor views.

    The Black-Litterman model starts from a market-cap-weighted prior
    (implied equilibrium returns based on TVL proportions) and then
    blends in investor "views" (expressed through APY expectations and
    risk confidence) to produce posterior expected returns.  The result
    is a portfolio that tilts away from the market portfolio only where
    the investor has high-conviction views.

    **DeFi adaptation:**

    - Market weights ``w_mkt`` are derived from each pool's TVL share
      (pools with more TVL represent the market's revealed preference).
    - The risk aversion parameter ``delta`` controls how sensitive the
      equilibrium is to volatility.
    - Investor views are constructed from each pool's APY relative to
      the market-implied return, with uncertainty proportional to the
      risk score (higher risk → less confident view).
    - The covariance matrix ``Sigma`` is estimated from the risk-score
      based volatility model (same as Kelly / Risk Parity).

    Args:
        candidates: Pools with their risk scores.
        investment_usd: Total capital to allocate.
        config: Portfolio constraints.
        tau: Uncertainty scaling factor for the prior (typically 0.01–0.1).
            Higher ``tau`` means less trust in the equilibrium and more
            weight on the investor views.  Default 0.05.

    Returns:
        Optimized portfolio with Black-Litterman posterior allocations.
    """
    n = len(candidates)
    if n == 0:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    pools = [p for p, _ in candidates]
    risks = [r for _, r in candidates]

    # --- Step 1: Market weights from TVL proportions ---
    tvls = np.array([max(p.tvl_usd, 1.0) for p in pools])
    w_mkt = tvls / tvls.sum()

    # --- Step 2: Covariance matrix from risk-based volatilities ---
    vols = np.array([_pool_volatility(p, r) for p, r in candidates])
    Sigma = np.diag(vols ** 2)

    # --- Step 3: Risk aversion parameter ---
    # Use a moderate risk aversion (common literature value: 2.5)
    delta = 2.5

    # --- Step 4: Implied equilibrium returns (prior) ---
    # pi = delta * Sigma * w_mkt
    pi = delta * Sigma @ w_mkt

    # --- Step 5: Investor views ---
    # Each pool gets an "absolute view": the expected return from its APY,
    # discounted by risk.  P = I (identity — one view per asset).
    # Q = risk-adjusted APY for each pool.
    # Omega = diagonal uncertainty, proportional to risk score.
    P = np.eye(n)
    Q = np.array([
        p.apy * (1.0 - r.overall_score / 200.0)  # risk discount
        for p, r in candidates
    ])

    # View uncertainty: higher risk → higher uncertainty (less confident)
    # Base uncertainty scales with tau * Sigma (the Idzorek approach)
    omega_diag = np.array([
        tau * (r.overall_score / 100.0) * vols[i] ** 2
        for i, (_, r) in enumerate(candidates)
    ])
    Omega = np.diag(omega_diag)

    # --- Step 6: Black-Litterman posterior returns ---
    # mu_BL = [(tau * Sigma)^-1 + P^T Omega^-1 P]^-1
    #          * [(tau * Sigma)^-1 pi + P^T Omega^-1 Q]
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    M = tau_Sigma_inv + P.T @ Omega_inv @ P
    mu_BL = np.linalg.solve(M, tau_Sigma_inv @ pi + Omega_inv @ Q)

    # --- Step 7: Optimal weights from posterior returns ---
    # w* = (delta * Sigma)^-1 * mu_BL
    w_star = np.linalg.solve(delta * Sigma, mu_BL)

    # --- Step 8: Normalize, cap, and re-normalize ---
    # Remove negative weights (short selling not allowed in DeFi)
    w_star = np.maximum(w_star, 0.0)
    total_w = w_star.sum()
    if total_w > 0:
        w_star = w_star / total_w
    else:
        # Fallback to equal weight if all weights collapsed
        w_star = np.ones(n) / n

    # Take only the top max_positions pools by weight
    if n > config.max_positions:
        top_indices = np.argsort(w_star)[::-1][: config.max_positions]
        mask = np.zeros(n, dtype=bool)
        mask[top_indices] = True
        w_star = np.where(mask, w_star, 0.0)
        total_w = w_star.sum()
        if total_w > 0:
            w_star = w_star / total_w

    # Cap at max_single_allocation_pct (iterate until stable)
    for _ in range(10):
        w_star = np.minimum(w_star, config.max_single_allocation_pct)
        total_w = w_star.sum()
        if total_w >= 0.999:
            break
        if total_w > 0:
            w_star = w_star * (1.0 / total_w)
        else:
            break

    # --- Step 9: Build allocations ---
    allocations: list[PortfolioAllocation] = []
    for i, (pool, risk) in enumerate(candidates):
        frac = w_star[i]
        if frac < 0.001:
            continue
        amount = investment_usd * frac
        allocations.append(
            PortfolioAllocation(
                pool_id=pool.pool_id,
                protocol=pool.protocol,
                chain=pool.chain,
                token_pair=pool.token_pair,
                allocation_pct=round(float(frac), 4),
                expected_apy=pool.apy,
                risk_score=risk.overall_score,
                amount_usd=round(amount, 2),
            )
        )

    if not allocations:
        return OptimizedPortfolio(
            allocations=[], total_expected_apy=0.0,
            weighted_risk_score=0.0, total_investment_usd=0.0,
        )

    total_apy = sum(a.allocation_pct * a.expected_apy for a in allocations)
    total_risk = sum(a.allocation_pct * a.risk_score for a in allocations)

    return OptimizedPortfolio(
        allocations=allocations,
        total_expected_apy=round(total_apy, 6),
        weighted_risk_score=round(total_risk, 2),
        total_investment_usd=investment_usd,
    )


class PortfolioOptimizer:
    """Optimize DeFi portfolio allocation to maximize yield under risk constraints.

    Supports four allocation strategies:

    - ``GREEDY``: Sort by yield/risk ratio and allocate top pools.
    - ``KELLY``: Kelly Criterion sizing for maximum geometric growth.
    - ``RISK_PARITY``: Inverse-volatility weighting for equal risk contribution.
    - ``BLACK_LITTERMAN``: Bayesian blending of market equilibrium with investor views.

    Args:
        config: Portfolio constraints and preferences.
        strategy: Allocation strategy to use.
        kelly_fraction: Fractional Kelly multiplier (only used with KELLY strategy).
            0.5 = half-Kelly (conservative), 1.0 = full Kelly (aggressive).
        bl_tau: Uncertainty scaling factor for the prior (only used with
            BLACK_LITTERMAN strategy).  Higher values mean less trust in
            the market equilibrium and more weight on investor views.
            Typical range: 0.01–0.1.  Default 0.05.

    Example::

        optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
        portfolio = optimizer.optimize(pools, investment_usd=100_000)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        strategy: AllocationStrategy = AllocationStrategy.GREEDY,
        kelly_fraction: float = 0.5,
        bl_tau: float = 0.05,
    ) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in (0, 1], got {kelly_fraction}"
            )
        if not 0.0 < bl_tau <= 1.0:
            raise ValueError(
                f"bl_tau must be in (0, 1], got {bl_tau}"
            )
        self.config = config or Config()
        self.risk_engine = RiskEngine()
        self.strategy = strategy
        self.kelly_fraction = kelly_fraction
        self.bl_tau = bl_tau

    def optimize(
        self,
        pools: list[PoolInfo],
        investment_usd: float,
    ) -> OptimizedPortfolio:
        """Find optimal allocation across pools.

        Args:
            pools: Available yield farming pools.
            investment_usd: Total investment amount in USD.

        Returns:
            Optimized portfolio with allocations.

        Raises:
            ValueError: If no pools meet risk criteria.
        """
        if investment_usd <= 0:
            raise ValueError(f"Investment must be positive, got {investment_usd}")

        # Filter by minimum TVL
        eligible = [p for p in pools if p.tvl_usd >= self.config.min_tvl_usd]

        if self.config.stablecoins_only:
            eligible = [p for p in eligible if p.is_stable]

        # Filter by preferred chains/protocols
        if self.config.preferred_chains:
            eligible = [p for p in eligible if p.chain in self.config.preferred_chains]
        if self.config.preferred_protocols:
            eligible = [p for p in eligible if p.protocol in self.config.preferred_protocols]

        # Score and filter by risk
        risk_filtered = self.risk_engine.filter_by_risk(
            eligible, max_risk=self.config.max_risk_score
        )

        if not risk_filtered:
            return OptimizedPortfolio(
                allocations=[],
                total_expected_apy=0.0,
                weighted_risk_score=0.0,
                total_investment_usd=investment_usd,
            )

        # Dispatch to the selected strategy
        if self.strategy is AllocationStrategy.KELLY:
            return _kelly_optimize(
                risk_filtered, investment_usd, self.config, self.kelly_fraction
            )
        if self.strategy is AllocationStrategy.RISK_PARITY:
            return _risk_parity_optimize(risk_filtered, investment_usd, self.config)
        if self.strategy is AllocationStrategy.BLACK_LITTERMAN:
            return _black_litterman_optimize(
                risk_filtered, investment_usd, self.config, self.bl_tau
            )
        # Default: greedy
        return _greedy_optimize(risk_filtered, investment_usd, self.config)
