"""Risk scoring engine for DeFi protocol pools."""

from __future__ import annotations

from datetime import datetime, timezone

from defi_yield_aggregator.core.models import (
    PoolInfo,
    Protocol,
    RiskLevel,
    RiskScore,
)

# Protocol metadata: (launch_year, num_audits, chains_deployed)
PROTOCOL_META: dict[Protocol, tuple[int, int, int]] = {
    Protocol.AAVE: (2020, 8, 7),
    Protocol.COMPOUND: (2018, 6, 5),
    Protocol.UNISWAP: (2020, 5, 8),
    Protocol.CURVE: (2020, 4, 10),
    Protocol.YEARN: (2020, 3, 4),
    Protocol.LIDO: (2020, 5, 6),
    Protocol.BALANCER: (2021, 5, 5),
    Protocol.CONVEX: (2021, 3, 1),
    Protocol.SUSHISWAP: (2020, 3, 12),
    Protocol.ROCKET_POOL: (2021, 6, 3),
    Protocol.FRAX: (2021, 4, 3),
    Protocol.MAKERDAO: (2017, 10, 3),
}

# Smart contract risk metadata: (complexity_score, has_proxy_pattern, has_composability_risk)
PROTOCOL_SC_META: dict[Protocol, tuple[int, bool, bool]] = {
    Protocol.AAVE: (20, True, True),
    Protocol.COMPOUND: (25, True, True),
    Protocol.UNISWAP: (15, False, True),
    Protocol.CURVE: (35, False, True),
    Protocol.YEARN: (40, True, True),
    Protocol.LIDO: (20, True, False),
    Protocol.BALANCER: (30, True, True),
    Protocol.CONVEX: (35, True, True),
    Protocol.SUSHISWAP: (30, False, True),
    Protocol.ROCKET_POOL: (35, True, False),
    Protocol.FRAX: (30, True, True),
    Protocol.MAKERDAO: (20, True, False),
}
_DEFAULT_SC_META: tuple[int, bool, bool] = (50, True, True)

# Governance risk metadata: (decentralization_pct, has_timelock, has_emergency_pause, token_concentration_pct)
#   decentralization_pct: 0=fully centralized, 100=fully decentralized DAO
#   has_timelock: True if governance decisions have a time delay before execution
#   has_emergency_pause: True if a small group can pause/upgrade contracts (adds risk)
#   token_concentration_pct: 0-100, higher = more voting power concentrated (riskier)
PROTOCOL_GOV_META: dict[Protocol, tuple[int, bool, bool, int]] = {
    Protocol.AAVE: (80, True, True, 30),
    Protocol.COMPOUND: (75, True, False, 35),
    Protocol.UNISWAP: (90, True, True, 25),
    Protocol.CURVE: (60, True, False, 40),
    Protocol.YEARN: (55, True, True, 35),
    Protocol.LIDO: (65, True, True, 45),
    Protocol.BALANCER: (70, True, True, 30),
    Protocol.CONVEX: (45, False, True, 50),
    Protocol.ROCKET_POOL: (85, True, False, 20),
    Protocol.SUSHISWAP: (50, True, True, 45),
    Protocol.FRAX: (55, True, True, 55),
    Protocol.MAKERDAO: (80, True, True, 35),
}
_DEFAULT_GOV_META: tuple[int, bool, bool, int] = (30, False, True, 70)

# TVL thresholds for scoring
TVL_TIERS: list[tuple[float, float]] = [
    (10_000_000_000, 10),  # >$10B → score 10
    (1_000_000_000, 20),   # >$1B → score 20
    (500_000_000, 30),     # >$500M → score 30
    (100_000_000, 40),     # >$100M → score 40
    (10_000_000, 60),      # >$10M → score 60
    (1_000_000, 80),       # >$1M → score 80
    (0, 95),               # <$1M → score 95
]


def _tvl_score(tvl_usd: float) -> float:
    """Score based on Total Value Locked (lower TVL = higher risk)."""
    for threshold, score in TVL_TIERS:
        if tvl_usd >= threshold:
            return score
    return 95.0


def _age_score(protocol: Protocol) -> float:
    """Score based on protocol age (newer = riskier)."""
    launch_year, _, _ = PROTOCOL_META.get(protocol, (2024, 0, 1))
    years_active = datetime.now(timezone.utc).year - launch_year
    if years_active >= 5:
        return 10
    if years_active >= 3:
        return 25
    if years_active >= 2:
        return 40
    return 70


def _audit_score(protocol: Protocol) -> float:
    """Score based on number of security audits (more = safer)."""
    _, num_audits, _ = PROTOCOL_META.get(protocol, (2024, 0, 1))
    if num_audits >= 6:
        return 10
    if num_audits >= 4:
        return 20
    if num_audits >= 2:
        return 40
    if num_audits >= 1:
        return 60
    return 90


def _chain_diversity_score(protocol: Protocol) -> float:
    """Score based on multi-chain deployment (more chains = more battle-tested)."""
    _, _, chains = PROTOCOL_META.get(protocol, (2024, 0, 1))
    if chains >= 7:
        return 10
    if chains >= 4:
        return 25
    if chains >= 2:
        return 45
    return 70


def _liquidity_score(daily_volume_usd: float, tvl_usd: float) -> float:
    """Score based on liquidity risk (lower score = more liquid = safer).

    Uses the daily volume / TVL ratio to assess how easily positions can be
    exited. Also applies a penalty for very low absolute volume.

    Args:
        daily_volume_usd: Daily trading volume in USD.
        tvl_usd: Total Value Locked in USD.

    Returns:
        Liquidity risk score from 0-100 (lower = safer).
    """
    if tvl_usd <= 0:
        ratio = 0.0
    else:
        ratio = daily_volume_usd / tvl_usd

    if ratio >= 0.10:
        score = 10.0
    elif ratio >= 0.05:
        score = 25.0
    elif ratio >= 0.02:
        score = 40.0
    elif ratio >= 0.005:
        score = 60.0
    elif ratio > 0:
        score = 80.0
    else:
        # No volume data — unknown liquidity, moderate-high risk
        score = 70.0

    # Absolute volume penalty: very low volume pools are harder to exit
    if daily_volume_usd < 10_000:
        score = min(100.0, score + 15.0)

    return score


def _smart_contract_risk_score(protocol: Protocol) -> float:
    """Score based on smart contract complexity and known risk patterns.

    Higher scores indicate more complex contracts with greater attack surface.
    Factors include code complexity, proxy/upgradability patterns, and
    composability (interaction with other protocols).

    Args:
        protocol: The DeFi protocol to assess.

    Returns:
        Smart contract risk score from 0-100 (lower = safer).
    """
    complexity, has_proxy, has_composability = PROTOCOL_SC_META.get(
        protocol, _DEFAULT_SC_META
    )
    score = float(complexity)
    if has_proxy:
        score += 10.0
    if has_composability:
        score += 15.0
    return min(100.0, score)


def _governance_risk_score(protocol: Protocol) -> float:
    """Score based on governance decentralization and safety mechanisms.

    Protocols with decentralized DAO governance, timelock delays on proposals,
    and well-distributed token ownership are considered safer. Protocols with
    centralized control, concentrated voting power, or emergency multisig
    capabilities carry more governance risk.

    Args:
        protocol: The DeFi protocol to assess.

    Returns:
        Governance risk score from 0-100 (lower = safer).
    """
    decentralization, has_timelock, has_emergency_pause, concentration = (
        PROTOCOL_GOV_META.get(protocol, _DEFAULT_GOV_META)
    )
    # Base: low decentralization = high risk
    score = float(100 - decentralization)
    # Timelock reduces risk (governance can be reviewed before execution)
    if has_timelock:
        score -= 10.0
    # Emergency pause increases risk (small group can act unilaterally)
    if has_emergency_pause:
        score += 10.0
    # Concentrated tokens increase risk
    score += (concentration - 50) * 0.3
    return max(0.0, min(100.0, score))


def _classify(overall: float) -> RiskLevel:
    """Map numeric score to risk level."""
    if overall <= 30:
        return RiskLevel.LOW
    if overall <= 55:
        return RiskLevel.MEDIUM
    if overall <= 75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


class RiskEngine:
    """Evaluates risk for DeFi pools using multiple factors."""

    def __init__(
        self,
        tvl_weight: float = 0.20,
        age_weight: float = 0.10,
        audit_weight: float = 0.15,
        chain_weight: float = 0.10,
        liquidity_weight: float | None = None,
        sc_risk_weight: float | None = None,
        governance_weight: float | None = None,
    ) -> None:
        # Backward compatibility: if only 4 old-style weights are provided
        # and they sum to 1.0, default new weights to 0.
        if liquidity_weight is None and sc_risk_weight is None and governance_weight is None:
            old_total = tvl_weight + age_weight + audit_weight + chain_weight
            if abs(old_total - 1.0) < 1e-6:
                liquidity_weight = 0.0
                sc_risk_weight = 0.0
                governance_weight = 0.0
            else:
                liquidity_weight = 0.15
                sc_risk_weight = 0.15
                governance_weight = 0.15
        else:
            if liquidity_weight is None:
                liquidity_weight = 0.0
            if sc_risk_weight is None:
                sc_risk_weight = 0.0
            if governance_weight is None:
                governance_weight = 0.0
        total = tvl_weight + age_weight + audit_weight + chain_weight + liquidity_weight + sc_risk_weight + governance_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.tvl_weight = tvl_weight
        self.age_weight = age_weight
        self.audit_weight = audit_weight
        self.chain_weight = chain_weight
        self.liquidity_weight = liquidity_weight
        self.sc_risk_weight = sc_risk_weight
        self.governance_weight = governance_weight

    def score_pool(self, pool: PoolInfo) -> RiskScore:
        """Calculate a comprehensive risk score for a pool.

        Args:
            pool: Pool information to assess.

        Returns:
            Detailed risk score breakdown.
        """
        tvl_s = _tvl_score(pool.tvl_usd)
        age_s = _age_score(pool.protocol)
        audit_s = _audit_score(pool.protocol)
        chain_s = _chain_diversity_score(pool.protocol)
        liq_s = _liquidity_score(pool.daily_volume_usd, pool.tvl_usd)
        sc_s = _smart_contract_risk_score(pool.protocol)
        gov_s = _governance_risk_score(pool.protocol)

        # Stable pools get a small bonus
        stable_bonus = -5.0 if pool.is_stable else 0.0
        # Impermanent loss risk penalty
        il_penalty = pool.impermanent_loss_risk * 20.0

        overall = (
            tvl_s * self.tvl_weight
            + age_s * self.age_weight
            + audit_s * self.audit_weight
            + chain_s * self.chain_weight
            + liq_s * self.liquidity_weight
            + sc_s * self.sc_risk_weight
            + gov_s * self.governance_weight
            + stable_bonus
            + il_penalty
        )
        overall = max(0, min(100, overall))

        return RiskScore(
            pool_id=pool.pool_id,
            protocol=pool.protocol,
            overall_score=round(overall, 2),
            risk_level=_classify(overall),
            tvl_score=round(tvl_s, 2),
            age_score=round(age_s, 2),
            audit_score=round(audit_s, 2),
            chain_diversity_score=round(chain_s, 2),
            liquidity_score=round(liq_s, 2),
            smart_contract_risk_score=round(sc_s, 2),
            governance_risk_score=round(gov_s, 2),
            details={
                "stable_bonus": str(stable_bonus),
                "il_penalty": str(round(il_penalty, 2)),
                "liquidity_score": str(round(liq_s, 2)),
                "smart_contract_risk_score": str(round(sc_s, 2)),
                "governance_risk_score": str(round(gov_s, 2)),
            },
        )

    def score_pools(self, pools: list[PoolInfo]) -> list[RiskScore]:
        """Score multiple pools."""
        return [self.score_pool(p) for p in pools]

    def filter_by_risk(
        self,
        pools: list[PoolInfo],
        max_risk: float = 70.0,
    ) -> list[tuple[PoolInfo, RiskScore]]:
        """Return pools that meet the maximum risk threshold."""
        results = []
        for pool in pools:
            score = self.score_pool(pool)
            if score.overall_score <= max_risk:
                results.append((pool, score))
        return results
