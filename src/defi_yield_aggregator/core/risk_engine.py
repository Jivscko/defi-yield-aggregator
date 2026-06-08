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
}

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
        tvl_weight: float = 0.35,
        age_weight: float = 0.20,
        audit_weight: float = 0.25,
        chain_weight: float = 0.20,
    ) -> None:
        total = tvl_weight + age_weight + audit_weight + chain_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.tvl_weight = tvl_weight
        self.age_weight = age_weight
        self.audit_weight = audit_weight
        self.chain_weight = chain_weight

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

        # Stable pools get a small bonus
        stable_bonus = -5.0 if pool.is_stable else 0.0
        # Impermanent loss risk penalty
        il_penalty = pool.impermanent_loss_risk * 20.0

        overall = (
            tvl_s * self.tvl_weight
            + age_s * self.age_weight
            + audit_s * self.audit_weight
            + chain_s * self.chain_weight
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
            details={
                "stable_bonus": str(stable_bonus),
                "il_penalty": str(round(il_penalty, 2)),
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
