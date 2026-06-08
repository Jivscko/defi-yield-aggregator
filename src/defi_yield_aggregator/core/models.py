"""Data models for the DeFi Yield Aggregator."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Chain(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BASE = "base"
    AVALANCHE = "avalanche"


class Protocol(str, Enum):
    """Supported DeFi protocols."""
    AAVE = "aave"
    COMPOUND = "compound"
    UNISWAP = "uniswap"
    CURVE = "curve"
    YEARN = "yearn"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PoolInfo(BaseModel):
    """Represents a yield farming pool/vault."""
    protocol: Protocol
    chain: Chain
    pool_id: str
    pool_name: str
    token_pair: str
    apy: float = Field(ge=0, description="Annual Percentage Yield as decimal (e.g., 0.05 = 5%)")
    tvl_usd: float = Field(ge=0, description="Total Value Locked in USD")
    daily_volume_usd: float = Field(ge=0, default=0.0)
    is_stable: bool = False
    impermanent_loss_risk: float = Field(ge=0, le=1, default=0.0)
    deposit_fee: float = Field(ge=0, le=1, default=0.0)
    withdrawal_fee: float = Field(ge=0, le=1, default=0.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("apy")
    @classmethod
    def validate_apy(cls, v: float) -> float:
        """Reject unreasonably high APY values (>10000%)."""
        if v > 100:
            raise ValueError(f"APY {v} exceeds 10000% threshold - likely erroneous data")
        return v


class RiskScore(BaseModel):
    """Risk assessment for a pool."""
    pool_id: str
    protocol: Protocol
    overall_score: float = Field(ge=0, le=100, description="Risk score 0-100 (lower = safer)")
    risk_level: RiskLevel
    tvl_score: float = Field(ge=0, le=100, default=0.0)
    age_score: float = Field(ge=0, le=100, default=0.0)
    audit_score: float = Field(ge=0, le=100, default=0.0)
    chain_diversity_score: float = Field(ge=0, le=100, default=0.0)
    details: dict[str, str] = Field(default_factory=dict)

    @field_validator("risk_level", mode="before")
    @classmethod
    def classify_risk(cls, v: object, info: object) -> RiskLevel:
        """Auto-classify risk level from overall score if needed."""
        if isinstance(v, RiskLevel):
            return v
        return RiskLevel.LOW  # Will be overridden by engine


class PortfolioAllocation(BaseModel):
    """A single allocation in an optimized portfolio."""
    pool_id: str
    protocol: Protocol
    chain: Chain
    token_pair: str
    allocation_pct: float = Field(ge=0, le=1, description="Fraction of portfolio (0-1)")
    expected_apy: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=100)
    amount_usd: float = Field(ge=0, default=0.0)


class OptimizedPortfolio(BaseModel):
    """Result of portfolio optimization."""
    allocations: list[PortfolioAllocation]
    total_expected_apy: float = Field(ge=0)
    weighted_risk_score: float = Field(ge=0, le=100)
    total_investment_usd: float = Field(ge=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def num_positions(self) -> int:
        return len(self.allocations)

    @property
    def diversification_score(self) -> float:
        """Higher score = more diversified (0-100)."""
        if not self.allocations:
            return 0.0
        import math
        entropy = 0.0
        for a in self.allocations:
            if a.allocation_pct > 0:
                entropy -= a.allocation_pct * math.log(a.allocation_pct)
        max_entropy = math.log(len(self.allocations))
        if max_entropy == 0:
            return 0.0
        return (entropy / max_entropy) * 100


class Config(BaseModel):
    """Application configuration."""
    max_risk_score: float = Field(default=70.0, ge=0, le=100)
    min_tvl_usd: float = Field(default=1_000_000.0, ge=0)
    max_single_allocation_pct: float = Field(default=0.30, gt=0, le=1)
    min_positions: int = Field(default=2, ge=1)
    max_positions: int = Field(default=10, ge=1)
    preferred_chains: list[Chain] = Field(default_factory=lambda: list(Chain))
    preferred_protocols: list[Protocol] = Field(default_factory=lambda: list(Protocol))
    stablecoins_only: bool = False
    refresh_interval_seconds: int = Field(default=300, ge=30)
