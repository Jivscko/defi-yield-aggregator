# DeFi Yield Aggregator – API Reference

> **Version:** 0.1.0 &nbsp;|&nbsp; **Python:** ≥3.10 &nbsp;|&nbsp; **License:** MIT

---

## Table of Contents

1. [Core Models](#core-models)
2. [Risk Engine](#risk-engine)
3. [Portfolio Optimizer](#portfolio-optimizer)
4. [APY Calculator](#apy-calculator)
5. [Impermanent Loss Calculator](#impermanent-loss-calculator)
6. [Protocol Adapters](#protocol-adapters)
7. [CLI Commands](#cli-commands)

---

## Core Models

All models live in `defi_yield_aggregator.core.models` and use **Pydantic v2**.

### Enums

#### `Chain`

```python
class Chain(str, Enum):
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BASE = "base"
    AVALANCHE = "avalanche"
```

#### `Protocol`

```python
class Protocol(str, Enum):
    AAVE = "aave"
    COMPOUND = "compound"
    UNISWAP = "uniswap"
    CURVE = "curve"
    YEARN = "yearn"
    LIDO = "lido"
    BALANCER = "balancer"
```

#### `RiskLevel`

```python
class RiskLevel(str, Enum):
    LOW = "low"           # overall_score ≤ 30
    MEDIUM = "medium"     # overall_score ≤ 55
    HIGH = "high"         # overall_score ≤ 75
    CRITICAL = "critical" # overall_score > 75
```

---

### `PoolInfo`

Represents a yield farming pool/vault.

```python
class PoolInfo(BaseModel):
    protocol: Protocol
    chain: Chain
    pool_id: str
    pool_name: str
    token_pair: str
    apy: float                    # Annual Percentage Yield (decimal, e.g. 0.05 = 5%). 0–100.
    tvl_usd: float                # Total Value Locked in USD. ≥ 0.
    daily_volume_usd: float = 0.0 # 24 h trading volume. ≥ 0.
    is_stable: bool = False
    impermanent_loss_risk: float = 0.0  # 0–1 scale.
    deposit_fee: float = 0.0      # One-time deposit fee fraction. 0–1.
    withdrawal_fee: float = 0.0   # One-time withdrawal fee fraction. 0–1.
    last_updated: datetime        # Auto-set to utcnow().
```

**Validation:** APY values > 100 (10 000 %) are rejected.

---

### `RiskScore`

Returned by the risk engine for every scored pool.

```python
class RiskScore(BaseModel):
    pool_id: str
    protocol: Protocol
    overall_score: float        # 0–100 (lower = safer)
    risk_level: RiskLevel
    tvl_score: float            # 0–100
    age_score: float            # 0–100
    audit_score: float          # 0–100
    chain_diversity_score: float  # 0–100
    liquidity_score: float       # 0–100
    smart_contract_risk_score: float  # 0–100
    details: dict[str, str]     # Extra info (stable_bonus, il_penalty, etc.)
```

---

### `PortfolioAllocation`

A single allocation within an optimized portfolio.

```python
class PortfolioAllocation(BaseModel):
    pool_id: str
    protocol: Protocol
    chain: Chain
    token_pair: str
    allocation_pct: float   # 0–1 (fraction of total portfolio)
    expected_apy: float     # Decimal APY
    risk_score: float       # 0–100
    amount_usd: float       # Dollar amount allocated
```

---

### `OptimizedPortfolio`

The full result returned by the optimizer.

```python
class OptimizedPortfolio(BaseModel):
    allocations: list[PortfolioAllocation]
    total_expected_apy: float        # Weighted average APY
    weighted_risk_score: float       # Weighted average risk
    total_investment_usd: float
    generated_at: datetime

    # Properties:
    @property
    def num_positions(self) -> int: ...

    @property
    def diversification_score(self) -> float:  # 0–100, entropy-based
        ...
```

---

### `Config`

Configuration object used by the optimizer and risk engine.

```python
class Config(BaseModel):
    max_risk_score: float = 70.0             # 0–100
    min_tvl_usd: float = 1_000_000.0         # Minimum pool TVL
    max_single_allocation_pct: float = 0.30  # Max fraction per pool (0–1)
    min_positions: int = 2
    max_positions: int = 10
    preferred_chains: list[Chain] = [...]     # Defaults to all chains
    preferred_protocols: list[Protocol] = [...]  # Defaults to all protocols
    stablecoins_only: bool = False
    refresh_interval_seconds: int = 300       # ≥ 30
```

---

## Risk Engine

**Module:** `defi_yield_aggregator.core.risk_engine`

### `RiskEngine`

Scores DeFi pools on multiple risk dimensions.

```python
class RiskEngine:
    def __init__(
        self,
        tvl_weight: float = 0.25,
        age_weight: float = 0.15,
        audit_weight: float = 0.20,
        chain_weight: float = 0.10,
        liquidity_weight: float | None = None,   # Default 0.15
        sc_risk_weight: float | None = None,      # Default 0.15
    ) -> None:
        """Weights must sum to 1.0. Raise ValueError otherwise."""
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `score_pool` | `(pool: PoolInfo) -> RiskScore` | Score a single pool. |
| `score_pools` | `(pools: list[PoolInfo]) -> list[RiskScore]` | Score a list of pools. |
| `filter_by_risk` | `(pools: list[PoolInfo], max_risk: float = 70.0) -> list[tuple[PoolInfo, RiskScore]]` | Return pools that pass the risk threshold. |

#### Risk Dimensions

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| TVL Score | 0.25 | Higher TVL → lower risk. Tiers from >\$10B (score 10) to <\$1M (score 95). |
| Age Score | 0.15 | Years since protocol launch. ≥5 yrs → 10, <2 yrs → 70. |
| Audit Score | 0.20 | Number of security audits. ≥6 → 10, 0 → 90. |
| Chain Diversity | 0.10 | Multi-chain deployments. ≥7 chains → 10, 1 chain → 70. |
| Liquidity | 0.15 | Volume/TVL ratio + absolute volume penalty. |
| Smart Contract Risk | 0.15 | Code complexity, proxy patterns, composability. |

Stable pools get a **−5 bonus**; impermanent‑loss risk adds up to **+20 penalty**.

#### Example

```python
from defi_yield_aggregator.core.risk_engine import RiskEngine
from defi_yield_aggregator.core.models import PoolInfo, Protocol, Chain

pool = PoolInfo(
    protocol=Protocol.AAVE,
    chain=Chain.ETHEREUM,
    pool_id="aave-v3-usdc",
    pool_name="Aave V3 USDC",
    token_pair="USDC",
    apy=0.0385,
    tvl_usd=6_200_000_000,
    is_stable=True,
)

engine = RiskEngine()
score = engine.score_pool(pool)
print(score.risk_level)      # RiskLevel.LOW
print(score.overall_score)   # ~18.0
```

---

## Portfolio Optimizer

**Module:** `defi_yield_aggregator.core.optimizer`

### `PortfolioOptimizer`

```python
class PortfolioOptimizer:
    def __init__(self, config: Config | None = None) -> None: ...

    def optimize(
        self,
        pools: list[PoolInfo],
        investment_usd: float,
    ) -> OptimizedPortfolio:
        """Find optimal allocation across pools.

        Steps:
        1. Filter pools by min_tvl_usd, stablecoins_only, preferred_chains/protocols.
        2. Score and filter by risk (max_risk_score).
        3. Greedy allocation by yield/risk ratio, respecting max_single_allocation_pct.
        4. Distribute remaining budget proportionally.

        Returns empty portfolio if no pools pass filters.
        Raises ValueError if investment_usd ≤ 0.
        """
```

#### Example

```python
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer
from defi_yield_aggregator.core.models import Config

config = Config(
    max_risk_score=50,
    min_tvl_usd=500_000_000,
    max_single_allocation_pct=0.25,
    stablecoins_only=False,
)

optimizer = PortfolioOptimizer(config=config)
portfolio = optimizer.optimize(pools, investment_usd=100_000)

for alloc in portfolio.allocations:
    print(f"{alloc.pool_id}: {alloc.allocation_pct:.1%} (${alloc.amount_usd:,.0f})")

print(f"Expected APY: {portfolio.total_expected_apy:.2%}")
print(f"Risk Score:   {portfolio.weighted_risk_score:.1f}")
```

---

## APY Calculator

**Module:** `defi_yield_aggregator.core.apy_calculator`

All inputs/outputs use **decimal** format (0.05 = 5 %).

| Function | Signature | Description |
|----------|-----------|-------------|
| `apy_to_apr` | `(apy: float, compounding_periods: int = 365) -> float` | Convert APY to APR. |
| `apr_to_apy` | `(apr: float, compounding_periods: int = 365) -> float` | Convert APR to APY. |
| `future_value` | `(principal: float, apy: float, years: float, compounding_periods: int = 365) -> float` | Compound interest FV. |
| `daily_yield` | `(principal: float, apy: float) -> float` | Expected daily earnings. |
| `effective_apy` | `(base_apy: float, reward_apy: float = 0.0, deposit_fee: float = 0.0, withdrawal_fee: float = 0.0, holding_period_days: int = 365) -> float` | Net APY after fees/rewards. |
| `impermanent_loss` | `(price_ratio: float) -> float` | IL for a 50/50 LP position (negative fraction). |

#### Example

```python
from defi_yield_aggregator.core.apy_calculator import future_value, daily_yield, effective_apy

# What will $10,000 be worth in 1 year at 5% APY?
fv = future_value(10_000, 0.05, 1.0)
print(f"1-year value: ${fv:,.2f}")  # $10,512.71

# Daily earnings
print(f"Daily: ${daily_yield(10_000, 0.05):.2f}")  # $1.34

# Effective APY with 2% deposit fee, 0.5% withdrawal, 10% base + 3% reward
eff = effective_apy(0.10, 0.03, deposit_fee=0.02, withdrawal_fee=0.005)
print(f"Effective APY: {eff:.2%}")
```

---

## Impermanent Loss Calculator

**Module:** `defi_yield_aggregator.core.il_calculator`

### Data Classes

```python
@dataclass(frozen=True)
class LPPosition:
    initial_value_usd: float       # Total USD deposited
    token_weights: list[float]     # Must sum to 1.0 (e.g. [0.5, 0.5])
    initial_prices: list[float]    # USD price per token at deposit
    pool_type: str = "constant_product"  # or "weighted"

@dataclass(frozen=True)
class ILResult:
    il_pct: float              # Negative fraction (e.g. -0.05 = 5% loss)
    hold_value_usd: float      # Value if tokens were held
    lp_value_usd: float        # Value of LP position
    price_changes: list[float] # Per-token new/initial ratios

@dataclass(frozen=True)
class NetPnLResult:
    il_result: ILResult
    yield_earned_usd: float
    fees_earned_usd: float
    net_pnl_usd: float
    net_pnl_pct: float
    days_held: int
    annualized_return_pct: float

@dataclass(frozen=True)
class BreakEvenResult:
    price_ratio: float
    is_profitable_at_current: bool
    il_at_break_even: float
    min_yield_apr_needed: float

@dataclass(frozen=True)
class ILTimeSeriesPoint:
    day: int
    price_ratio: float
    il_pct: float
    cumulative_yield_pct: float
    cumulative_fees_pct: float
    net_pnl_pct: float
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `il_constant_product` | `(price_ratio: float) -> float` | IL for x\*y=k 50/50 pool. |
| `il_weighted_pool` | `(price_ratios: list[float], weights: list[float]) -> float` | IL for weighted (Balancer-style) pool. |
| `calculate_il` | `(position: LPPosition, new_prices: list[float]) -> ILResult` | Full IL calculation for any position. |
| `calculate_net_pnl` | `(position: LPPosition, new_prices: list[float], apr_yield: float = 0.0, daily_fee_apr: float = 0.0, days_held: int = 365) -> NetPnLResult` | IL + yield + fees P&L. |
| `calculate_break_even` | `(position: LPPosition, apr_yield: float = 0.0, daily_fee_apr: float = 0.0, days_held: int = 365, current_price_ratio: float \| None = None) -> BreakEvenResult` | Find break-even price ratio. |
| `calculate_il_timeseries` | `(position: LPPosition, price_ratios_over_time: list[float], apr_yield: float = 0.0, daily_fee_apr: float = 0.0) -> list[ILTimeSeriesPoint]` | IL evolution over time. |
| `max_il_for_pool_type` | `(pool_type: str = "constant_product") -> float` | Theoretical max IL. |
| `il_sensitivity_table` | `(ratios: list[float] \| None = None) -> list[tuple[float, float, float]]` | IL at various price ratios. |

#### Example

```python
from defi_yield_aggregator.core.il_calculator import (
    LPPosition, calculate_il, calculate_net_pnl, calculate_break_even,
)

pos = LPPosition(
    initial_value_usd=10_000,
    token_weights=[0.5, 0.5],
    initial_prices=[1_800.0, 1.0],  # ETH/USDC
    pool_type="constant_product",
)

# ETH price doubles to $3,600
result = calculate_il(pos, new_prices=[3_600.0, 1.0])
print(f"IL: {result.il_pct:.2%}")        # -5.72%
print(f"LP value: ${result.lp_value_usd:,.2f}")
print(f"Hold value: ${result.hold_value_usd:,.2f}")

# Full P&L with 10% yield APR and 5% fee APR over 180 days
pnl = calculate_net_pnl(pos, [3_600.0, 1.0], apr_yield=0.10, daily_fee_apr=0.05, days_held=180)
print(f"Net P&L: ${pnl.net_pnl_usd:,.2f}")
print(f"Annualized: {pnl.annualized_return_pct:.2%}")

# Break-even analysis
be = calculate_break_even(pos, apr_yield=0.10, daily_fee_apr=0.05, days_held=180)
print(f"Break-even at price ratio: {be.price_ratio:.4f}")
print(f"Min APR needed: {be.min_yield_apr_needed:.2%}")
```

---

## Protocol Adapters

**Module:** `defi_yield_aggregator.adapters`

### Base Interface

```python
from abc import ABC, abstractmethod
from defi_yield_aggregator.core.models import Chain, PoolInfo, Protocol

class BaseAdapter(ABC):
    def __init__(self, protocol: Protocol, base_url: str = "") -> None: ...

    @abstractmethod
    async def fetch_pools(self, chain: Chain | None = None) -> list[PoolInfo]:
        """Fetch available pools. Optionally filter by chain."""
        ...

    @abstractmethod
    async def fetch_pool_detail(self, pool_id: str) -> PoolInfo | None:
        """Fetch a single pool by ID, or None if not found."""
        ...
```

### Available Adapters

| Adapter | Protocol | Description |
|---------|----------|-------------|
| `AaveAdapter` | AAVE | Aave V3 lending markets |
| `CompoundAdapter` | COMPOUND | Compound V3 lending |
| `UniswapAdapter` | UNISWAP | Uniswap V3 concentrated liquidity |
| `CurveAdapter` | CURVE | Curve Finance stableswap |
| `YearnAdapter` | YEARN | Yearn Finance vaults |
| `LidoAdapter` | LIDO | Lido liquid staking (stETH, stMATIC) |
| `BalancerAdapter` | BALANCER | Balancer V2 weighted & boosted pools |

### Adapter Registry

```python
from defi_yield_aggregator.adapters.protocols import ADAPTERS, get_all_adapters

# Map of Protocol -> Adapter class
ADAPTERS: dict[Protocol, type[BaseAdapter]]

# Convenience: instantiate all adapters
adapters: list[BaseAdapter] = get_all_adapters()

# Override base URLs
adapters = get_all_adapters(base_urls={"aave": "https://custom-aave-api.example.com"})
```

---

## CLI Commands

**Entry point:** `defi-yield` (installed via `pip install -e .`)

### `defi-yield pools`

List available yield farming pools.

```
defi-yield pools [OPTIONS]

Options:
  --chain TEXT        Filter by chain (ethereum, arbitrum, optimism, polygon, base, avalanche)
  --protocol TEXT     Filter by protocol (aave, compound, uniswap, curve, yearn, lido, balancer)
  --stablecoins       Show only stablecoin pools
  --min-tvl FLOAT     Minimum TVL in USD
  --sort [apy|tvl|risk]  Sort by (default: apy)
```

### `defi-yield risk`

Show risk scores for available pools.

```
defi-yield risk [OPTIONS]

Options:
  --chain TEXT        Filter by chain
  --protocol TEXT     Filter by protocol
  --max-risk FLOAT    Maximum risk score 0–100 (default: 70.0)
```

### `defi-yield optimize`

Optimize portfolio allocation for a given investment.

```
defi-yield optimize INVESTMENT [OPTIONS]

Arguments:
  INVESTMENT FLOAT    Investment amount in USD

Options:
  --max-risk FLOAT     Maximum risk score (default: 70.0)
  --min-tvl FLOAT      Minimum pool TVL (default: 1000000)
  --max-alloc FLOAT    Max allocation per pool 0–1 (default: 0.30)
  --stablecoins        Stablecoins only
```

### `defi-yield version`

Show version information.

### Docker Usage

```bash
# Build
make docker-build

# Run CLI commands
make docker-run ARGS='pools --chain ethereum'
make docker-run ARGS='risk --max-risk 50'
make docker-run ARGS='optimize 10000 --stablecoins'

# Run tests in container
make docker-test

# Interactive shell
make docker-shell
```
