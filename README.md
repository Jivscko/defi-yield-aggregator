# DeFi Yield Aggregator

A production-quality Python tool for monitoring yield farming opportunities across multiple DeFi protocols. Maximize your returns while managing risk through intelligent portfolio optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI (Click + Rich)                         │
│   pools │ risk │ optimize │ version                             │
├─────────────────────────────────────────────────────────────────┤
│              Portfolio Optimizer                                 │
│   greedy │ Kelly Criterion │ risk parity                        │
├────────────────────────────┬────────────────────────────────────┤
│   Strategy Simulator       │    Historical APY Tracker          │
│   Monte Carlo backtesting  │    time-series analysis            │
├────────────────────────────┼────────────────────────────────────┤
│   Risk Engine              │    APY Calculator                  │
│   TVL / Age / Audit / SC   │    compound interest, fees, IL     │
│   Chain diversity / Liq    │                                    │
├────────────────────────────┼────────────────────────────────────┤
│   Gas Estimator            │    Alerts & Notifications          │
│   cross-chain cost model   │    webhook / Telegram / console    │
├────────────────────────────┼────────────────────────────────────┤
│   Cache (async TTL+LRU)   │    Rate Limiter                    │
│                            │    token bucket + sliding window   │
├────────────────────────────┴────────────────────────────────────┤
│                   Data Models (Pydantic v2)                      │
│   PoolInfo │ RiskScore │ Portfolio │ Config                     │
├─────────────────────────────────────────────────────────────────┤
│                    Protocol Adapters (11)                        │
│  Aave │ Compound │ Uniswap │ Curve │ Yearn │ Lido              │
│  Balancer │ Convex │ SushiSwap │ Rocket Pool │ Frax             │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **11 protocol adapters** — Aave, Compound, Uniswap, Curve, Yearn, Lido, Balancer, Convex, SushiSwap, Rocket Pool, Frax
- **6-chain support** — Ethereum, Arbitrum, Optimism, Polygon, Base, Avalanche
- **Risk scoring engine** — TVL, protocol age, audit status, chain diversity, liquidity risk, smart contract risk
- **3 optimization strategies** — Greedy allocation, Kelly Criterion, Risk Parity
- **Strategy simulator** — Monte Carlo backtesting with rebalancing, gas costs, slippage
- **Historical APY tracker** — Time-series storage and trend analysis
- **Impermanent loss calculator** — IL estimation for LP positions
- **Gas cost estimator** — Cross-chain gas modeling with net APY calculation
- **Alert system** — Webhook, Telegram, and console notifications with configurable rules
- **Async caching** — TTL cache with LRU eviction and hit-rate statistics
- **Rate limiting** — Token bucket and sliding window strategies
- **Rich CLI** — Beautiful terminal output with tables, panels, and color
- **Docker support** — Multi-stage Dockerfile and docker-compose
- **CI/CD** — GitHub Actions with lint, typecheck, test matrix, and security audit
- **Type-safe** — Full type hints, Pydantic v2 validation, mypy strict mode
- **Well-tested** — Comprehensive test suite with property-based testing (hypothesis)

## Quick Start

```bash
# Clone and install
git clone https://github.com/Jivscko/defi-yield-aggregator.git
cd defi-yield-aggregator
pip install -e ".[dev]"

# List available pools
defi-yield pools

# Run risk assessment
defi-yield risk --max-risk 50

# Optimize a portfolio
defi-yield optimize 100000 --max-risk 60

# Run tests
make test

# Set up pre-commit hooks
make pre-commit-install
```

## Usage

### List Available Pools

```bash
# Show all pools
defi-yield pools

# Filter by chain and protocol
defi-yield pools --chain ethereum --protocol aave

# Stablecoins only, sorted by APY
defi-yield pools --stablecoins --sort apy

# Filter by minimum TVL
defi-yield pools --min-tvl 100000000
```

### Risk Assessment

```bash
# Show risk scores for all pools
defi-yield risk

# Filter by maximum risk score
defi-yield risk --max-risk 50

# Filter by chain
defi-yield risk --chain arbitrum
```

### Portfolio Optimization

```bash
# Optimize a $100,000 portfolio
defi-yield optimize 100000

# Conservative: low risk, stablecoins only
defi-yield optimize 50000 --max-risk 30 --stablecoins

# Aggressive: higher risk tolerance, concentrated
defi-yield optimize 200000 --max-risk 80 --max-alloc 0.40
```

### Example Output

```
📊 Optimized Portfolio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Investment: $100,000.00
  Expected APY: 4.28%
  Risk Score: 28.5
  Positions: 5
  Diversification: 85.2/100

Allocations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Protocol   Chain     Pool          Alloc    Amount       APY    Risk
 Aave       ethereum  USDC          30.0%    $30,000.00   3.85%  18.2
 Curve      ethereum  3Pool         25.0%    $25,000.00   3.20%  22.1
 Yearn      ethereum  yvUSDC        20.0%    $20,000.00   5.20%  25.4
 Compound   ethereum  USDC          15.0%    $15,000.00   3.40%  19.8
 Yearn      arbitrum  yvUSDC        10.0%    $10,000.00   5.80%  32.1
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
risk:
  weights:
    tvl: 0.25
    age: 0.15
    audit: 0.20
    chain_diversity: 0.10
    liquidity: 0.15
    smart_contract_risk: 0.15
  max_score: 70.0

portfolio:
  max_single_allocation_pct: 0.30
  max_positions: 10
  stablecoins_only: false
```

## Programmatic Usage

```python
import asyncio
from defi_yield_aggregator.adapters.protocols import get_all_adapters
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer, AllocationStrategy
from defi_yield_aggregator.core.risk_engine import RiskEngine
from defi_yield_aggregator.core.gas_estimator import GasEstimator, Operation
from defi_yield_aggregator.core.models import Config, Chain

async def main():
    # Fetch pools from all protocols
    adapters = get_all_adapters()
    pools = []
    for adapter in adapters:
        pools.extend(await adapter.fetch_pools())

    # Score risk
    engine = RiskEngine()
    scores = engine.score_pools(pools)

    # Optimize portfolio
    optimizer = PortfolioOptimizer(strategy=AllocationStrategy.KELLY)
    portfolio = optimizer.optimize(pools, investment_usd=100_000)

    # Estimate gas costs
    gas = GasEstimator()
    net = gas.estimate_net_apy(pools[0], investment_usd=10_000, holding_period_days=90)
    print(f"Net APY after gas: {net['net_apy']:.2%}")

asyncio.run(main())
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Format code
make format

# Type check
make typecheck

# Run all checks
make check

# Docker
make docker-build
make docker-run ARGS='pools --chain ethereum'
```

## Project Structure

```
src/defi_yield_aggregator/
├── adapters/
│   ├── base.py          # Abstract adapter interface
│   └── protocols.py     # 11 protocol adapters (Aave → Frax)
├── cli/
│   └── main.py          # Click CLI with Rich output
└── core/
    ├── alerts.py         # Alert engine (webhook/Telegram/console)
    ├── apy_calculator.py # Compound interest & fee math
    ├── cache.py          # Async TTL cache with LRU eviction
    ├── gas_estimator.py  # Cross-chain gas cost modeling
    ├── historical_tracker.py  # APY time-series storage
    ├── il_calculator.py  # Impermanent loss calculator
    ├── models.py         # Pydantic data models
    ├── optimizer.py      # Portfolio optimization (greedy/Kelly/risk parity)
    ├── rate_limiter.py   # Token bucket & sliding window
    ├── risk_engine.py    # Multi-factor risk scoring
    └── strategy_simulator.py  # Monte Carlo backtesting
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always do your own research before investing in DeFi protocols. Past yields are not indicative of future performance.
