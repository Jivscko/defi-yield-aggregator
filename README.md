# DeFi Yield Aggregator

A production-quality Python tool for monitoring yield farming opportunities across multiple DeFi protocols. Maximize your returns while managing risk through intelligent portfolio optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (Click + Rich)                   │
│  pools │ risk │ optimize │ version                          │
├─────────────────────────────────────────────────────────────┤
│                    Portfolio Optimizer                       │
│         (greedy allocation, risk constraints)                │
├──────────────────────┬──────────────────────────────────────┤
│    Risk Engine       │         APY Calculator               │
│  TVL / Age / Audit   │   compound interest, fees, IL        │
│  Chain diversity     │                                      │
├──────────────────────┴──────────────────────────────────────┤
│                   Data Models (Pydantic)                     │
│  PoolInfo │ RiskScore │ Portfolio │ Config                  │
├─────────────────────────────────────────────────────────────┤
│                    Protocol Adapters                         │
│  ┌───────┐ ┌──────────┐ ┌─────────┐ ┌───────┐ ┌─────────┐ │
│  │ Aave  │ │ Compound │ │Uniswap  │ │ Curve │ │  Yearn  │ │
│  └───────┘ └──────────┘ └─────────┘ └───────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-protocol monitoring** — Aave, Compound, Uniswap, Curve, Yearn
- **Risk scoring engine** — TVL-based, protocol age, audit status, chain diversity
- **Portfolio optimizer** — Maximize yield given configurable risk constraints
- **APY calculator** — Compound interest, fee-adjusted returns, impermanent loss
- **Rich CLI** — Beautiful terminal output with tables, panels, and color
- **YAML configuration** — Easy to customize protocols, risk weights, and thresholds
- **Type-safe** — Full type hints and Pydantic data validation
- **Tested** — Comprehensive test suite with 40+ tests

## Installation

```bash
# Clone the repository
git clone https://github.com/Jivscko/defi-yield-aggregator.git
cd defi-yield-aggregator

# Install in development mode
pip install -e ".[dev]"

# Or install from PyPI (when published)
pip install defi-yield-aggregator
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
    tvl: 0.35
    age: 0.20
    audit: 0.25
    chain_diversity: 0.20
  max_score: 70.0

portfolio:
  max_single_allocation_pct: 0.30
  max_positions: 10
  stablecoins_only: false
```

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run all checks
make check
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always do your own research before investing in DeFi protocols. Past yields are not indicative of future performance.
