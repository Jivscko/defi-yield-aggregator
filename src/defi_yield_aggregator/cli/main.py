"""CLI interface with rich output for the DeFi Yield Aggregator."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from defi_yield_aggregator import __version__
from defi_yield_aggregator.adapters.protocols import get_all_adapters
from defi_yield_aggregator.core.models import Chain, Config, PoolInfo, Protocol, RiskScore
from defi_yield_aggregator.core.optimizer import PortfolioOptimizer
from defi_yield_aggregator.core.risk_engine import RiskEngine

console = Console()


def _run_async(coro: object) -> object:
    """Run an async function, compatible with existing event loops."""
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)  # type: ignore[arg-type]
        finally:
            loop.close()


async def _fetch_all_pools() -> list[PoolInfo]:
    """Fetch pools from all registered adapters."""
    adapters = get_all_adapters()
    all_pools: list[PoolInfo] = []
    for adapter in adapters:
        try:
            pools = await adapter.fetch_pools()
            all_pools.extend(pools)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch from {adapter.protocol.value}: {e}[/yellow]")
    return all_pools


@click.group()
@click.version_option(version=__version__, prog_name="defi-yield")
def cli() -> None:
    """DeFi Yield Aggregator - Monitor yield farming opportunities across protocols."""
    pass


@cli.command()
@click.option("--chain", type=click.Choice([c.value for c in Chain]), help="Filter by chain")
@click.option("--protocol", type=click.Choice([p.value for p in Protocol]), help="Filter by protocol")
@click.option("--stablecoins", is_flag=True, help="Show only stablecoin pools")
@click.option("--min-tvl", type=float, default=0, help="Minimum TVL in USD")
@click.option("--sort", type=click.Choice(["apy", "tvl", "risk"]), default="apy", help="Sort by")
def pools(chain: Optional[str], protocol: Optional[str], stablecoins: bool, min_tvl: float, sort: str) -> None:
    """List available yield farming pools."""
    all_pools = _run_async(_fetch_all_pools())

    # Apply filters
    filtered = all_pools
    if chain:
        filtered = [p for p in filtered if p.chain.value == chain]
    if protocol:
        filtered = [p for p in filtered if p.protocol.value == protocol]
    if stablecoins:
        filtered = [p for p in filtered if p.is_stable]
    if min_tvl > 0:
        filtered = [p for p in filtered if p.tvl_usd >= min_tvl]

    # Sort
    sort_keys = {
        "apy": lambda p: p.apy,
        "tvl": lambda p: p.tvl_usd,
    }
    if sort in sort_keys:
        filtered.sort(key=sort_keys[sort], reverse=True)

    if not filtered:
        console.print("[yellow]No pools match your filters.[/yellow]")
        return

    table = Table(title="🏊 Yield Farming Pools", show_lines=True)
    table.add_column("Protocol", style="cyan", width=12)
    table.add_column("Chain", style="magenta", width=10)
    table.add_column("Pool", style="white", width=35)
    table.add_column("APY", justify="right", style="green", width=10)
    table.add_column("TVL", justify="right", style="blue", width=14)
    table.add_column("Stable", justify="center", width=8)
    table.add_column("IL Risk", justify="right", width=8)

    for pool in filtered:
        apy_str = f"{pool.apy * 100:.2f}%"
        tvl_str = f"${pool.tvl_usd / 1e6:,.1f}M" if pool.tvl_usd >= 1e6 else f"${pool.tvl_usd:,.0f}"
        stable_str = "✅" if pool.is_stable else "❌"
        il_str = f"{pool.impermanent_loss_risk * 100:.1f}%"

        table.add_row(
            pool.protocol.value,
            pool.chain.value,
            pool.pool_name,
            apy_str,
            tvl_str,
            stable_str,
            il_str,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(filtered)} pools[/dim]")


@cli.command()
@click.option("--chain", type=click.Choice([c.value for c in Chain]), help="Filter by chain")
@click.option("--protocol", type=click.Choice([p.value for p in Protocol]), help="Filter by protocol")
@click.option("--max-risk", type=float, default=70.0, help="Maximum risk score (0-100)")
def risk(chain: Optional[str], protocol: Optional[str], max_risk: float) -> None:
    """Show risk scores for available pools."""
    all_pools = _run_async(_fetch_all_pools())

    filtered = all_pools
    if chain:
        filtered = [p for p in filtered if p.chain.value == chain]
    if protocol:
        filtered = [p for p in filtered if p.protocol.value == protocol]

    engine = RiskEngine()
    scores = engine.score_pools(filtered)

    # Filter by max risk
    scores = [s for s in scores if s.overall_score <= max_risk]
    scores.sort(key=lambda s: s.overall_score)

    if not scores:
        console.print("[yellow]No pools match your risk criteria.[/yellow]")
        return

    risk_colors = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }

    table = Table(title="⚠️  Risk Assessment", show_lines=True)
    table.add_column("Pool ID", style="white", width=30)
    table.add_column("Protocol", style="cyan", width=10)
    table.add_column("Overall", justify="right", width=10)
    table.add_column("Level", justify="center", width=10)
    table.add_column("TVL", justify="right", width=8)
    table.add_column("Age", justify="right", width=8)
    table.add_column("Audit", justify="right", width=8)
    table.add_column("Chains", justify="right", width=8)

    for score in scores:
        color = risk_colors.get(score.risk_level.value, "white")
        table.add_row(
            score.pool_id,
            score.protocol.value,
            f"[{color}]{score.overall_score:.1f}[/{color}]",
            f"[{color}]{score.risk_level.value.upper()}[/{color}]",
            f"{score.tvl_score:.0f}",
            f"{score.age_score:.0f}",
            f"{score.audit_score:.0f}",
            f"{score.chain_diversity_score:.0f}",
        )

    console.print(table)


@cli.command()
@click.argument("investment", type=float)
@click.option("--max-risk", type=float, default=70.0, help="Maximum risk score (0-100)")
@click.option("--min-tvl", type=float, default=1_000_000, help="Minimum pool TVL")
@click.option("--max-alloc", type=float, default=0.30, help="Max allocation per pool (0-1)")
@click.option("--stablecoins", is_flag=True, help="Stablecoins only")
def optimize(investment: float, max_risk: float, min_tvl: float, max_alloc: float, stablecoins: bool) -> None:
    """Optimize portfolio allocation for a given investment amount."""
    if investment <= 0:
        console.print("[red]Investment must be positive.[/red]")
        sys.exit(1)

    config = Config(
        max_risk_score=max_risk,
        min_tvl_usd=min_tvl,
        max_single_allocation_pct=max_alloc,
        stablecoins_only=stablecoins,
    )

    all_pools = _run_async(_fetch_all_pools())
    optimizer = PortfolioOptimizer(config=config)
    portfolio = optimizer.optimize(all_pools, investment)

    if not portfolio.allocations:
        console.print("[yellow]No eligible pools found with current constraints.[/yellow]")
        return

    # Header panel
    header = Text()
    header.append(f"Investment: ${investment:,.2f}\n", style="bold")
    header.append(f"Expected APY: {portfolio.total_expected_apy * 100:.2f}%\n", style="green")
    header.append(f"Risk Score: {portfolio.weighted_risk_score:.1f}\n", style="yellow")
    header.append(f"Positions: {portfolio.num_positions}\n", style="cyan")
    header.append(f"Diversification: {portfolio.diversification_score:.1f}/100", style="magenta")
    console.print(Panel(header, title="📊 Optimized Portfolio", border_style="blue"))

    # Allocations table
    table = Table(title="Allocations", show_lines=True)
    table.add_column("Protocol", style="cyan", width=12)
    table.add_column("Chain", style="magenta", width=10)
    table.add_column("Pool", style="white", width=30)
    table.add_column("Allocation", justify="right", style="bold", width=12)
    table.add_column("Amount", justify="right", style="green", width=14)
    table.add_column("APY", justify="right", style="green", width=10)
    table.add_column("Risk", justify="right", width=8)

    for alloc in portfolio.allocations:
        table.add_row(
            alloc.protocol.value,
            alloc.chain.value,
            alloc.token_pair,
            f"{alloc.allocation_pct * 100:.1f}%",
            f"${alloc.amount_usd:,.2f}",
            f"{alloc.expected_apy * 100:.2f}%",
            f"{alloc.risk_score:.1f}",
        )

    console.print(table)

    # Projections
    from defi_yield_aggregator.core.apy_calculator import future_value, daily_yield

    daily = daily_yield(investment, portfolio.total_expected_apy)
    fv_1y = future_value(investment, portfolio.total_expected_apy, 1.0)
    fv_5y = future_value(investment, portfolio.total_expected_apy, 5.0)

    proj = Table(title="📈 Projections", show_lines=True)
    proj.add_column("Timeframe", style="white", width=15)
    proj.add_column("Value", justify="right", style="green", width=15)
    proj.add_column("Profit", justify="right", style="cyan", width=15)
    proj.add_row("Daily", f"${daily:,.2f}", f"${daily:,.2f}")
    proj.add_row("1 Year", f"${fv_1y:,.2f}", f"${fv_1y - investment:,.2f}")
    proj.add_row("5 Years", f"${fv_5y:,.2f}", f"${fv_5y - investment:,.2f}")
    console.print(proj)


@cli.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold]DeFi Yield Aggregator[/bold] v{__version__}")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
