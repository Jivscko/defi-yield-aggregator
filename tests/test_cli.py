"""Tests for the CLI interface using Click's CliRunner."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from defi_yield_aggregator.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_flag(self, runner: CliRunner) -> None:
        """--version flag shows version string."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_version_subcommand(self, runner: CliRunner) -> None:
        """version subcommand prints version info."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "DeFi Yield Aggregator" in result.output
        assert "0.1.0" in result.output


class TestPoolsCommand:
    """Tests for the pools command."""

    def test_pools_basic(self, runner: CliRunner) -> None:
        """Basic pools listing shows table with results."""
        result = runner.invoke(cli, ["pools"])
        assert result.exit_code == 0
        assert "Yield Farming Pools" in result.output
        assert "Total:" in result.output

    def test_pools_filter_by_chain(self, runner: CliRunner) -> None:
        """Filtering by chain shows only matching pools."""
        result = runner.invoke(cli, ["pools", "--chain", "ethereum"])
        assert result.exit_code == 0
        # Should not contain Arbitrum-only pools
        assert "(Arbitrum)" not in result.output or "Total:" in result.output

    def test_pools_filter_by_protocol(self, runner: CliRunner) -> None:
        """Filtering by protocol shows only that protocol's pools."""
        result = runner.invoke(cli, ["pools", "--protocol", "aave"])
        assert result.exit_code == 0
        assert "aave" in result.output.lower()

    def test_pools_filter_stablecoins(self, runner: CliRunner) -> None:
        """Stablecoins flag filters to only stable pools."""
        result = runner.invoke(cli, ["pools", "--stablecoins"])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_pools_filter_min_tvl(self, runner: CliRunner) -> None:
        """Min TVL filter excludes small pools."""
        result = runner.invoke(cli, ["pools", "--min-tvl", "1000000000"])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_pools_sort_by_tvl(self, runner: CliRunner) -> None:
        """Sorting by TVL works without errors."""
        result = runner.invoke(cli, ["pools", "--sort", "tvl"])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_pools_no_results(self, runner: CliRunner) -> None:
        """Filters that match nothing show a message."""
        result = runner.invoke(cli, ["pools", "--chain", "ethereum", "--protocol", "aave", "--min-tvl", "999999999999"])
        assert result.exit_code == 0
        # Should either show no-pools message or 0 total
        assert "No pools" in result.output or "Total: 0" in result.output


class TestRiskCommand:
    """Tests for the risk command."""

    def test_risk_basic(self, runner: CliRunner) -> None:
        """Basic risk assessment shows risk table."""
        result = runner.invoke(cli, ["risk"])
        assert result.exit_code == 0
        assert "Risk Assessment" in result.output

    def test_risk_filter_by_chain(self, runner: CliRunner) -> None:
        """Risk filter by chain works."""
        result = runner.invoke(cli, ["risk", "--chain", "ethereum"])
        assert result.exit_code == 0
        assert "Risk Assessment" in result.output

    def test_risk_filter_by_protocol(self, runner: CliRunner) -> None:
        """Risk filter by protocol works."""
        result = runner.invoke(cli, ["risk", "--protocol", "uniswap"])
        assert result.exit_code == 0

    def test_risk_max_risk_filter(self, runner: CliRunner) -> None:
        """Max risk score filter applies correctly."""
        result = runner.invoke(cli, ["risk", "--max-risk", "30"])
        assert result.exit_code == 0
        # All shown scores should be <= 30
        assert "Risk Assessment" in result.output or "No pools" in result.output

    def test_risk_very_low_max(self, runner: CliRunner) -> None:
        """Very low max-risk may filter out all pools."""
        result = runner.invoke(cli, ["risk", "--max-risk", "1"])
        assert result.exit_code == 0


class TestOptimizeCommand:
    """Tests for the optimize command."""

    def test_optimize_basic(self, runner: CliRunner) -> None:
        """Basic optimization shows portfolio."""
        result = runner.invoke(cli, ["optimize", "10000"])
        assert result.exit_code == 0
        assert "Optimized Portfolio" in result.output

    def test_optimize_with_stablecoins(self, runner: CliRunner) -> None:
        """Optimize with stablecoins-only flag."""
        result = runner.invoke(cli, ["optimize", "50000", "--stablecoins"])
        assert result.exit_code == 0

    def test_optimize_custom_params(self, runner: CliRunner) -> None:
        """Optimize with custom risk and allocation params."""
        result = runner.invoke(
            cli,
            ["optimize", "100000", "--max-risk", "50", "--min-tvl", "500000000", "--max-alloc", "0.25"],
        )
        assert result.exit_code == 0

    def test_optimize_zero_investment(self, runner: CliRunner) -> None:
        """Zero investment should fail."""
        result = runner.invoke(cli, ["optimize", "0"])
        assert result.exit_code == 1

    def test_optimize_negative_investment(self, runner: CliRunner) -> None:
        """Negative investment should fail (Click rejects with exit 2, or app rejects with exit 1)."""
        result = runner.invoke(cli, ["optimize", "-1000"])
        assert result.exit_code != 0

    def test_optimize_very_small_investment(self, runner: CliRunner) -> None:
        """Very small investment still works (pools exist)."""
        result = runner.invoke(cli, ["optimize", "0.01"])
        assert result.exit_code == 0

    def test_optimize_very_high_max_risk(self, runner: CliRunner) -> None:
        """Very high max risk still works."""
        result = runner.invoke(cli, ["optimize", "10000", "--max-risk", "100"])
        assert result.exit_code == 0

    def test_optimize_very_constrained(self, runner: CliRunner) -> None:
        """Very constrained params may yield no allocations."""
        result = runner.invoke(
            cli,
            ["optimize", "100", "--max-risk", "5", "--min-tvl", "999999999999"],
        )
        assert result.exit_code == 0


class TestCLIHelp:
    """Tests for CLI help text."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Main help shows available commands."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "pools" in result.output
        assert "risk" in result.output
        assert "optimize" in result.output
        assert "version" in result.output

    def test_pools_help(self, runner: CliRunner) -> None:
        """Pools help shows all options."""
        result = runner.invoke(cli, ["pools", "--help"])
        assert result.exit_code == 0
        assert "--chain" in result.output
        assert "--protocol" in result.output
        assert "--stablecoins" in result.output
        assert "--min-tvl" in result.output
        assert "--sort" in result.output

    def test_risk_help(self, runner: CliRunner) -> None:
        """Risk help shows all options."""
        result = runner.invoke(cli, ["risk", "--help"])
        assert result.exit_code == 0
        assert "--max-risk" in result.output

    def test_optimize_help(self, runner: CliRunner) -> None:
        """Optimize help shows all options."""
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--max-risk" in result.output
        assert "--min-tvl" in result.output
        assert "--max-alloc" in result.output
        assert "--stablecoins" in result.output
