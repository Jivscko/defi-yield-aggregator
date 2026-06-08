"""Tests for the Alert and Notification system."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from defi_yield_aggregator.core.alerts import (
    Alert,
    AlertCondition,
    AlertEngine,
    AlertHistory,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertSummary,
    BaseChannel,
    ConsoleChannel,
    TelegramChannel,
    WebhookChannel,
)
from defi_yield_aggregator.core.models import (
    Chain,
    OptimizedPortfolio,
    PoolInfo,
    PortfolioAllocation,
    Protocol,
    RiskLevel,
    RiskScore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pools() -> list[PoolInfo]:
    """Sample pool data for testing."""
    return [
        PoolInfo(
            protocol=Protocol.AAVE,
            chain=Chain.ETHEREUM,
            pool_id="aave-v3-usdc",
            pool_name="Aave V3 USDC",
            token_pair="USDC",
            apy=0.038,
            tvl_usd=6_200_000_000,
            is_stable=True,
            impermanent_loss_risk=0.0,
        ),
        PoolInfo(
            protocol=Protocol.UNISWAP,
            chain=Chain.ETHEREUM,
            pool_id="uni-v3-eth-usdc",
            pool_name="Uniswap V3 ETH/USDC",
            token_pair="ETH/USDC",
            apy=0.068,
            tvl_usd=2_400_000_000,
            is_stable=False,
            impermanent_loss_risk=0.35,
        ),
        PoolInfo(
            protocol=Protocol.CURVE,
            chain=Chain.ETHEREUM,
            pool_id="curve-3pool",
            pool_name="Curve 3Pool",
            token_pair="DAI/USDC/USDT",
            apy=0.032,
            tvl_usd=1_500_000_000,
            is_stable=True,
            impermanent_loss_risk=0.005,
        ),
    ]


@pytest.fixture
def sample_risks() -> dict[str, RiskScore]:
    """Sample risk scores for testing."""
    return {
        "aave-v3-usdc": RiskScore(
            pool_id="aave-v3-usdc",
            protocol=Protocol.AAVE,
            overall_score=18.0,
            risk_level=RiskLevel.LOW,
        ),
        "uni-v3-eth-usdc": RiskScore(
            pool_id="uni-v3-eth-usdc",
            protocol=Protocol.UNISWAP,
            overall_score=45.0,
            risk_level=RiskLevel.MEDIUM,
        ),
        "curve-3pool": RiskScore(
            pool_id="curve-3pool",
            protocol=Protocol.CURVE,
            overall_score=75.0,
            risk_level=RiskLevel.HIGH,
        ),
    }


@pytest.fixture
def console_channel() -> ConsoleChannel:
    return ConsoleChannel()


@pytest.fixture
def webhook_channel() -> WebhookChannel:
    return WebhookChannel("https://hooks.example.com/defi-alerts")


@pytest.fixture
def telegram_channel() -> TelegramChannel:
    return TelegramChannel("bot123456:ABC-DEF", "-100123456")


@pytest.fixture
def engine() -> AlertEngine:
    return AlertEngine()


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------


class TestAlertModel:
    def test_default_fingerprint(self) -> None:
        """Alert generates a fingerprint from rule:pool:severity."""
        alert = Alert(
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            pool_id="pool-1",
            message="test",
        )
        assert len(alert.fingerprint) == 16
        assert alert.status == AlertStatus.PENDING

    def test_same_fingerprint_for_same_fields(self) -> None:
        """Same rule+pool+severity = same fingerprint."""
        a1 = Alert(rule_name="r", severity=AlertSeverity.INFO, pool_id="p", message="m1")
        a2 = Alert(rule_name="r", severity=AlertSeverity.INFO, pool_id="p", message="m2")
        assert a1.fingerprint == a2.fingerprint

    def test_different_fingerprint_different_severity(self) -> None:
        a1 = Alert(rule_name="r", severity=AlertSeverity.INFO, pool_id="p", message="m")
        a2 = Alert(rule_name="r", severity=AlertSeverity.CRITICAL, pool_id="p", message="m")
        assert a1.fingerprint != a2.fingerprint

    def test_custom_fingerprint(self) -> None:
        alert = Alert(
            rule_name="r",
            severity=AlertSeverity.INFO,
            pool_id="p",
            message="m",
            fingerprint="custom-fp",
        )
        assert alert.fingerprint == "custom-fp"


# ---------------------------------------------------------------------------
# Channel tests
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    def test_valid_url(self, webhook_channel: WebhookChannel) -> None:
        assert "webhook" in webhook_channel.name

    def test_invalid_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid webhook URL"):
            WebhookChannel("ftp://not-a-url")

    def test_send_stores_payload(self, webhook_channel: WebhookChannel) -> None:
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            pool_id="pool-1",
            message="APY dropped",
            details={"apy": 0.02},
        )
        result = webhook_channel.send(alert)
        assert result is True
        assert len(webhook_channel.sent_payloads) == 1
        payload = webhook_channel.sent_payloads[0]
        assert payload["rule"] == "test"
        assert payload["severity"] == "warning"
        assert payload["pool_id"] == "pool-1"
        assert "triggered_at" in payload

    def test_send_includes_headers(self) -> None:
        channel = WebhookChannel(
            "https://example.com/hook",
            headers={"Authorization": "Bearer token123"},
        )
        alert = Alert(
            rule_name="r",
            severity=AlertSeverity.INFO,
            pool_id="p",
            message="m",
        )
        assert channel.send(alert) is True


class TestTelegramChannel:
    def test_missing_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="required"):
            TelegramChannel("", "-100123")

    def test_missing_chat_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="required"):
            TelegramChannel("bot-token", "")

    def test_send_formats_message(self, telegram_channel: TelegramChannel) -> None:
        alert = Alert(
            rule_name="risk-critical",
            severity=AlertSeverity.CRITICAL,
            pool_id="uni-v3-eth",
            message="Risk score at 78/100",
            details={"risk_level": "high"},
        )
        result = telegram_channel.send(alert)
        assert result is True
        assert len(telegram_channel.sent_messages) == 1
        msg = telegram_channel.sent_messages[0]
        assert "CRIT" in msg
        assert "risk-critical" in msg
        assert "uni-v3-eth" in msg

    def test_info_severity_label(self, telegram_channel: TelegramChannel) -> None:
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.INFO,
            pool_id="p",
            message="m",
        )
        telegram_channel.send(alert)
        assert "INFO" in telegram_channel.sent_messages[0]

    def test_name_masks_token(self, telegram_channel: TelegramChannel) -> None:
        name = telegram_channel.name
        assert "bot123" in name
        assert "ABC-DEF" not in name  # Should be masked


class TestConsoleChannel:
    def test_send_logs_warning(self, console_channel: ConsoleChannel, caplog: _CaplogType) -> None:
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            pool_id="pool-1",
            message="Warning message",
        )
        with caplog.at_level(logging.WARNING):
            result = console_channel.send(alert)
        assert result is True
        assert "Warning message" in caplog.text

    def test_send_logs_critical(self, console_channel: ConsoleChannel, caplog: _CaplogType) -> None:
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            pool_id="pool-1",
            message="Critical message",
        )
        with caplog.at_level(logging.CRITICAL):
            result = console_channel.send(alert)
        assert result is True

    def test_name_is_console(self, console_channel: ConsoleChannel) -> None:
        assert console_channel.name == "console"


# ---------------------------------------------------------------------------
# AlertEngine - rule management
# ---------------------------------------------------------------------------


class TestAlertEngineRules:
    def test_add_rule(self, engine: AlertEngine) -> None:
        rule = AlertRule(
            name="test-rule",
            condition=AlertCondition.APY_DROP,
            threshold=2.0,
            severity=AlertSeverity.WARNING,
        )
        engine.add_rule(rule)
        assert len(engine.rules) == 1
        assert engine.get_rule("test-rule") is rule

    def test_duplicate_rule_rejected(self, engine: AlertEngine) -> None:
        rule = AlertRule(
            name="dup",
            condition=AlertCondition.APY_DROP,
            threshold=1.0,
            severity=AlertSeverity.INFO,
        )
        engine.add_rule(rule)
        with pytest.raises(ValueError, match="already exists"):
            engine.add_rule(rule)

    def test_remove_rule(self, engine: AlertEngine) -> None:
        rule = AlertRule(
            name="removable",
            condition=AlertCondition.APY_DROP,
            threshold=1.0,
            severity=AlertSeverity.INFO,
        )
        engine.add_rule(rule)
        assert engine.remove_rule("removable") is True
        assert engine.get_rule("removable") is None
        assert engine.remove_rule("nonexistent") is False

    def test_disabled_rule_skipped(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Disabled rules should not trigger alerts."""
        engine.add_rule(
            AlertRule(
                name="disabled",
                condition=AlertCondition.APY_DROP,
                threshold=99.0,  # Would trigger for all pools
                severity=AlertSeverity.WARNING,
                enabled=False,
            )
        )
        summary = engine.evaluate_pools(sample_pools)
        assert summary.total_alerts_triggered == 0


# ---------------------------------------------------------------------------
# AlertEngine - APY drop
# ---------------------------------------------------------------------------


class TestAlertEngineAPYDrop:
    def test_apy_below_threshold_triggers(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Pools with APY below threshold should trigger."""
        engine.add_rule(
            AlertRule(
                name="apy-low",
                condition=AlertCondition.APY_DROP,
                threshold=5.0,  # 5% — pools at 3.8%, 3.2% will trigger
                severity=AlertSeverity.WARNING,
                channels=[ConsoleChannel()],
            )
        )
        summary = engine.evaluate_pools(sample_pools)
        # aave-v3-usdc (3.8%) and curve-3pool (3.2%) are below 5%
        assert summary.total_alerts_triggered == 2
        assert summary.alerts_sent == 2

    def test_apy_above_threshold_no_trigger(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Pools above threshold should not trigger."""
        engine.add_rule(
            AlertRule(
                name="apy-high-threshold",
                condition=AlertCondition.APY_DROP,
                threshold=1.0,  # All pools above 1%
                severity=AlertSeverity.WARNING,
            )
        )
        summary = engine.evaluate_pools(sample_pools)
        assert summary.total_alerts_triggered == 0


# ---------------------------------------------------------------------------
# AlertEngine - Risk score
# ---------------------------------------------------------------------------


class TestAlertEngineRiskScore:
    def test_high_risk_triggers(
        self,
        engine: AlertEngine,
        sample_pools: list[PoolInfo],
        sample_risks: dict[str, RiskScore],
    ) -> None:
        engine.add_rule(
            AlertRule(
                name="risk-alert",
                condition=AlertCondition.RISK_SCORE_ABOVE,
                threshold=60.0,
                severity=AlertSeverity.CRITICAL,
                channels=[ConsoleChannel()],
            )
        )
        summary = engine.evaluate_pools(sample_pools, sample_risks)
        # Only curve-3pool (75.0) is above 60
        assert summary.total_alerts_triggered == 1
        assert summary.alerts[0].pool_id == "curve-3pool"
        assert summary.alerts[0].severity == AlertSeverity.CRITICAL

    def test_no_risk_scores_no_trigger(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Without risk scores, risk rules should not trigger."""
        engine.add_rule(
            AlertRule(
                name="risk-alert",
                condition=AlertCondition.RISK_SCORE_ABOVE,
                threshold=10.0,
                severity=AlertSeverity.WARNING,
            )
        )
        summary = engine.evaluate_pools(sample_pools, risk_scores=None)
        assert summary.total_alerts_triggered == 0


# ---------------------------------------------------------------------------
# AlertEngine - TVL drop
# ---------------------------------------------------------------------------


class TestAlertEngineTVLDrop:
    def test_tvl_drop_triggers(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """After baselines are set, a TVL drop should trigger."""
        # Set baselines with higher TVL
        high_tvl_pools = []
        for pool in sample_pools:
            high_tvl_pools.append(
                PoolInfo(
                    protocol=pool.protocol,
                    chain=pool.chain,
                    pool_id=pool.pool_id,
                    pool_name=pool.pool_name,
                    token_pair=pool.token_pair,
                    apy=pool.apy,
                    tvl_usd=pool.tvl_usd * 2,  # Double TVL
                    is_stable=pool.is_stable,
                    impermanent_loss_risk=pool.impermanent_loss_risk,
                )
            )

        engine.update_baselines(high_tvl_pools)

        engine.add_rule(
            AlertRule(
                name="tvl-drop",
                condition=AlertCondition.TVL_DROP,
                threshold=30.0,  # 30% drop
                severity=AlertSeverity.CRITICAL,
                channels=[ConsoleChannel()],
            )
        )

        # Now evaluate with lower TVL (50% drop)
        summary = engine.evaluate_pools(sample_pools)
        assert summary.total_alerts_triggered == 3  # All pools dropped 50%


# ---------------------------------------------------------------------------
# AlertEngine - APY change
# ---------------------------------------------------------------------------


class TestAlertEngineAPYChange:
    def test_apy_change_triggers(self, engine: AlertEngine) -> None:
        """Large APY changes should trigger alerts."""
        initial_pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.10,  # 10% APY
                tvl_usd=1_000_000_000,
            ),
        ]
        engine.update_baselines(initial_pools)

        engine.add_rule(
            AlertRule(
                name="apy-volatile",
                condition=AlertCondition.APY_CHANGE,
                threshold=30.0,  # 30% change
                severity=AlertSeverity.WARNING,
                channels=[ConsoleChannel()],
            )
        )

        # APY dropped from 10% to 5% = 50% change
        changed_pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.05,
                tvl_usd=1_000_000_000,
            ),
        ]

        summary = engine.evaluate_pools(changed_pools)
        assert summary.total_alerts_triggered == 1
        assert summary.alerts[0].details["previous_apy"] == 10.0
        assert summary.alerts[0].details["current_apy"] == 5.0


# ---------------------------------------------------------------------------
# AlertEngine - Pool degradation
# ---------------------------------------------------------------------------


class TestAlertEnginePoolDegraded:
    def test_degraded_pool_triggers(
        self,
        engine: AlertEngine,
        sample_pools: list[PoolInfo],
        sample_risks: dict[str, RiskScore],
    ) -> None:
        """Pool with risk up AND APY down should trigger degradation alert."""
        # Set baselines
        engine.update_baselines(sample_pools, sample_risks)

        engine.add_rule(
            AlertRule(
                name="degraded",
                condition=AlertCondition.POOL_DEGRADED,
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                channels=[ConsoleChannel()],
            )
        )

        # Degrade aave: APY dropped, risk increased
        degraded_pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-v3-usdc",
                pool_name="Aave V3 USDC",
                token_pair="USDC",
                apy=0.015,  # Was 3.8%, now 1.5% (60% drop)
                tvl_usd=6_200_000_000,
                is_stable=True,
            ),
            sample_pools[1],
            sample_pools[2],
        ]

        degraded_risks = dict(sample_risks)
        degraded_risks["aave-v3-usdc"] = RiskScore(
            pool_id="aave-v3-usdc",
            protocol=Protocol.AAVE,
            overall_score=35.0,  # Was 18, now 35 (change > 10)
            risk_level=RiskLevel.MEDIUM,
        )

        summary = engine.evaluate_pools(degraded_pools, degraded_risks)
        degraded_alerts = [
            a for a in summary.alerts if a.rule_name == "degraded"
        ]
        assert len(degraded_alerts) == 1
        assert degraded_alerts[0].pool_id == "aave-v3-usdc"


# ---------------------------------------------------------------------------
# AlertEngine - Deduplication and cooldown
# ---------------------------------------------------------------------------


class TestAlertEngineDedup:
    def test_duplicate_suppressed_within_window(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Same alert should be suppressed within dedup window."""
        engine.add_rule(
            AlertRule(
                name="dedup-test",
                condition=AlertCondition.APY_DROP,
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                dedup_window_minutes=60,
                channels=[ConsoleChannel()],
            )
        )

        # First evaluation triggers
        summary1 = engine.evaluate_pools(sample_pools)
        assert summary1.alerts_sent > 0

        # Second evaluation immediately — should be suppressed
        summary2 = engine.evaluate_pools(sample_pools)
        assert summary2.alerts_suppressed > 0
        assert summary2.alerts_sent == 0

    def test_alert_after_window_expires(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        """Alert should re-trigger after dedup window expires."""
        rule = AlertRule(
            name="window-test",
            condition=AlertCondition.APY_DROP,
            threshold=5.0,
            severity=AlertSeverity.WARNING,
            dedup_window_minutes=1,  # 1 minute window
            cooldown_minutes=0,  # No cooldown
            channels=[ConsoleChannel()],
        )
        engine.add_rule(rule)

        # First trigger
        engine.evaluate_pools(sample_pools)

        # Simulate time passing by manipulating history
        for fp in engine._history:
            engine._history[fp].last_sent_at = datetime.now(timezone.utc) - timedelta(
                minutes=5
            )

        # Should trigger again
        summary = engine.evaluate_pools(sample_pools)
        assert summary.alerts_sent > 0

    def test_clear_history_resets_dedup(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        engine.add_rule(
            AlertRule(
                name="clear-test",
                condition=AlertCondition.APY_DROP,
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                channels=[ConsoleChannel()],
            )
        )

        engine.evaluate_pools(sample_pools)
        assert len(engine._history) > 0

        engine.clear_history()
        assert len(engine._history) == 0

        # Should trigger again
        summary = engine.evaluate_pools(sample_pools)
        assert summary.alerts_sent > 0


# ---------------------------------------------------------------------------
# AlertEngine - Portfolio drift
# ---------------------------------------------------------------------------


class TestPortfolioDrift:
    def test_drift_within_threshold_no_alert(self) -> None:
        """Small drift should not trigger alerts."""
        portfolio = OptimizedPortfolio(
            allocations=[
                PortfolioAllocation(
                    pool_id="pool-a",
                    protocol=Protocol.AAVE,
                    chain=Chain.ETHEREUM,
                    token_pair="USDC",
                    allocation_pct=0.5,
                    expected_apy=0.04,
                    risk_score=20.0,
                ),
            ],
            total_expected_apy=0.04,
            weighted_risk_score=20.0,
            total_investment_usd=10000.0,
        )

        engine = AlertEngine()
        # Current allocation is 48% vs target 50% — 2% drift
        summary = engine.evaluate_portfolio_drift(
            portfolio, {"pool-a": 0.48}, drift_threshold_pct=5.0
        )
        assert summary.total_alerts_triggered == 0

    def test_drift_exceeds_threshold_triggers(self) -> None:
        """Significant drift should trigger alerts."""
        portfolio = OptimizedPortfolio(
            allocations=[
                PortfolioAllocation(
                    pool_id="pool-a",
                    protocol=Protocol.AAVE,
                    chain=Chain.ETHEREUM,
                    token_pair="USDC",
                    allocation_pct=0.6,
                    expected_apy=0.04,
                    risk_score=20.0,
                ),
                PortfolioAllocation(
                    pool_id="pool-b",
                    protocol=Protocol.UNISWAP,
                    chain=Chain.ETHEREUM,
                    token_pair="ETH/USDC",
                    allocation_pct=0.4,
                    expected_apy=0.07,
                    risk_score=45.0,
                ),
            ],
            total_expected_apy=0.052,
            weighted_risk_score=30.0,
            total_investment_usd=10000.0,
        )

        engine = AlertEngine()
        # pool-a drifted from 60% to 40% = 20% drift
        # pool-b drifted from 40% to 60% = 20% drift
        summary = engine.evaluate_portfolio_drift(
            portfolio,
            {"pool-a": 0.40, "pool-b": 0.60},
            drift_threshold_pct=5.0,
        )
        assert summary.total_alerts_triggered == 2
        # Both should be CRITICAL (>2x threshold)
        assert all(a.severity == AlertSeverity.CRITICAL for a in summary.alerts)

    def test_missing_pool_triggers_drift(self) -> None:
        """Pool in target but not in current = 100% drift."""
        portfolio = OptimizedPortfolio(
            allocations=[
                PortfolioAllocation(
                    pool_id="missing-pool",
                    protocol=Protocol.CURVE,
                    chain=Chain.ETHEREUM,
                    token_pair="DAI/USDC",
                    allocation_pct=0.5,
                    expected_apy=0.03,
                    risk_score=15.0,
                ),
            ],
            total_expected_apy=0.03,
            weighted_risk_score=15.0,
            total_investment_usd=10000.0,
        )

        engine = AlertEngine()
        summary = engine.evaluate_portfolio_drift(
            portfolio, {}, drift_threshold_pct=5.0
        )
        assert summary.total_alerts_triggered == 1
        assert summary.alerts[0].severity == AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# AlertEngine - with_defaults factory
# ---------------------------------------------------------------------------


class TestWithDefaults:
    def test_creates_default_rules(self) -> None:
        engine = AlertEngine.with_defaults()
        assert len(engine.rules) == 5
        rule_names = {r.name for r in engine.rules}
        assert "apy-drop-low" in rule_names
        assert "risk-critical" in rule_names
        assert "tvl-crash" in rule_names

    def test_with_webhook(self) -> None:
        engine = AlertEngine.with_defaults(
            webhook_url="https://hooks.example.com/test"
        )
        # Each rule should have console + webhook channels
        for rule in engine.rules:
            assert len(rule.channels) == 2
            assert any(isinstance(c, WebhookChannel) for c in rule.channels)

    def test_with_telegram(self) -> None:
        engine = AlertEngine.with_defaults(
            telegram_token="bot-token",
            telegram_chat_id="-100123",
        )
        for rule in engine.rules:
            assert any(isinstance(c, TelegramChannel) for c in rule.channels)

    def test_with_all_channels(self) -> None:
        engine = AlertEngine.with_defaults(
            webhook_url="https://hooks.example.com/test",
            telegram_token="bot-token",
            telegram_chat_id="-100123",
        )
        for rule in engine.rules:
            assert len(rule.channels) == 3

    def test_default_rules_evaluate(
        self, sample_pools: list[PoolInfo], sample_risks: dict[str, RiskScore]
    ) -> None:
        """Default rules should work out of the box."""
        engine = AlertEngine.with_defaults()
        # Set baselines then degrade
        engine.update_baselines(sample_pools, sample_risks)

        # Degrade all pools
        bad_pools = [
            PoolInfo(
                protocol=p.protocol,
                chain=p.chain,
                pool_id=p.pool_id,
                pool_name=p.pool_name,
                token_pair=p.token_pair,
                apy=0.005,  # 0.5% APY
                tvl_usd=p.tvl_usd * 0.5,  # 50% TVL drop
                is_stable=p.is_stable,
                impermanent_loss_risk=p.impermanent_loss_risk,
            )
            for p in sample_pools
        ]

        summary = engine.evaluate_pools(bad_pools, sample_risks)
        assert summary.total_alerts_triggered > 0


# ---------------------------------------------------------------------------
# AlertSummary
# ---------------------------------------------------------------------------


class TestAlertSummary:
    def test_summary_counts(self) -> None:
        alerts = [
            Alert(rule_name="r", severity=AlertSeverity.INFO, pool_id="p", message="m"),
            Alert(
                rule_name="r",
                severity=AlertSeverity.WARNING,
                pool_id="p",
                message="m",
                status=AlertStatus.SUPPRESSED,
            ),
        ]
        summary = AlertSummary(
            total_alerts_triggered=2,
            alerts_sent=1,
            alerts_suppressed=1,
            alerts_failed=0,
            alerts=alerts,
        )
        assert summary.total_alerts_triggered == 2
        assert summary.alerts_sent == 1
        assert summary.alerts_suppressed == 1


# ---------------------------------------------------------------------------
# Custom message templates
# ---------------------------------------------------------------------------


class TestCustomTemplates:
    def test_custom_template_used(
        self, engine: AlertEngine, sample_pools: list[PoolInfo]
    ) -> None:
        engine.add_rule(
            AlertRule(
                name="custom-msg",
                condition=AlertCondition.APY_DROP,
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                message_template="ALERT: {pool_id} ({protocol}/{chain}) APY={value:.2f}%",
                channels=[ConsoleChannel()],
            )
        )
        summary = engine.evaluate_pools(sample_pools)
        assert summary.total_alerts_triggered > 0
        assert "ALERT:" in summary.alerts[0].message
        assert "APY=" in summary.alerts[0].message


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self) -> None:
        """End-to-end: create engine, add rules, evaluate, check results."""
        # Track delivered messages
        delivered: list[dict] = []

        class TrackingChannel(BaseChannel):
            @property
            def name(self) -> str:
                return "tracking"

            def send(self, alert: Alert) -> bool:
                delivered.append(
                    {"rule": alert.rule_name, "pool": alert.pool_id, "sev": alert.severity.value}
                )
                return True

        channel = TrackingChannel()

        engine = AlertEngine()
        engine.add_rule(
            AlertRule(
                name="low-apy",
                condition=AlertCondition.APY_DROP,
                threshold=4.0,
                severity=AlertSeverity.WARNING,
                channels=[channel],
            )
        )
        engine.add_rule(
            AlertRule(
                name="high-risk",
                condition=AlertCondition.RISK_SCORE_ABOVE,
                threshold=50.0,
                severity=AlertSeverity.CRITICAL,
                channels=[channel],
            )
        )

        pools = [
            PoolInfo(
                protocol=Protocol.AAVE,
                chain=Chain.ETHEREUM,
                pool_id="aave-usdc",
                pool_name="Aave USDC",
                token_pair="USDC",
                apy=0.02,  # 2% < 4% threshold
                tvl_usd=1_000_000_000,
            ),
            PoolInfo(
                protocol=Protocol.UNISWAP,
                chain=Chain.ETHEREUM,
                pool_id="uni-eth",
                pool_name="Uni ETH",
                token_pair="ETH",
                apy=0.08,  # 8% > 4% threshold
                tvl_usd=500_000_000,
            ),
        ]

        risks = {
            "uni-eth": RiskScore(
                pool_id="uni-eth",
                protocol=Protocol.UNISWAP,
                overall_score=65.0,
                risk_level=RiskLevel.HIGH,
            ),
        }

        summary = engine.evaluate_pools(pools, risks)

        # aave-usdc triggers low-apy (2% < 4%)
        # uni-eth triggers high-risk (65 > 50)
        assert summary.total_alerts_triggered == 2
        assert summary.alerts_sent == 2
        assert len(delivered) == 2

        # Verify content
        rules_delivered = {d["rule"] for d in delivered}
        assert "low-apy" in rules_delivered
        assert "high-risk" in rules_delivered

    def test_multiple_channels_all_receive(
        self, sample_pools: list[PoolInfo]
    ) -> None:
        """All channels should receive the alert."""
        channel_a = ConsoleChannel()
        channel_b = WebhookChannel("https://example.com/hook")

        engine = AlertEngine()
        engine.add_rule(
            AlertRule(
                name="multi-channel",
                condition=AlertCondition.APY_DROP,
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                channels=[channel_a, channel_b],
            )
        )

        summary = engine.evaluate_pools(sample_pools)
        # aave (3.8%) and curve (3.2%) trigger
        assert summary.alerts_sent == 2
        # Each alert sent to 2 channels = 2 payloads per alert
        # But webhook tracks per-sent, so let's check
        assert len(channel_b.sent_payloads) == 2


# Type alias for pytest caplog fixture
try:
    from _pytest.logging import LogCaptureFixture as _CaplogType
except ImportError:
    _CaplogType = object  # type: ignore[misc,assignment]
