"""Alert and notification system for DeFi yield monitoring.

Monitors portfolio health, pool risk changes, APY shifts, and TVL drops.
Supports multiple delivery channels (webhook, Telegram, console) with
configurable thresholds, rate limiting, and alert deduplication.

Usage::

    from defi_yield_aggregator.core.alerts import AlertEngine, AlertRule, AlertSeverity

    engine = AlertEngine()
    engine.add_rule(AlertRule(
        name="high-risk-breach",
        condition=AlertCondition.RISK_SCORE_ABOVE,
        threshold=70.0,
        severity=AlertSeverity.CRITICAL,
        channels=[WebhookChannel("https://hooks.example.com/defi")],
    ))
    alerts = engine.evaluate_pools(pools, risk_scores)
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from defi_yield_aggregator.core.models import (
    OptimizedPortfolio,
    PoolInfo,
    RiskScore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(str, Enum):
    """Types of conditions that trigger alerts."""

    APY_DROP = "apy_drop"  # APY dropped below threshold %
    APY_CHANGE = "apy_change"  # APY changed by more than threshold %
    RISK_SCORE_ABOVE = "risk_score_above"  # Risk score exceeds threshold
    TVL_DROP = "tvl_drop"  # TVL dropped by more than threshold %
    PORTFOLIO_DRIFT = "portfolio_drift"  # Allocation drifted more than threshold %
    POOL_DEGRADED = "pool_degraded"  # Composite: risk up + APY down


class AlertStatus(str, Enum):
    """Status of an individual alert."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SUPPRESSED = "suppressed"  # Deduplicated or rate-limited


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """A single alert instance.

    Attributes:
        rule_name: Name of the rule that triggered this alert.
        severity: How urgent the alert is.
        pool_id: Pool that triggered the alert (empty for portfolio-level).
        message: Human-readable alert message.
        details: Additional structured data for the alert.
        triggered_at: When the alert was generated.
        status: Current delivery status.
        fingerprint: Deduplication hash (same alert within window = suppressed).
    """

    rule_name: str
    severity: AlertSeverity
    pool_id: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AlertStatus = AlertStatus.PENDING
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            # Generate fingerprint from rule + pool + severity for dedup
            raw = f"{self.rule_name}:{self.pool_id}:{self.severity.value}"
            object.__setattr__(
                self, "fingerprint", hashlib.sha256(raw.encode()).hexdigest()[:16]
            )


@dataclass
class AlertRule:
    """Configuration for an alert rule.

    Attributes:
        name: Unique rule identifier.
        condition: What condition to check.
        threshold: Numeric threshold for the condition.
        severity: Alert severity when triggered.
        channels: Delivery channels for this rule.
        dedup_window_minutes: Suppress duplicate alerts within this window.
        enabled: Whether this rule is active.
        cooldown_minutes: Minimum time between alerts for the same pool.
        message_template: Custom message template. Use {pool_id}, {value},
            {threshold}, {protocol}, {chain} placeholders.
    """

    name: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity
    channels: list["BaseChannel"] = field(default_factory=list)
    dedup_window_minutes: int = 60
    enabled: bool = True
    cooldown_minutes: int = 15
    message_template: str = ""


@dataclass
class AlertHistory:
    """Tracks alert history for deduplication and rate limiting.

    Attributes:
        fingerprint: Alert fingerprint hash.
        last_sent_at: When this alert was last delivered.
        count: Total times this alert has been triggered.
        suppression_count: Times this alert was suppressed.
    """

    fingerprint: str
    last_sent_at: datetime
    count: int = 1
    suppression_count: int = 0


@dataclass
class AlertSummary:
    """Summary of an alert evaluation run.

    Attributes:
        total_alerts_triggered: Total alerts generated.
        alerts_sent: Successfully delivered alerts.
        alerts_suppressed: Alerts suppressed by dedup/rate-limit.
        alerts_failed: Alerts that failed delivery.
        alerts: All generated alerts.
        evaluated_at: When the evaluation ran.
    """

    total_alerts_triggered: int
    alerts_sent: int
    alerts_suppressed: int
    alerts_failed: int
    alerts: list[Alert] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Delivery channels
# ---------------------------------------------------------------------------


class BaseChannel(abc.ABC):
    """Abstract base class for alert delivery channels."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Channel identifier."""
        ...

    @abc.abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert through this channel.

        Args:
            alert: The alert to deliver.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        ...


class WebhookChannel(BaseChannel):
    """Deliver alerts via HTTP POST to a webhook URL.

    Sends a JSON payload with alert details. Supports custom headers for
    authentication (e.g., Bearer tokens, signing secrets).

    Attributes:
        url: Webhook endpoint URL.
        headers: Additional HTTP headers (e.g., Authorization).
        timeout_seconds: Request timeout.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout_seconds: int = 10,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid webhook URL: {url}")
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout_seconds
        self._sent_payloads: list[dict[str, Any]] = []  # For testing

    @property
    def name(self) -> str:
        return f"webhook({self._url[:40]}...)"

    def send(self, alert: Alert) -> bool:
        """Send alert as JSON POST.

        In production this would use aiohttp/requests. For now, we store
        the payload for testing and log the alert.
        """
        payload = {
            "rule": alert.rule_name,
            "severity": alert.severity.value,
            "pool_id": alert.pool_id,
            "message": alert.message,
            "details": alert.details,
            "triggered_at": alert.triggered_at.isoformat(),
            "fingerprint": alert.fingerprint,
        }
        self._sent_payloads.append(payload)

        # In production: requests.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
        logger.info("Webhook alert sent to %s: %s", self._url, alert.message)
        return True

    @property
    def sent_payloads(self) -> list[dict[str, Any]]:
        """Return sent payloads (for testing)."""
        return list(self._sent_payloads)


class TelegramChannel(BaseChannel):
    """Deliver alerts via Telegram Bot API.

    Sends formatted messages to a Telegram chat using the bot token.
    Messages include severity emoji and structured formatting.

    Attributes:
        bot_token: Telegram bot API token.
        chat_id: Target chat/group ID.
        timeout_seconds: API request timeout.
    """

    _SEVERITY_EMOJI: dict[AlertSeverity, str] = {
        AlertSeverity.INFO: "INFO",
        AlertSeverity.WARNING: "WARN",
        AlertSeverity.CRITICAL: "CRIT",
    }

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: int = 10,
        parse_mode: str = "MarkdownV2",
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("bot_token and chat_id are required")
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_seconds
        self._parse_mode = parse_mode
        self._sent_messages: list[str] = []  # For testing

    @property
    def name(self) -> str:
        masked = self._bot_token[:6] + "***" if len(self._bot_token) > 6 else "***"
        return f"telegram({masked})"

    def send(self, alert: Alert) -> bool:
        """Send alert as Telegram message.

        Formats the message with severity label, pool info, and details.
        In production this would POST to the Telegram Bot API.
        """
        severity_label = self._SEVERITY_EMOJI.get(alert.severity, "ALERT")

        message = (
            f"[{severity_label}] DeFi Alert: {alert.rule_name}\n"
            f"Pool: {alert.pool_id}\n"
            f"{alert.message}\n"
        )

        if alert.details:
            for key, value in alert.details.items():
                message += f"  {key}: {value}\n"

        self._sent_messages.append(message)

        # In production:
        # requests.post(
        #     f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
        #     json={"chat_id": self._chat_id, "text": message, "parse_mode": self._parse_mode},
        #     timeout=self._timeout,
        # )
        logger.info("Telegram alert sent to chat %s: %s", self._chat_id, alert.message)
        return True

    @property
    def sent_messages(self) -> list[str]:
        """Return sent messages (for testing)."""
        return list(self._sent_messages)


class ConsoleChannel(BaseChannel):
    """Deliver alerts to the Python logger.

    Useful for development, testing, and local monitoring. Logs at
    WARNING or CRITICAL level depending on severity.
    """

    @property
    def name(self) -> str:
        return "console"

    def send(self, alert: Alert) -> bool:
        """Log the alert message."""
        log_msg = "[%s] %s | pool=%s | %s | %s"
        log_args = (
            alert.severity.value.upper(),
            alert.rule_name,
            alert.pool_id,
            alert.message,
            json.dumps(alert.details) if alert.details else "",
        )

        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_msg, *log_args)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg, *log_args)
        else:
            logger.info(log_msg, *log_args)

        return True


# ---------------------------------------------------------------------------
# Default message templates
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: dict[AlertCondition, str] = {
    AlertCondition.APY_DROP: (
        "Pool {pool_id} APY dropped to {value:.2f}% (threshold: {threshold:.2f}%)"
    ),
    AlertCondition.APY_CHANGE: (
        "Pool {pool_id} APY changed by {value:+.2f}% (threshold: {threshold:.2f}%)"
    ),
    AlertCondition.RISK_SCORE_ABOVE: (
        "Pool {pool_id} risk score at {value:.1f}/100 (threshold: {threshold:.1f})"
    ),
    AlertCondition.TVL_DROP: (
        "Pool {pool_id} TVL dropped {value:.1f}% to ${tvl:,.0f} (threshold: {threshold:.1f}%)"
    ),
    AlertCondition.PORTFOLIO_DRIFT: (
        "Pool {pool_id} allocation drifted {value:.1f}% from target {target:.1f}% "
        "(threshold: {threshold:.1f}%)"
    ),
    AlertCondition.POOL_DEGRADED: (
        "Pool {pool_id} degraded: risk {risk:.1f} (+{risk_change:+.1f}), "
        "APY {apy:.2f}% ({apy_change:+.2f}%)"
    ),
}


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Core alert evaluation engine.

    Evaluates pool data, risk scores, and portfolio allocations against
    configured rules. Manages deduplication, cooldowns, and delivery.

    Usage::

        engine = AlertEngine()
        engine.add_rule(AlertRule(...))
        summary = engine.evaluate_pools(pools, risk_scores)
    """

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._history: dict[str, AlertHistory] = {}
        self._previous_apys: dict[str, float] = {}  # pool_id -> last APY
        self._previous_tvls: dict[str, float] = {}  # pool_id -> last TVL
        self._previous_risks: dict[str, float] = {}  # pool_id -> last risk score

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule.

        Args:
            rule: The alert rule to add.

        Raises:
            ValueError: If a rule with the same name already exists.
        """
        if rule.name in self._rules:
            raise ValueError(f"Rule '{rule.name}' already exists")
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule by name.

        Args:
            name: Rule name to remove.

        Returns:
            True if the rule was found and removed.
        """
        return self._rules.pop(name, None) is not None

    def get_rule(self, name: str) -> Optional[AlertRule]:
        """Get a rule by name."""
        return self._rules.get(name)

    @property
    def rules(self) -> list[AlertRule]:
        """All registered rules."""
        return list(self._rules.values())

    def clear_history(self) -> None:
        """Clear all alert history (useful for testing)."""
        self._history.clear()

    def update_baselines(
        self,
        pools: list[PoolInfo],
        risk_scores: Optional[dict[str, RiskScore]] = None,
    ) -> None:
        """Update stored baselines for change detection.

        Call this periodically to track APY/TVL/risk deltas over time.

        Args:
            pools: Current pool data.
            risk_scores: Current risk scores (optional).
        """
        for pool in pools:
            self._previous_apys[pool.pool_id] = pool.apy
            self._previous_tvls[pool.pool_id] = pool.tvl_usd

        if risk_scores:
            for pool_id, risk in risk_scores.items():
                self._previous_risks[pool_id] = risk.overall_score

    def evaluate_pools(
        self,
        pools: list[PoolInfo],
        risk_scores: Optional[dict[str, RiskScore]] = None,
    ) -> AlertSummary:
        """Evaluate all rules against current pool data.

        Args:
            pools: Current pool data to evaluate.
            risk_scores: Risk scores keyed by pool_id.

        Returns:
            Summary of alerts triggered and delivery status.
        """
        now = datetime.now(timezone.utc)
        all_alerts: list[Alert] = []

        risk_map = risk_scores or {}

        for pool in pools:
            risk = risk_map.get(pool.pool_id)

            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                alert = self._check_rule(rule, pool, risk, now)
                if alert is not None:
                    all_alerts.append(alert)

        # Deliver and track
        sent = 0
        suppressed = 0
        failed = 0

        for alert in all_alerts:
            if alert.status == AlertStatus.SUPPRESSED:
                suppressed += 1
                continue

            delivery_ok = self._deliver(alert, all_alerts)
            if delivery_ok:
                sent += 1
            else:
                failed += 1

        # Update baselines after evaluation
        self.update_baselines(pools, risk_scores)

        return AlertSummary(
            total_alerts_triggered=len(all_alerts),
            alerts_sent=sent,
            alerts_suppressed=suppressed,
            alerts_failed=failed,
            alerts=all_alerts,
            evaluated_at=now,
        )

    def evaluate_portfolio_drift(
        self,
        portfolio: OptimizedPortfolio,
        current_allocations: dict[str, float],
        drift_threshold_pct: float = 5.0,
    ) -> AlertSummary:
        """Check for portfolio allocation drift.

        Compares target allocations from an optimized portfolio against
        current actual allocations.

        Args:
            portfolio: The target optimized portfolio.
            current_allocations: Current allocation % keyed by pool_id.
            drift_threshold_pct: Alert if drift exceeds this %.

        Returns:
            Summary of drift alerts.
        """
        now = datetime.now(timezone.utc)
        alerts: list[Alert] = []

        for alloc in portfolio.allocations:
            current = current_allocations.get(alloc.pool_id, 0.0)
            target = alloc.allocation_pct * 100
            current_pct = current * 100
            drift = abs(current_pct - target)

            if drift > drift_threshold_pct:
                template = _DEFAULT_TEMPLATES[AlertCondition.PORTFOLIO_DRIFT]
                message = template.format(
                    pool_id=alloc.pool_id,
                    value=drift,
                    target=target,
                    threshold=drift_threshold_pct,
                )

                alert = Alert(
                    rule_name="portfolio-drift",
                    severity=(
                        AlertSeverity.WARNING
                        if drift < drift_threshold_pct * 2
                        else AlertSeverity.CRITICAL
                    ),
                    pool_id=alloc.pool_id,
                    message=message,
                    details={
                        "target_pct": round(target, 2),
                        "current_pct": round(current_pct, 2),
                        "drift_pct": round(drift, 2),
                        "token_pair": alloc.token_pair,
                        "protocol": alloc.protocol.value,
                    },
                )

                # Apply dedup
                if self._should_deliver(alert, now):
                    alerts.append(alert)
                    self._record_delivery(alert, now)
                else:
                    object.__setattr__(alert, "status", AlertStatus.SUPPRESSED)
                    alerts.append(alert)

        sent = sum(1 for a in alerts if a.status != AlertStatus.SUPPRESSED)
        return AlertSummary(
            total_alerts_triggered=len(alerts),
            alerts_sent=sent,
            alerts_suppressed=len(alerts) - sent,
            alerts_failed=0,
            alerts=alerts,
            evaluated_at=now,
        )

    # -------------------------------------------------------------------
    # Internal evaluation
    # -------------------------------------------------------------------

    def _check_rule(
        self,
        rule: AlertRule,
        pool: PoolInfo,
        risk: Optional[RiskScore],
        now: datetime,
    ) -> Optional[Alert]:
        """Check a single rule against a pool. Returns Alert or None."""
        value: Optional[float] = None
        details: dict[str, Any] = {}

        if rule.condition == AlertCondition.APY_DROP:
            # Alert if current APY is below threshold
            if pool.apy * 100 < rule.threshold:
                value = pool.apy * 100

        elif rule.condition == AlertCondition.APY_CHANGE:
            # Alert if APY changed by more than threshold %
            prev_apy = self._previous_apys.get(pool.pool_id)
            if prev_apy is not None and prev_apy > 0:
                change_pct = ((pool.apy - prev_apy) / prev_apy) * 100
                if abs(change_pct) > rule.threshold:
                    value = change_pct
                    details["previous_apy"] = round(prev_apy * 100, 4)
                    details["current_apy"] = round(pool.apy * 100, 4)

        elif rule.condition == AlertCondition.RISK_SCORE_ABOVE:
            if risk is not None and risk.overall_score > rule.threshold:
                value = risk.overall_score
                details["risk_level"] = risk.risk_level.value

        elif rule.condition == AlertCondition.TVL_DROP:
            prev_tvl = self._previous_tvls.get(pool.pool_id)
            if prev_tvl is not None and prev_tvl > 0:
                drop_pct = ((prev_tvl - pool.tvl_usd) / prev_tvl) * 100
                if drop_pct > rule.threshold:
                    value = drop_pct
                    details["previous_tvl"] = prev_tvl
                    details["current_tvl"] = pool.tvl_usd

        elif rule.condition == AlertCondition.POOL_DEGRADED:
            # Composite: risk went up AND APY went down
            prev_apy = self._previous_apys.get(pool.pool_id)
            prev_risk = self._previous_risks.get(pool.pool_id)
            if prev_apy is not None and prev_risk is not None and risk is not None:
                apy_change = ((pool.apy - prev_apy) / prev_apy) * 100 if prev_apy > 0 else 0
                risk_change = risk.overall_score - prev_risk
                if apy_change < -rule.threshold and risk_change > rule.threshold:
                    value = risk_change
                    details["risk_change"] = round(risk_change, 2)
                    details["apy_change_pct"] = round(apy_change, 2)
                    details["current_risk"] = round(risk.overall_score, 1)
                    details["current_apy"] = round(pool.apy * 100, 4)

        if value is None:
            return None

        # Build message
        template = rule.message_template or _DEFAULT_TEMPLATES.get(
            rule.condition, "Alert: {pool_id} = {value}"
        )
        message = template.format(
            pool_id=pool.pool_id,
            value=value,
            threshold=rule.threshold,
            protocol=pool.protocol.value,
            chain=pool.chain.value,
            tvl=pool.tvl_usd,
            apy=pool.apy * 100,
            risk=risk.overall_score if risk else 0,
            risk_change=details.get("risk_change", 0),
            apy_change=details.get("apy_change_pct", 0),
            target=details.get("target_pct", 0),
        )

        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            pool_id=pool.pool_id,
            message=message,
            details=details,
            triggered_at=now,
        )

        # Dedup / cooldown check
        if not self._should_deliver(alert, now, rule):
            object.__setattr__(alert, "status", AlertStatus.SUPPRESSED)
            return alert

        self._record_delivery(alert, now)
        return alert

    def _should_deliver(
        self,
        alert: Alert,
        now: datetime,
        rule: Optional[AlertRule] = None,
    ) -> bool:
        """Check if an alert should be delivered (not deduplicated or cooled down)."""
        hist = self._history.get(alert.fingerprint)
        if hist is None:
            return True

        # Check dedup window
        if rule:
            window = timedelta(minutes=rule.dedup_window_minutes)
        else:
            window = timedelta(minutes=60)

        if now - hist.last_sent_at < window:
            return False

        # Check cooldown
        if rule and rule.cooldown_minutes > 0:
            cooldown = timedelta(minutes=rule.cooldown_minutes)
            if now - hist.last_sent_at < cooldown:
                return False

        return True

    def _record_delivery(self, alert: Alert, now: datetime) -> None:
        """Record that an alert was delivered."""
        if alert.fingerprint in self._history:
            self._history[alert.fingerprint].last_sent_at = now
            self._history[alert.fingerprint].count += 1
        else:
            self._history[alert.fingerprint] = AlertHistory(
                fingerprint=alert.fingerprint,
                last_sent_at=now,
            )

    def _deliver(self, alert: Alert, all_alerts: list[Alert]) -> bool:
        """Deliver an alert through all configured channels."""
        rule = self._rules.get(alert.rule_name)
        if rule is None:
            # Portfolio drift alerts don't have a registered rule
            # Use console as fallback
            console = ConsoleChannel()
            success = console.send(alert)
            object.__setattr__(
                alert,
                "status",
                AlertStatus.SENT if success else AlertStatus.FAILED,
            )
            return success

        all_ok = True
        for channel in rule.channels:
            try:
                ok = channel.send(alert)
                if not ok:
                    all_ok = False
            except Exception:
                logger.exception(
                    "Failed to deliver alert via %s", channel.name
                )
                all_ok = False

        object.__setattr__(
            alert,
            "status",
            AlertStatus.SENT if all_ok else AlertStatus.FAILED,
        )
        return all_ok

    # -------------------------------------------------------------------
    # Convenience factory methods
    # -------------------------------------------------------------------

    @classmethod
    def with_defaults(
        cls,
        webhook_url: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> "AlertEngine":
        """Create an engine with sensible default rules.

        Default rules:
        - APY drop below 2% (WARNING)
        - APY change >50% (WARNING)
        - Risk score above 70 (CRITICAL)
        - TVL drop >30% (CRITICAL)
        - Pool degradation (WARNING)

        Args:
            webhook_url: Optional webhook URL for delivery.
            telegram_token: Optional Telegram bot token.
            telegram_chat_id: Optional Telegram chat ID.

        Returns:
            Configured AlertEngine instance.
        """
        engine = cls()

        channels: list[BaseChannel] = [ConsoleChannel()]
        if webhook_url:
            channels.append(WebhookChannel(webhook_url))
        if telegram_token and telegram_chat_id:
            channels.append(TelegramChannel(telegram_token, telegram_chat_id))

        default_rules = [
            AlertRule(
                name="apy-drop-low",
                condition=AlertCondition.APY_DROP,
                threshold=2.0,
                severity=AlertSeverity.WARNING,
                channels=list(channels),
            ),
            AlertRule(
                name="apy-volatile",
                condition=AlertCondition.APY_CHANGE,
                threshold=50.0,
                severity=AlertSeverity.WARNING,
                channels=list(channels),
                cooldown_minutes=30,
            ),
            AlertRule(
                name="risk-critical",
                condition=AlertCondition.RISK_SCORE_ABOVE,
                threshold=70.0,
                severity=AlertSeverity.CRITICAL,
                channels=list(channels),
                dedup_window_minutes=120,
            ),
            AlertRule(
                name="tvl-crash",
                condition=AlertCondition.TVL_DROP,
                threshold=30.0,
                severity=AlertSeverity.CRITICAL,
                channels=list(channels),
            ),
            AlertRule(
                name="pool-degraded",
                condition=AlertCondition.POOL_DEGRADED,
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                channels=list(channels),
                cooldown_minutes=60,
            ),
        ]

        for rule in default_rules:
            engine.add_rule(rule)

        return engine
