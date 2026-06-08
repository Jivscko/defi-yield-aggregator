"""Tests for the Historical APY Tracker module."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from defi_yield_aggregator.core.historical_tracker import (
    APYSnapshot,
    HistoricalAPYTracker,
    PoolStatistics,
    TrendAnalysis,
    TrendDirection,
)


# --- APYSnapshot tests ---


class TestAPYSnapshot:
    """Tests for APYSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        ts = datetime(2025, 1, 1)
        snap = APYSnapshot(pool_id="p1", apy=0.05, timestamp=ts, tvl_usd=1_000_000)
        assert snap.pool_id == "p1"
        assert snap.apy == 0.05
        assert snap.tvl_usd == 1_000_000
        assert snap.timestamp == ts

    def test_snapshot_defaults(self) -> None:
        snap = APYSnapshot(pool_id="p1", apy=0.03, timestamp=datetime.utcnow())
        assert snap.tvl_usd == 0.0

    def test_negative_apy_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            APYSnapshot(pool_id="p1", apy=-0.01, timestamp=datetime.utcnow())

    def test_extreme_apy_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds 10000%"):
            APYSnapshot(pool_id="p1", apy=101.0, timestamp=datetime.utcnow())

    def test_frozen(self) -> None:
        snap = APYSnapshot(pool_id="p1", apy=0.05, timestamp=datetime.utcnow())
        with pytest.raises(AttributeError):
            snap.apy = 0.10  # type: ignore[misc]


# --- HistoricalAPYTracker tests ---


def _make_tracker_with_data(
    pool_id: str = "pool-a",
    days: int = 30,
    base_apy: float = 0.05,
    daily_change: float = 0.0005,
) -> tuple[HistoricalAPYTracker, datetime]:
    """Create a tracker pre-loaded with synthetic data."""
    tracker = HistoricalAPYTracker()
    base = datetime(2025, 1, 1)
    for i in range(days):
        ts = base + timedelta(days=i)
        apy = max(0.001, base_apy + daily_change * i)
        tracker.record(pool_id, apy=apy, tvl_usd=1_000_000 + i * 10_000, timestamp=ts)
    return tracker, base


class TestRecording:
    """Tests for recording APY snapshots."""

    def test_record_single(self) -> None:
        tracker = HistoricalAPYTracker()
        snap = tracker.record("p1", apy=0.05, tvl_usd=100)
        assert snap.pool_id == "p1"
        assert snap.apy == 0.05
        assert tracker.snapshot_count("p1") == 1

    def test_record_multiple_pools(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("p1", apy=0.05)
        tracker.record("p2", apy=0.03)
        tracker.record("p1", apy=0.06)
        assert tracker.snapshot_count("p1") == 2
        assert tracker.snapshot_count("p2") == 1

    def test_record_batch(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        snaps = [
            APYSnapshot(pool_id="p1", apy=0.05 + i * 0.001, timestamp=base + timedelta(days=i))
            for i in range(5)
        ]
        count = tracker.record_batch(snaps)
        assert count == 5
        assert tracker.snapshot_count("p1") == 5

    def test_max_snapshots_pruning(self) -> None:
        tracker = HistoricalAPYTracker(max_snapshots_per_pool=5)
        base = datetime(2025, 1, 1)
        for i in range(10):
            tracker.record("p1", apy=0.05, timestamp=base + timedelta(days=i))
        assert tracker.snapshot_count("p1") == 5
        # Should have kept the last 5
        latest = tracker.get_latest("p1")
        assert latest is not None
        assert latest.timestamp == base + timedelta(days=9)

    def test_custom_timestamp(self) -> None:
        tracker = HistoricalAPYTracker()
        ts = datetime(2024, 6, 15, 12, 30)
        snap = tracker.record("p1", apy=0.04, timestamp=ts)
        assert snap.timestamp == ts


class TestRetrieval:
    """Tests for retrieving snapshots."""

    def test_get_snapshots_all(self) -> None:
        tracker, _ = _make_tracker_with_data(days=10)
        snaps = tracker.get_snapshots("pool-a")
        assert len(snaps) == 10

    def test_get_snapshots_with_since(self) -> None:
        tracker, base = _make_tracker_with_data(days=10)
        since = base + timedelta(days=5)
        snaps = tracker.get_snapshots("pool-a", since=since)
        assert all(s.timestamp >= since for s in snaps)
        assert len(snaps) == 5

    def test_get_snapshots_with_until(self) -> None:
        tracker, base = _make_tracker_with_data(days=10)
        until = base + timedelta(days=4)
        snaps = tracker.get_snapshots("pool-a", until=until)
        assert all(s.timestamp <= until for s in snaps)
        assert len(snaps) == 5

    def test_get_snapshots_with_both_bounds(self) -> None:
        tracker, base = _make_tracker_with_data(days=10)
        since = base + timedelta(days=3)
        until = base + timedelta(days=6)
        snaps = tracker.get_snapshots("pool-a", since=since, until=until)
        assert len(snaps) == 4

    def test_get_snapshots_empty_pool(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_snapshots("nonexistent") == []

    def test_get_latest(self) -> None:
        tracker, base = _make_tracker_with_data(days=5)
        latest = tracker.get_latest("pool-a")
        assert latest is not None
        assert latest.timestamp == base + timedelta(days=4)

    def test_get_latest_empty(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_latest("p1") is None

    def test_get_pool_ids(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("c", apy=0.03)
        tracker.record("a", apy=0.01)
        tracker.record("b", apy=0.02)
        assert tracker.get_pool_ids() == ["a", "b", "c"]

    def test_get_pool_ids_empty(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_pool_ids() == []


class TestStatistics:
    """Tests for statistical calculations."""

    def test_basic_statistics(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        # Record known APYs: 0.02, 0.04, 0.06, 0.08, 0.10
        for i in range(5):
            tracker.record("p1", apy=0.02 + i * 0.02, timestamp=base + timedelta(days=i))

        stats = tracker.get_statistics("p1")
        assert stats is not None
        assert stats.pool_id == "p1"
        assert stats.num_snapshots == 5
        assert stats.mean_apy == pytest.approx(0.06)
        assert stats.median_apy == pytest.approx(0.06)
        assert stats.min_apy == pytest.approx(0.02)
        assert stats.max_apy == pytest.approx(0.10)

    def test_statistics_with_window(self) -> None:
        tracker, base = _make_tracker_with_data(days=30, base_apy=0.05)
        # Only last 7 days
        stats = tracker.get_statistics("pool-a", window_days=7)
        assert stats is not None
        # window_days=7 with >= cutoff includes 8 snapshots (cutoff day + 7)
        assert stats.num_snapshots == 8
        assert stats.window_days == 7

    def test_statistics_empty_pool(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_statistics("p1") is None

    def test_statistics_single_snapshot(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("p1", apy=0.05)
        stats = tracker.get_statistics("p1")
        assert stats is not None
        assert stats.std_dev == 0.0
        assert stats.mean_apy == pytest.approx(0.05)

    def test_coefficient_of_variation(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        # High variance data
        for i, apy in enumerate([0.01, 0.10, 0.01, 0.10, 0.01]):
            tracker.record("volatile", apy=apy, timestamp=base + timedelta(days=i))
        stats = tracker.get_statistics("volatile")
        assert stats is not None
        assert stats.coefficient_of_variation > 0.5  # High CV for volatile data


class TestMovingAverage:
    """Tests for moving average calculations."""

    def test_7d_moving_average(self) -> None:
        tracker, _ = _make_tracker_with_data(days=14, base_apy=0.05, daily_change=0.001)
        ma = tracker.get_moving_average("pool-a", window_days=7)
        assert ma is not None
        # MA should be between min and max
        stats = tracker.get_statistics("pool-a")
        assert stats is not None
        assert stats.min_apy <= ma <= stats.max_apy

    def test_moving_average_empty(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_moving_average("p1", window_days=7) is None

    def test_moving_average_single_point(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("p1", apy=0.05)
        ma = tracker.get_moving_average("p1", window_days=7)
        assert ma == pytest.approx(0.05)


class TestTrendAnalysis:
    """Tests for trend detection."""

    def test_increasing_trend(self) -> None:
        tracker, _ = _make_tracker_with_data(
            days=30, base_apy=0.02, daily_change=0.002
        )
        trend = tracker.get_trend("pool-a", window_days=30)
        assert trend is not None
        assert trend.direction == TrendDirection.INCREASING
        assert trend.slope > 0
        assert trend.r_squared > 0.8  # Should be a good fit

    def test_decreasing_trend(self) -> None:
        tracker, _ = _make_tracker_with_data(
            days=30, base_apy=0.10, daily_change=-0.002
        )
        trend = tracker.get_trend("pool-a", window_days=30)
        assert trend is not None
        assert trend.direction == TrendDirection.DECREASING
        assert trend.slope < 0

    def test_stable_trend(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        for i in range(30):
            # Very small random-looking variation around 5%
            apy = 0.05 + 0.00001 * (i % 3 - 1)
            tracker.record("p1", apy=apy, timestamp=base + timedelta(days=i))

        trend = tracker.get_trend("p1", window_days=30)
        assert trend is not None
        assert trend.direction == TrendDirection.STABLE

    def test_insufficient_data(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        tracker.record("p1", apy=0.05, timestamp=base)
        tracker.record("p1", apy=0.06, timestamp=base + timedelta(days=1))

        trend = tracker.get_trend("p1", min_points=3)
        assert trend is not None
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA

    def test_trend_includes_moving_averages(self) -> None:
        tracker, _ = _make_tracker_with_data(days=30)
        trend = tracker.get_trend("pool-a")
        assert trend is not None
        assert trend.ma_7d is not None
        assert trend.ma_30d is not None

    def test_trend_current_apy(self) -> None:
        tracker, _ = _make_tracker_with_data(days=10, base_apy=0.05, daily_change=0.001)
        trend = tracker.get_trend("pool-a")
        assert trend is not None
        assert trend.current_apy == pytest.approx(0.05 + 0.001 * 9)

    def test_trend_empty_pool(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.get_trend("p1") is None

    def test_get_all_trends(self) -> None:
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)
        for i in range(10):
            tracker.record("p1", apy=0.05 + i * 0.001, timestamp=base + timedelta(days=i))
            tracker.record("p2", apy=0.03, timestamp=base + timedelta(days=i))

        trends = tracker.get_all_trends()
        assert "p1" in trends
        assert "p2" in trends
        assert len(trends) == 2


class TestClear:
    """Tests for clearing stored data."""

    def test_clear_specific_pool(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("p1", apy=0.05)
        tracker.record("p2", apy=0.03)
        removed = tracker.clear("p1")
        assert removed == 1
        assert tracker.snapshot_count("p1") == 0
        assert tracker.snapshot_count("p2") == 1

    def test_clear_all(self) -> None:
        tracker = HistoricalAPYTracker()
        tracker.record("p1", apy=0.05)
        tracker.record("p2", apy=0.03)
        removed = tracker.clear()
        assert removed == 2
        assert tracker.get_pool_ids() == []

    def test_clear_nonexistent(self) -> None:
        tracker = HistoricalAPYTracker()
        assert tracker.clear("nonexistent") == 0


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self) -> None:
        """Simulate tracking APY over 60 days with multiple pools."""
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)

        # Pool A: gradually increasing yield
        for i in range(60):
            tracker.record(
                "aave-usdc",
                apy=0.03 + i * 0.0005,
                tvl_usd=5_000_000 + i * 50_000,
                timestamp=base + timedelta(days=i),
            )

        # Pool B: volatile yield
        for i in range(60):
            apy = 0.08 + 0.02 * math.sin(i * 0.5)
            tracker.record(
                "curve-3pool",
                apy=apy,
                tvl_usd=10_000_000,
                timestamp=base + timedelta(days=i),
            )

        # Verify tracking
        assert tracker.snapshot_count("aave-usdc") == 60
        assert tracker.snapshot_count("curve-3pool") == 60

        # Pool A should show increasing trend
        trend_a = tracker.get_trend("aave-usdc")
        assert trend_a is not None
        assert trend_a.direction == TrendDirection.INCREASING

        # Pool B should show stable (sin wave averages out)
        trend_b = tracker.get_trend("curve-3pool")
        assert trend_b is not None
        # Sin wave oscillation should be classified as stable or have low r_squared
        assert trend_b.r_squared < 0.5  # Sin wave is poorly fit by linear model

        # Get stats for different windows
        stats_30d = tracker.get_statistics("aave-usdc", window_days=30)
        stats_all = tracker.get_statistics("aave-usdc")
        assert stats_30d is not None
        assert stats_all is not None
        # window_days=30 with >= cutoff includes 31 snapshots (cutoff day + 30)
        assert stats_30d.num_snapshots == 31
        assert stats_all.num_snapshots == 60

        # 30d mean should be higher than overall mean (increasing trend)
        assert stats_30d.mean_apy > stats_all.mean_apy

    def test_data_window_filtering(self) -> None:
        """Verify time-based filtering works correctly."""
        tracker = HistoricalAPYTracker()
        base = datetime(2025, 1, 1)

        for i in range(90):
            tracker.record("p1", apy=0.05, timestamp=base + timedelta(days=i))

        # 7-day window
        snaps_7d = tracker.get_snapshots("p1", since=base + timedelta(days=83))
        assert len(snaps_7d) == 7

        # 30-day window
        snaps_30d = tracker.get_snapshots("p1", since=base + timedelta(days=60))
        assert len(snaps_30d) == 30
