"""Historical APY tracking with trend analysis and volatility metrics.

Tracks APY snapshots over time for each pool and provides analytics:
- Moving averages (configurable window)
- Trend detection (increasing, decreasing, stable)
- Volatility metrics (std dev, coefficient of variation)
- Min/max/mean statistics over time windows
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class TrendDirection(str, Enum):
    """Detected trend direction for APY time series."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class APYSnapshot:
    """A single APY measurement at a point in time."""

    pool_id: str
    apy: float
    timestamp: datetime
    tvl_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.apy < 0:
            raise ValueError(f"APY must be non-negative, got {self.apy}")
        if self.apy > 100:
            raise ValueError(f"APY {self.apy} exceeds 10000% threshold")


@dataclass
class PoolStatistics:
    """Statistical summary for a pool's APY history."""

    pool_id: str
    mean_apy: float
    median_apy: float
    std_dev: float
    min_apy: float
    max_apy: float
    coefficient_of_variation: float
    num_snapshots: int
    window_days: int


@dataclass
class TrendAnalysis:
    """Trend analysis result for a pool's APY."""

    pool_id: str
    direction: TrendDirection
    slope: float  # APY change per day
    r_squared: float  # Goodness of fit (0-1)
    current_apy: float
    ma_7d: Optional[float]
    ma_30d: Optional[float]
    window_days: int


class HistoricalAPYTracker:
    """Tracks and analyzes historical APY data for DeFi pools.

    Stores APY snapshots in memory and provides trend analysis,
    moving averages, and volatility metrics.

    Example::

        tracker = HistoricalAPYTracker()
        tracker.record("pool-1", apy=0.05, tvl_usd=1_000_000)
        tracker.record("pool-1", apy=0.055, tvl_usd=1_100_000)
        stats = tracker.get_statistics("pool-1")
        trend = tracker.get_trend("pool-1")
    """

    def __init__(self, max_snapshots_per_pool: int = 10_000) -> None:
        """Initialize the tracker.

        Args:
            max_snapshots_per_pool: Maximum snapshots to retain per pool.
                Oldest snapshots are pruned when exceeded.
        """
        self._max_snapshots = max_snapshots_per_pool
        self._snapshots: dict[str, list[APYSnapshot]] = {}

    def record(
        self,
        pool_id: str,
        apy: float,
        tvl_usd: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> APYSnapshot:
        """Record an APY snapshot for a pool.

        Args:
            pool_id: Unique pool identifier.
            apy: Current APY as decimal (e.g., 0.05 for 5%).
            tvl_usd: Total value locked at time of snapshot.
            timestamp: When the measurement was taken. Defaults to now (UTC).

        Returns:
            The created snapshot.

        Raises:
            ValueError: If APY is negative or exceeds threshold.
        """
        ts = timestamp or datetime.utcnow()
        snapshot = APYSnapshot(pool_id=pool_id, apy=apy, timestamp=ts, tvl_usd=tvl_usd)

        if pool_id not in self._snapshots:
            self._snapshots[pool_id] = []

        self._snapshots[pool_id].append(snapshot)

        # Prune old snapshots if exceeding limit
        if len(self._snapshots[pool_id]) > self._max_snapshots:
            self._snapshots[pool_id] = self._snapshots[pool_id][-self._max_snapshots :]

        return snapshot

    def record_batch(self, snapshots: list[APYSnapshot]) -> int:
        """Record multiple snapshots at once.

        Args:
            snapshots: List of APY snapshots to record.

        Returns:
            Number of snapshots successfully recorded.
        """
        count = 0
        for snap in snapshots:
            self.record(
                pool_id=snap.pool_id,
                apy=snap.apy,
                tvl_usd=snap.tvl_usd,
                timestamp=snap.timestamp,
            )
            count += 1
        return count

    def get_snapshots(
        self,
        pool_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[APYSnapshot]:
        """Retrieve snapshots for a pool within a time window.

        Args:
            pool_id: Pool to query.
            since: Start of time window (inclusive). None = no lower bound.
            until: End of time window (inclusive). None = no upper bound.

        Returns:
            List of snapshots in chronological order.
        """
        if pool_id not in self._snapshots:
            return []

        snaps = self._snapshots[pool_id]
        if since:
            snaps = [s for s in snaps if s.timestamp >= since]
        if until:
            snaps = [s for s in snaps if s.timestamp <= until]
        return snaps

    def get_latest(self, pool_id: str) -> Optional[APYSnapshot]:
        """Get the most recent snapshot for a pool.

        Args:
            pool_id: Pool to query.

        Returns:
            Latest snapshot or None if no data exists.
        """
        snaps = self._snapshots.get(pool_id, [])
        return snaps[-1] if snaps else None

    def get_pool_ids(self) -> list[str]:
        """Get all tracked pool IDs.

        Returns:
            Sorted list of pool IDs with recorded data.
        """
        return sorted(self._snapshots.keys())

    def snapshot_count(self, pool_id: str) -> int:
        """Get the number of recorded snapshots for a pool.

        Args:
            pool_id: Pool to query.

        Returns:
            Number of snapshots stored.
        """
        return len(self._snapshots.get(pool_id, []))

    def get_statistics(
        self,
        pool_id: str,
        window_days: Optional[int] = None,
    ) -> Optional[PoolStatistics]:
        """Calculate statistical summary for a pool's APY history.

        Args:
            pool_id: Pool to analyze.
            window_days: Look back N days from latest snapshot.
                None = use all available data.

        Returns:
            Statistics summary, or None if insufficient data.
        """
        snaps = self._snapshots.get(pool_id, [])
        if not snaps:
            return None

        if window_days is not None:
            cutoff = snaps[-1].timestamp - timedelta(days=window_days)
            snaps = [s for s in snaps if s.timestamp >= cutoff]

        if not snaps:
            return None

        apys = [s.apy for s in snaps]
        mean_val = statistics.mean(apys)
        std_val = statistics.pstdev(apys) if len(apys) > 1 else 0.0
        cov = (std_val / mean_val) if mean_val > 0 else 0.0

        return PoolStatistics(
            pool_id=pool_id,
            mean_apy=mean_val,
            median_apy=statistics.median(apys),
            std_dev=std_val,
            min_apy=min(apys),
            max_apy=max(apys),
            coefficient_of_variation=cov,
            num_snapshots=len(snaps),
            window_days=window_days or 0,
        )

    def get_moving_average(
        self,
        pool_id: str,
        window_days: int,
    ) -> Optional[float]:
        """Calculate simple moving average for a pool.

        Args:
            pool_id: Pool to analyze.
            window_days: Number of days for the moving average window.

        Returns:
            Moving average APY, or None if insufficient data.
        """
        snaps = self._snapshots.get(pool_id, [])
        if not snaps:
            return None

        cutoff = snaps[-1].timestamp - timedelta(days=window_days)
        window_snaps = [s for s in snaps if s.timestamp >= cutoff]

        if not window_snaps:
            return None

        return statistics.mean(s.apy for s in window_snaps)

    def get_trend(
        self,
        pool_id: str,
        window_days: int = 30,
        min_points: int = 3,
    ) -> Optional[TrendAnalysis]:
        """Analyze APY trend using linear regression.

        Uses least-squares regression on APY vs time to detect
        whether yields are increasing, decreasing, or stable.

        Args:
            pool_id: Pool to analyze.
            window_days: Number of days to look back for trend analysis.
            min_points: Minimum data points required for analysis.

        Returns:
            Trend analysis, or None if insufficient data.
        """
        snaps = self._snapshots.get(pool_id, [])
        if not snaps:
            return None

        cutoff = snaps[-1].timestamp - timedelta(days=window_days)
        window_snaps = [s for s in snaps if s.timestamp >= cutoff]

        if len(window_snaps) < min_points:
            latest = snaps[-1]
            return TrendAnalysis(
                pool_id=pool_id,
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope=0.0,
                r_squared=0.0,
                current_apy=latest.apy,
                ma_7d=self.get_moving_average(pool_id, 7),
                ma_30d=self.get_moving_average(pool_id, 30),
                window_days=window_days,
            )

        # Linear regression: APY = slope * days + intercept
        base_time = window_snaps[0].timestamp
        x = [(s.timestamp - base_time).total_seconds() / 86400 for s in window_snaps]
        y = [s.apy for s in window_snaps]
        n = len(x)

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        ss_yy = sum((yi - y_mean) ** 2 for yi in y)

        if ss_xx == 0:
            slope = 0.0
            r_squared = 0.0
        else:
            slope = ss_xy / ss_xx
            r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

        # Classify direction based on slope magnitude relative to mean APY
        relative_slope = abs(slope / y_mean) if y_mean > 0 else 0.0
        stability_threshold = 0.001  # 0.1% per day relative change

        if relative_slope < stability_threshold:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        return TrendAnalysis(
            pool_id=pool_id,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            current_apy=window_snaps[-1].apy,
            ma_7d=self.get_moving_average(pool_id, 7),
            ma_30d=self.get_moving_average(pool_id, 30),
            window_days=window_days,
        )

    def get_all_trends(
        self,
        window_days: int = 30,
    ) -> dict[str, TrendAnalysis]:
        """Get trend analysis for all tracked pools.

        Args:
            window_days: Look back period for trend analysis.

        Returns:
            Mapping of pool_id to trend analysis.
        """
        results: dict[str, TrendAnalysis] = {}
        for pool_id in self._snapshots:
            trend = self.get_trend(pool_id, window_days=window_days)
            if trend is not None:
                results[pool_id] = trend
        return results

    def clear(self, pool_id: Optional[str] = None) -> int:
        """Clear stored snapshots.

        Args:
            pool_id: Clear only this pool's data. None = clear all.

        Returns:
            Number of snapshots removed.
        """
        if pool_id is not None:
            removed = len(self._snapshots.pop(pool_id, []))
            return removed
        total = sum(len(v) for v in self._snapshots.values())
        self._snapshots.clear()
        return total
