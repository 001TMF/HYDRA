"""Data quality monitoring with staleness detection, validators, and reporting.

Monitors all data sources (futures, options, COT) for freshness, completeness,
and validity. Staleness detection is weekend-aware to avoid false alerts when
markets are closed. Options chain quality checks include liquid strike count,
call price monotonicity, and put-call parity.

This is the early warning system that prevents the system from trading on
stale or corrupted data -- the most insidious failure mode in a trading system
(research pitfall #3: data pipeline silent failures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import structlog

from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes for structured reporting
# ---------------------------------------------------------------------------


@dataclass
class StalenessAlert:
    """Alert for a single data source's staleness status."""

    source: str  # "futures", "options", "cot"
    last_update: Optional[datetime]
    threshold_days: float
    is_stale: bool
    days_since_update: Optional[float]


@dataclass
class OptionsQualityReport:
    """Quality assessment for an options chain."""

    liquid_strike_count: int
    total_strikes: int
    has_arbitrage_violation: bool
    put_call_parity_max_error: Optional[float]
    warnings: list[str] = field(default_factory=list)


@dataclass
class COTFreshnessReport:
    """Freshness assessment for COT data."""

    as_of_date: Optional[datetime]
    available_at_date: Optional[datetime]
    days_since_update: Optional[float]
    is_stale: bool


@dataclass
class QualityReport:
    """Structured quality report across all data sources."""

    market: str
    timestamp: datetime
    staleness_alerts: list[StalenessAlert]
    options_quality: Optional[OptionsQualityReport]
    cot_freshness: Optional[COTFreshnessReport]
    overall_status: str  # "healthy", "degraded", "stale"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_trading_day(date: datetime) -> bool:
    """Return True if the date is a weekday (Mon-Fri).

    Uses a weekend-only heuristic for Phase 1. Does NOT track exchange
    holidays -- that is too complex and fragile for the initial implementation.
    """
    return date.weekday() < 5  # 0=Monday, 4=Friday


def _count_trading_days_between(start: datetime, end: datetime) -> int:
    """Count trading days (weekdays) between start and end, exclusive of start."""
    if end <= start:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


# ---------------------------------------------------------------------------
# DataQualityMonitor
# ---------------------------------------------------------------------------


class DataQualityMonitor:
    """Monitor data quality across all sources with configurable thresholds.

    Parameters
    ----------
    parquet_lake : ParquetLake
        Parquet data lake for reading raw data.
    feature_store : FeatureStore
        Feature store for reading latest feature timestamps.
    config : dict
        Configuration dict containing staleness thresholds and quality params.
        Expected keys:
            staleness_thresholds.futures_days: int
            staleness_thresholds.options_days: int
            staleness_thresholds.cot_days: int
            quality.min_liquid_strikes: int
            quality.max_spread_pct: float
            quality.min_oi: int
    """

    def __init__(
        self,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
        config: dict,
    ) -> None:
        self.parquet_lake = parquet_lake
        self.feature_store = feature_store
        self.config = config

        staleness = config.get("staleness_thresholds", {})
        self.futures_stale_days = staleness.get("futures_days", 1)
        self.options_stale_days = staleness.get("options_days", 1)
        self.cot_stale_days = staleness.get("cot_days", 7)

        quality = config.get("quality", {})
        self.min_liquid_strikes = quality.get("min_liquid_strikes", 8)
        self.max_spread_pct = quality.get("max_spread_pct", 0.20)
        self.min_oi = quality.get("min_oi", 50)

    def check_staleness(
        self, market: str, current_time: datetime
    ) -> list[StalenessAlert]:
        """Check staleness across all data sources for a market.

        Weekend-aware: if current_time is a weekend or follows a weekend,
        trading day gaps are counted rather than calendar days for
        futures/options. COT uses calendar days (7-day cycle).

        Parameters
        ----------
        market : str
            Market identifier (e.g. "HE").
        current_time : datetime
            The reference time for staleness checks.

        Returns
        -------
        list[StalenessAlert]
            One alert per data source.
        """
        alerts: list[StalenessAlert] = []

        for source, threshold_days, use_trading_days in [
            ("futures", self.futures_stale_days, True),
            ("options", self.options_stale_days, True),
            ("cot", self.cot_stale_days, False),
        ]:
            last_update = self._get_latest_timestamp(market, source)

            if last_update is None:
                alerts.append(
                    StalenessAlert(
                        source=source,
                        last_update=None,
                        threshold_days=threshold_days,
                        is_stale=True,
                        days_since_update=None,
                    )
                )
                continue

            if use_trading_days:
                days_gap = _count_trading_days_between(last_update, current_time)
            else:
                days_gap = (current_time - last_update).total_seconds() / 86400.0

            is_stale = days_gap > threshold_days

            alerts.append(
                StalenessAlert(
                    source=source,
                    last_update=last_update,
                    threshold_days=threshold_days,
                    is_stale=is_stale,
                    days_since_update=days_gap,
                )
            )

        return alerts

    def check_options_quality(
        self, market: str, date: datetime
    ) -> OptionsQualityReport:
        """Check options chain quality for a market on a given date.

        Validates:
        - Liquid strike count (OI >= min_oi AND spread <= max_spread_pct)
        - Call price monotonicity (calls should decrease with strike)
        - Put-call parity (rough check)

        Parameters
        ----------
        market : str
            Market identifier.
        date : datetime
            Reference date to read options data.

        Returns
        -------
        OptionsQualityReport with quality metrics.
        """
        warnings_list: list[str] = []

        try:
            table = self.parquet_lake.read("options", market)
        except Exception:
            return OptionsQualityReport(
                liquid_strike_count=0,
                total_strikes=0,
                has_arbitrage_violation=False,
                put_call_parity_max_error=None,
                warnings=["No options data available"],
            )

        if len(table) == 0:
            return OptionsQualityReport(
                liquid_strike_count=0,
                total_strikes=0,
                has_arbitrage_violation=False,
                put_call_parity_max_error=None,
                warnings=["Empty options dataset"],
            )

        # Convert to pandas-like column access via PyArrow
        columns = table.column_names

        # Extract arrays
        strikes = table.column("strike").to_pylist() if "strike" in columns else []
        total_strikes = len(strikes)

        oi = table.column("oi").to_pylist() if "oi" in columns else [0] * total_strikes
        spread_pct = (
            table.column("bid_ask_spread_pct").to_pylist()
            if "bid_ask_spread_pct" in columns
            else [0.0] * total_strikes
        )
        call_prices = (
            table.column("call_price").to_pylist()
            if "call_price" in columns
            else []
        )

        # Count liquid strikes
        liquid_count = 0
        for i in range(total_strikes):
            oi_val = oi[i] if i < len(oi) else 0
            spread_val = spread_pct[i] if i < len(spread_pct) else 1.0
            if oi_val >= self.min_oi and spread_val <= self.max_spread_pct:
                liquid_count += 1

        # Check call price monotonicity (calls decrease with strike for same expiry)
        has_arb = False
        if len(call_prices) >= 2:
            for i in range(len(call_prices) - 1):
                if call_prices[i + 1] > call_prices[i]:
                    has_arb = True
                    warnings_list.append(
                        f"Call price arbitrage: C({strikes[i + 1]})={call_prices[i + 1]:.4f} "
                        f"> C({strikes[i]})={call_prices[i]:.4f}"
                    )
                    break

        # Put-call parity check (if both call and put prices exist)
        pcp_max_error: Optional[float] = None
        if "put_price" in columns and "call_price" in columns:
            put_prices = table.column("put_price").to_pylist()
            # Get forward/spot if available
            if "forward" in columns:
                forwards = table.column("forward").to_pylist()
            else:
                forwards = None

            if forwards is not None and len(forwards) > 0:
                errors = []
                r = 0.05  # default risk-free rate
                T = 0.25  # default expiry
                for i in range(min(len(call_prices), len(put_prices), len(strikes))):
                    F = forwards[i] if i < len(forwards) else forwards[0]
                    K = strikes[i]
                    C = call_prices[i]
                    P = put_prices[i]
                    # Put-call parity: C - P ~ (F - K) * exp(-rT)
                    parity_diff = abs(
                        (C - P) - (F - K) * np.exp(-r * T)
                    )
                    errors.append(parity_diff)
                if errors:
                    pcp_max_error = max(errors)

        if liquid_count < self.min_liquid_strikes:
            warnings_list.append(
                f"Only {liquid_count} liquid strikes (need {self.min_liquid_strikes})"
            )

        return OptionsQualityReport(
            liquid_strike_count=liquid_count,
            total_strikes=total_strikes,
            has_arbitrage_violation=has_arb,
            put_call_parity_max_error=pcp_max_error,
            warnings=warnings_list,
        )

    def check_cot_freshness(
        self, market: str, current_time: datetime
    ) -> COTFreshnessReport:
        """Check COT data freshness.

        COT reports are released weekly (Friday) for data as of Tuesday.
        Staleness threshold is configurable (default 7 calendar days).

        Parameters
        ----------
        market : str
            Market identifier.
        current_time : datetime
            Reference time for freshness check.

        Returns
        -------
        COTFreshnessReport with freshness metrics.
        """
        last_update = self._get_latest_timestamp(market, "cot")

        if last_update is None:
            return COTFreshnessReport(
                as_of_date=None,
                available_at_date=None,
                days_since_update=None,
                is_stale=True,
            )

        days_since = (current_time - last_update).total_seconds() / 86400.0

        return COTFreshnessReport(
            as_of_date=last_update,
            available_at_date=last_update,  # simplified for Phase 1
            days_since_update=days_since,
            is_stale=days_since > self.cot_stale_days,
        )

    def generate_report(
        self, market: str, current_time: datetime
    ) -> QualityReport:
        """Generate a comprehensive quality report for a market.

        Runs all checks and produces a structured report with an overall
        status determination.

        Parameters
        ----------
        market : str
            Market identifier.
        current_time : datetime
            Reference time for all checks.

        Returns
        -------
        QualityReport with all check results and overall status.
        """
        staleness_alerts = self.check_staleness(market, current_time)
        options_quality = self.check_options_quality(market, current_time)
        cot_freshness = self.check_cot_freshness(market, current_time)

        # Determine overall status
        any_stale = any(a.is_stale for a in staleness_alerts)
        low_liquidity = (
            options_quality.liquid_strike_count < self.min_liquid_strikes
        )
        has_arb = options_quality.has_arbitrage_violation

        if any_stale:
            overall_status = "stale"
        elif low_liquidity or has_arb:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        report = QualityReport(
            market=market,
            timestamp=current_time,
            staleness_alerts=staleness_alerts,
            options_quality=options_quality,
            cot_freshness=cot_freshness,
            overall_status=overall_status,
        )

        logger.info(
            "quality_report_generated",
            market=market,
            overall_status=overall_status,
            stale_sources=[a.source for a in staleness_alerts if a.is_stale],
            liquid_strikes=options_quality.liquid_strike_count,
        )

        return report

    def _get_latest_timestamp(
        self, market: str, source: str
    ) -> Optional[datetime]:
        """Get the latest data timestamp for a source from the feature store.

        Uses the feature store to find the most recent available_at time
        for a given market and source. Returns None if no data exists.
        """
        features = self.feature_store.get_features_at(market, datetime.max)

        # Look for a timestamp feature for this source
        ts_key = f"{source}_last_timestamp"
        if ts_key in features:
            # Feature store stores numeric values; interpret as epoch
            epoch_val = features[ts_key]
            if epoch_val is not None:
                return datetime.fromtimestamp(epoch_val, tz=None)

        return None
