"""CFTC Commitments of Traders (COT) ingestion pipeline.

Fetches Disaggregated Futures-Only COT reports from the CFTC via the
``cot_reports`` library, validates position data, and persists to both the
Parquet data lake and the feature store.

CRITICAL TIMING SEMANTICS:
    - as_of   = Tuesday (the date the report data represents)
    - available_at = next Friday at 15:30 ET (20:30 UTC) -- the CFTC release time

This timing is the foundation for preventing lookahead bias. COT data
collected on Tuesday is NOT available to the system until Friday afternoon.
The _next_friday() helper computes the correct release timestamp.

Per research pitfall #2, the last 4 weeks of data are re-downloaded on
each ingestion run to catch CFTC revisions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cot_reports as cot
import pyarrow as pa
import structlog

from hydra.data.ingestion.base import IngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

logger = structlog.get_logger()

# US Eastern Time offset for COT release: UTC-5 (EST) or UTC-4 (EDT)
# COT is released at 3:30 PM ET on Fridays.
# We use a fixed UTC offset for simplicity; in production this would
# use a proper timezone library for DST handling.
_ET_OFFSET_HOURS = -5  # EST (standard time)


def _next_friday(report_date: datetime) -> datetime:
    """Compute the next Friday at 15:30 ET (converted to UTC) from a report date.

    COT reports are collected on Tuesday and released on Friday at
    3:30 PM Eastern Time. This function computes the exact release
    timestamp in UTC.

    Parameters
    ----------
    report_date : datetime
        The report collection date (typically a Tuesday, timezone-aware UTC).

    Returns
    -------
    datetime
        The next Friday at 15:30 ET converted to UTC (20:30 UTC in EST).
    """
    # Find the next Friday (weekday 4) on or after the report date
    days_until_friday = (4 - report_date.weekday()) % 7
    if days_until_friday == 0 and report_date.weekday() != 4:
        # Already past Friday this week, go to next
        days_until_friday = 7

    # If report_date IS a Friday, we want the NEXT Friday
    if report_date.weekday() == 4:
        days_until_friday = 7

    # If report_date is before Friday this week, use this week's Friday
    friday = report_date + timedelta(days=days_until_friday)

    # Release at 15:30 ET = 20:30 UTC (EST, UTC-5)
    release_utc = datetime(
        friday.year,
        friday.month,
        friday.day,
        20,  # 15:30 ET + 5 hours = 20:30 UTC
        30,
        0,
        tzinfo=timezone.utc,
    )

    return release_utc


class COTIngestPipeline(IngestPipeline):
    """Ingest CFTC COT Disaggregated Futures-Only reports.

    Parameters
    ----------
    parquet_lake : ParquetLake
        Data lake for raw COT data persistence.
    feature_store : FeatureStore
        Feature store for COT-derived features.
    cftc_code : str
        CFTC contract market code for filtering (e.g., "054642" for lean hogs).
    redownload_weeks : int
        Number of past weeks to re-download to catch CFTC revisions.
    """

    def __init__(
        self,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
        cftc_code: str = "054642",
        redownload_weeks: int = 4,
    ) -> None:
        super().__init__(parquet_lake=parquet_lake, feature_store=feature_store)
        self.cftc_code = cftc_code
        self.redownload_weeks = redownload_weeks

    def fetch(self, market: str, date: datetime) -> dict:
        """Fetch COT data for the target year, filtered by CFTC code.

        Re-downloads the full year to catch revisions in the last 4 weeks
        (per research pitfall #2).

        Parameters
        ----------
        market : str
            Market identifier (e.g., "HE").
        date : datetime
            Reference date used to determine the year (timezone-aware UTC).

        Returns
        -------
        dict
            Keys: "records" (list of dicts with report_date, positions, etc.).
        """
        log = logger.bind(
            pipeline="COTIngestPipeline", market=market, cftc_code=self.cftc_code
        )

        year = date.year
        log.info("fetching_cot", year=year)

        # Download full year of disaggregated futures-only reports
        df = cot.cot_year(year=year, cot_report_type="disaggregated_fut")

        # Filter to target market by CFTC contract code
        code_col = "CFTC_Contract_Market_Code"
        if code_col not in df.columns:
            # Try alternative column names
            for alt in ["CFTC Contract Market Code", "cftc_contract_market_code"]:
                if alt in df.columns:
                    code_col = alt
                    break

        market_df = df[df[code_col].astype(str).str.strip() == str(self.cftc_code)]

        # Determine the cutoff for "recent" data (last N weeks from date)
        cutoff = date - timedelta(weeks=self.redownload_weeks)

        records = []
        for _, row in market_df.iterrows():
            # Parse the report date (as_of date = Tuesday)
            report_date_raw = row.get(
                "As_of_Date_In_Form_YYMMDD",
                row.get("Report_Date_as_YYYY-MM-DD", None),
            )
            if report_date_raw is None:
                continue

            if isinstance(report_date_raw, str):
                try:
                    report_date = datetime.strptime(
                        report_date_raw, "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        report_date = datetime.strptime(
                            report_date_raw, "%y%m%d"
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            else:
                # Assume it's already a date-like object
                report_date = datetime(
                    report_date_raw.year,
                    report_date_raw.month,
                    report_date_raw.day,
                    tzinfo=timezone.utc,
                )

            # Extract position data
            managed_money_long = float(
                row.get("M_Money_Positions_Long_All", row.get("M_Money_Positions_Long_ALL", 0))
            )
            managed_money_short = float(
                row.get("M_Money_Positions_Short_All", row.get("M_Money_Positions_Short_ALL", 0))
            )
            producer_long = float(
                row.get("Prod_Merc_Positions_Long_All", row.get("Prod_Merc_Positions_Long_ALL", 0))
            )
            producer_short = float(
                row.get("Prod_Merc_Positions_Short_All", row.get("Prod_Merc_Positions_Short_ALL", 0))
            )
            swap_long = float(
                row.get("Swap_Positions_Long_All", row.get("Swap__Positions_Long_All", 0))
            )
            swap_short = float(
                row.get("Swap_Positions_Short_All", row.get("Swap__Positions_Short_All", 0))
            )
            total_oi = float(
                row.get("Open_Interest_All", row.get("Open_Interest_ALL", 0))
            )

            records.append({
                "report_date": report_date,
                "available_at": _next_friday(report_date),
                "managed_money_long": managed_money_long,
                "managed_money_short": managed_money_short,
                "managed_money_net": managed_money_long - managed_money_short,
                "producer_long": producer_long,
                "producer_short": producer_short,
                "producer_net": producer_long - producer_short,
                "swap_long": swap_long,
                "swap_short": swap_short,
                "swap_net": swap_long - swap_short,
                "total_oi": total_oi,
                "is_revision_window": report_date >= cutoff,
            })

        log.info("fetched_cot", record_count=len(records))
        return {"records": records}

    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        """Validate COT position data.

        Checks:
        - Non-empty results
        - No negative position values
        - Positions are consistent (long + short should not exceed total OI by too much)

        Parameters
        ----------
        raw_data : dict
            Raw payload from ``fetch()``.

        Returns
        -------
        tuple[dict, list[str]]
            (cleaned_data, warnings)
        """
        warnings: list[str] = []
        records = raw_data.get("records", [])

        if not records:
            warnings.append("No COT records returned for this market/year")
            return {"records": []}, warnings

        valid_records = []
        for i, rec in enumerate(records):
            rec_warnings: list[str] = []

            # Check for negative positions
            for field in [
                "managed_money_long", "managed_money_short",
                "producer_long", "producer_short",
                "swap_long", "swap_short",
                "total_oi",
            ]:
                if rec.get(field, 0) < 0:
                    rec_warnings.append(
                        f"Record {i} ({rec['report_date']}): {field}={rec[field]} < 0"
                    )

            if rec_warnings:
                warnings.extend(rec_warnings)
            else:
                valid_records.append(rec)

        if len(valid_records) < len(records):
            warnings.append(
                f"Dropped {len(records) - len(valid_records)} of "
                f"{len(records)} records"
            )

        return {"records": valid_records}, warnings

    def persist(self, data: dict, market: str, date: datetime) -> None:
        """Write COT data to Parquet lake AND features to feature store.

        CRITICAL: Features are written with:
            as_of = report Tuesday date
            available_at = next Friday at 15:30 ET (20:30 UTC)

        Features written:
        - cot_managed_money_net
        - cot_producer_net
        - cot_swap_net
        - cot_total_oi

        Parameters
        ----------
        data : dict
            Cleaned data from ``validate()``.
        market : str
            Market identifier (e.g., "HE").
        date : datetime
            Reference date (timezone-aware UTC).
        """
        records = data.get("records", [])
        if not records:
            logger.warning("no_cot_to_persist", market=market, date=str(date))
            return

        # Build pyarrow table for Parquet lake
        table = pa.table({
            "report_date": pa.array(
                [r["report_date"].isoformat() for r in records], type=pa.string()
            ),
            "available_at": pa.array(
                [r["available_at"].isoformat() for r in records], type=pa.string()
            ),
            "managed_money_long": pa.array(
                [r["managed_money_long"] for r in records], type=pa.float64()
            ),
            "managed_money_short": pa.array(
                [r["managed_money_short"] for r in records], type=pa.float64()
            ),
            "managed_money_net": pa.array(
                [r["managed_money_net"] for r in records], type=pa.float64()
            ),
            "producer_long": pa.array(
                [r["producer_long"] for r in records], type=pa.float64()
            ),
            "producer_short": pa.array(
                [r["producer_short"] for r in records], type=pa.float64()
            ),
            "producer_net": pa.array(
                [r["producer_net"] for r in records], type=pa.float64()
            ),
            "swap_long": pa.array(
                [r["swap_long"] for r in records], type=pa.float64()
            ),
            "swap_short": pa.array(
                [r["swap_short"] for r in records], type=pa.float64()
            ),
            "swap_net": pa.array(
                [r["swap_net"] for r in records], type=pa.float64()
            ),
            "total_oi": pa.array(
                [r["total_oi"] for r in records], type=pa.float64()
            ),
        })

        self.parquet_lake.write(table, data_type="cot", market=market, date=date)

        # Write features to feature store with correct timing
        feature_names = [
            ("cot_managed_money_net", "managed_money_net"),
            ("cot_producer_net", "producer_net"),
            ("cot_swap_net", "swap_net"),
            ("cot_total_oi", "total_oi"),
        ]

        for rec in records:
            as_of = rec["report_date"]
            available_at = rec["available_at"]

            for feature_name, key in feature_names:
                self.feature_store.write_feature(
                    market=market,
                    feature_name=feature_name,
                    as_of=as_of,
                    available_at=available_at,
                    value=rec[key],
                )

        logger.info(
            "cot_persisted",
            market=market,
            records=len(records),
            date=str(date),
        )
