"""Point-in-time correct feature store with as_of/available_at semantics.

Every feature has two temporal dimensions:
    - as_of:        When the data *represents* (e.g., Tuesday for COT collection).
    - available_at: When the data *became available* (e.g., Friday for COT release).

All queries filter by ``available_at <= query_time`` to prevent lookahead bias.
This is the foundational invariant that protects every downstream backtest and
model training run from seeing future information.

Schema:
    features(
        market TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        as_of TEXT NOT NULL,           -- ISO 8601 UTC
        available_at TEXT NOT NULL,     -- ISO 8601 UTC
        value REAL,
        quality TEXT DEFAULT 'normal',
        PRIMARY KEY (market, feature_name, as_of)
    )
    INDEX idx_features_pit ON features (market, feature_name, available_at)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


class FeatureStore:
    """Point-in-time correct feature store backed by SQLite.

    Designed to be a drop-in replaceable backend; the query interface is
    intentionally simple so that migrating to TimescaleDB in Phase 3+ is
    a backend swap, not a rewrite.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the features table and indexes if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                market TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                as_of TEXT NOT NULL,
                available_at TEXT NOT NULL,
                value REAL,
                quality TEXT DEFAULT 'normal',
                PRIMARY KEY (market, feature_name, as_of)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_pit
            ON features (market, feature_name, available_at)
        """)
        self.conn.commit()

    def write_feature(
        self,
        market: str,
        feature_name: str,
        as_of: datetime,
        available_at: datetime,
        value: float,
        quality: str = "normal",
    ) -> None:
        """Write a single feature observation.

        Parameters
        ----------
        market : str
            Market identifier (e.g., "HE").
        feature_name : str
            Name of the feature (e.g., "cot_managed_money_net").
        as_of : datetime
            When the underlying data represents (timezone-aware UTC).
        available_at : datetime
            When the data became available for queries (timezone-aware UTC).
        value : float
            Numeric feature value.
        quality : str
            Quality flag: "normal", "degraded", "stale", or "missing".
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO features
            (market, feature_name, as_of, available_at, value, quality)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                market,
                feature_name,
                as_of.isoformat(),
                available_at.isoformat(),
                value,
                quality,
            ),
        )
        self.conn.commit()
        logger.debug(
            "feature_written",
            market=market,
            feature_name=feature_name,
            as_of=as_of.isoformat(),
            available_at=available_at.isoformat(),
        )

    def get_features_at(
        self, market: str, query_time: datetime
    ) -> dict[str, float]:
        """Get the latest available value for each feature as of query_time.

        This is the point-in-time correct query that prevents lookahead bias.
        Only features whose ``available_at`` is on or before ``query_time``
        are returned, and for each feature name the most recent ``as_of``
        value (among those available) is selected.

        Parameters
        ----------
        market : str
            Market identifier.
        query_time : datetime
            The point-in-time to query (timezone-aware UTC).

        Returns
        -------
        dict[str, float]
            Mapping of feature_name -> value for all available features.
        """
        rows = self.conn.execute(
            """
            SELECT f.feature_name, f.value
            FROM features f
            INNER JOIN (
                SELECT feature_name, MAX(as_of) AS max_as_of
                FROM features
                WHERE market = ?
                  AND available_at <= ?
                GROUP BY feature_name
            ) latest
            ON f.feature_name = latest.feature_name
               AND f.as_of = latest.max_as_of
            WHERE f.market = ?
            """,
            (market, query_time.isoformat(), market),
        ).fetchall()
        return {name: val for name, val in rows}

    def get_feature_history(
        self,
        market: str,
        feature_name: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Get all feature values in a date range for debugging/analysis.

        Parameters
        ----------
        market : str
            Market identifier.
        feature_name : str
            Name of the feature.
        start : datetime
            Inclusive start of the as_of range (timezone-aware UTC).
        end : datetime
            Inclusive end of the as_of range (timezone-aware UTC).

        Returns
        -------
        list[dict]
            List of dicts with keys: as_of, available_at, value, quality.
        """
        rows = self.conn.execute(
            """
            SELECT as_of, available_at, value, quality
            FROM features
            WHERE market = ?
              AND feature_name = ?
              AND as_of >= ?
              AND as_of <= ?
            ORDER BY as_of
            """,
            (
                market,
                feature_name,
                start.isoformat(),
                end.isoformat(),
            ),
        ).fetchall()
        return [
            {
                "as_of": row[0],
                "available_at": row[1],
                "value": row[2],
                "quality": row[3],
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("feature_store_closed", db_path=str(self.db_path))
