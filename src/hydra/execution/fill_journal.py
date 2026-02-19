"""Fill journal for logging execution fills with slippage tracking.

Every fill from IB is logged with timestamp, prices, predicted slippage
(from ``estimate_slippage()``), actual slippage, and market conditions
(volume, spread, latency).  The journal supports querying by symbol and
date range, and can extract (predicted, actual) slippage pairs for
reconciliation by the ``SlippageReconciler``.

Storage uses SQLite with WAL mode, consistent with the ExperimentJournal
pattern in ``src/hydra/sandbox/journal.py``.

Schema:
    fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,          -- ISO 8601 UTC
        symbol TEXT NOT NULL,
        direction INTEGER NOT NULL,       -- +1 long, -1 short
        n_contracts INTEGER NOT NULL,
        order_price REAL NOT NULL,        -- mid-price at order entry
        fill_price REAL NOT NULL,         -- actual fill price from IB
        predicted_slippage REAL NOT NULL, -- from estimate_slippage()
        actual_slippage REAL NOT NULL,    -- abs(fill_price - order_price)
        volume_at_fill REAL NOT NULL,     -- market volume at time of fill
        spread_at_fill REAL NOT NULL,     -- bid-ask spread at time of fill
        fill_latency_ms REAL NOT NULL,    -- time from order to fill
        order_id INTEGER                  -- IB order ID (nullable)
    )
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class FillRecord:
    """A single fill entry in the journal.

    Parameters
    ----------
    timestamp : str
        ISO 8601 UTC timestamp of the fill.
    symbol : str
        Instrument symbol (e.g., "ZO", "HE").
    direction : int
        +1 for long, -1 for short.
    n_contracts : int
        Number of contracts filled.
    order_price : float
        Mid-price at order entry.
    fill_price : float
        Actual fill price from IB.
    predicted_slippage : float
        Slippage predicted by ``estimate_slippage()``.
    actual_slippage : float
        Absolute difference between fill and order price per contract.
    volume_at_fill : float
        Market volume at the time of fill.
    spread_at_fill : float
        Bid-ask spread at the time of fill.
    fill_latency_ms : float
        Time from order submission to fill in milliseconds.
    order_id : int | None
        IB order ID, if available.
    id : int | None
        Auto-assigned by the database on insert.
    """

    timestamp: str
    symbol: str
    direction: int
    n_contracts: int
    order_price: float
    fill_price: float
    predicted_slippage: float
    actual_slippage: float
    volume_at_fill: float
    spread_at_fill: float
    fill_latency_ms: float
    order_id: int | None = None
    id: int | None = None


class FillJournal:
    """SQLite-backed fill journal with slippage tracking and query layer.

    Logs every fill with timestamps, prices, predicted/actual slippage,
    and market conditions.  Supports querying by symbol and date range,
    and extracting slippage pairs for reconciliation.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.debug("fill_journal_opened", db_path=str(self.db_path))

    def _create_tables(self) -> None:
        """Create the fills table and indexes if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                n_contracts INTEGER NOT NULL,
                order_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                predicted_slippage REAL NOT NULL,
                actual_slippage REAL NOT NULL,
                volume_at_fill REAL NOT NULL,
                spread_at_fill REAL NOT NULL,
                fill_latency_ms REAL NOT NULL,
                order_id INTEGER
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fills_timestamp
            ON fills(timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fills_symbol
            ON fills(symbol)
        """)
        self.conn.commit()

    def log_fill(self, record: FillRecord) -> int:
        """Insert a fill record into the journal.

        Parameters
        ----------
        record : FillRecord
            The fill to log.  The ``id`` field is ignored (auto-assigned).

        Returns
        -------
        int
            The auto-generated fill ID.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO fills
            (timestamp, symbol, direction, n_contracts, order_price,
             fill_price, predicted_slippage, actual_slippage,
             volume_at_fill, spread_at_fill, fill_latency_ms, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.symbol,
                record.direction,
                record.n_contracts,
                record.order_price,
                record.fill_price,
                record.predicted_slippage,
                record.actual_slippage,
                record.volume_at_fill,
                record.spread_at_fill,
                record.fill_latency_ms,
                record.order_id,
            ),
        )
        self.conn.commit()
        fill_id = cursor.lastrowid
        logger.info(
            "fill_logged",
            fill_id=fill_id,
            symbol=record.symbol,
            direction=record.direction,
            predicted_slippage=record.predicted_slippage,
            actual_slippage=record.actual_slippage,
        )
        return fill_id

    def get_fills(
        self,
        symbol: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[FillRecord]:
        """Query fills with AND-combined filters.

        Parameters
        ----------
        symbol : str | None
            Filter by instrument symbol (exact match).
        since : str | None
            Inclusive lower bound on timestamp (ISO 8601).
        until : str | None
            Inclusive upper bound on timestamp (ISO 8601).
        limit : int
            Maximum number of results (default 1000).

        Returns
        -------
        list[FillRecord]
            Matching fills ordered by timestamp DESC.
        """
        conditions: list[str] = []
        params: list[object] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM fills
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_slippage_pairs(
        self,
        symbol: str | None = None,
        since: str | None = None,
    ) -> list[tuple[float, float]]:
        """Return (predicted_slippage, actual_slippage) pairs for reconciliation.

        Parameters
        ----------
        symbol : str | None
            Filter by instrument symbol.
        since : str | None
            Inclusive lower bound on timestamp (ISO 8601).

        Returns
        -------
        list[tuple[float, float]]
            List of (predicted_slippage, actual_slippage) pairs.
        """
        conditions: list[str] = []
        params: list[object] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT predicted_slippage, actual_slippage FROM fills
            {where_clause}
            ORDER BY timestamp DESC
        """

        rows = self.conn.execute(sql, params).fetchall()
        return [(row[0], row[1]) for row in rows]

    def count(self) -> int:
        """Return the total number of fills in the journal.

        Returns
        -------
        int
            Total fill count.
        """
        row = self.conn.execute("SELECT COUNT(*) FROM fills").fetchone()
        return row[0]

    def _row_to_record(self, row: tuple) -> FillRecord:
        """Deserialize a database row into a FillRecord.

        Parameters
        ----------
        row : tuple
            A row from the fills table with all columns in schema order.

        Returns
        -------
        FillRecord
            The deserialized fill record.
        """
        return FillRecord(
            id=row[0],
            timestamp=row[1],
            symbol=row[2],
            direction=row[3],
            n_contracts=row[4],
            order_price=row[5],
            fill_price=row[6],
            predicted_slippage=row[7],
            actual_slippage=row[8],
            volume_at_fill=row[9],
            spread_at_fill=row[10],
            fill_latency_ms=row[11],
            order_id=row[12],
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("fill_journal_closed", db_path=str(self.db_path))
