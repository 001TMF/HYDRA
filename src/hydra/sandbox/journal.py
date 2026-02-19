"""Experiment journal for logging and querying experiment history.

Every experiment is logged with its hypothesis, config diff, results, and
promotion decision. The journal supports querying by tag, date range,
mutation type, and outcome (promotion_decision). All filters combine with
AND logic.

Storage uses SQLite with WAL mode, consistent with the FeatureStore pattern
in ``src/hydra/data/store/feature_store.py``. JSON columns (config_diff,
results, champion_metrics, tags, metadata) provide extensibility for Phase 4
agent loop integration.

Schema:
    experiments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,          -- ISO 8601 UTC
        hypothesis TEXT NOT NULL,
        mutation_type TEXT NOT NULL,
        config_diff TEXT NOT NULL,         -- JSON
        results TEXT NOT NULL,             -- JSON
        champion_metrics TEXT,             -- JSON (nullable)
        promotion_decision TEXT NOT NULL,  -- "promoted" | "rejected" | "pending"
        promotion_reason TEXT,
        run_id TEXT,                       -- MLflow run ID
        model_version INTEGER,            -- MLflow model version
        tags TEXT DEFAULT '[]',            -- JSON array
        metadata TEXT DEFAULT '{}'         -- JSON object
    )
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class ExperimentRecord:
    """A single experiment entry in the journal.

    Parameters
    ----------
    id : int | None
        Auto-assigned by the database on insert.
    created_at : str
        ISO 8601 UTC timestamp of when the experiment was created.
    hypothesis : str
        What we are testing (human-readable description).
    mutation_type : str
        Category of change: "hyperparameter", "feature_add", "feature_remove",
        "architecture", etc.
    config_diff : dict
        What changed from the champion config. Stored as JSON.
    results : dict
        All metrics from evaluation. Stored as JSON.
    champion_metrics : dict | None
        Champion metrics at time of experiment for comparison. Stored as JSON.
    promotion_decision : str
        One of: "promoted", "rejected", "pending".
    promotion_reason : str | None
        Why the experiment was promoted or rejected.
    run_id : str | None
        MLflow run ID for cross-reference.
    model_version : int | None
        MLflow model version.
    tags : list[str]
        Queryable tags. Stored as JSON array.
    metadata : dict
        Extensible escape hatch for Phase 4. Stored as JSON.
    """

    hypothesis: str
    mutation_type: str
    config_diff: dict
    results: dict
    promotion_decision: str
    id: int | None = None
    created_at: str = ""
    champion_metrics: dict | None = None
    promotion_reason: str | None = None
    run_id: str | None = None
    model_version: int | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ExperimentJournal:
    """SQLite-backed experiment journal with query layer.

    Logs every experiment with hypothesis, config diff, results, and
    promotion decision. Supports querying by tag, date range, mutation
    type, and outcome with AND-combined filters.

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
        logger.debug("experiment_journal_opened", db_path=str(self.db_path))

    def _create_tables(self) -> None:
        """Create the experiments table and indexes if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                mutation_type TEXT NOT NULL,
                config_diff TEXT NOT NULL,
                results TEXT NOT NULL,
                champion_metrics TEXT,
                promotion_decision TEXT NOT NULL,
                promotion_reason TEXT,
                run_id TEXT,
                model_version INTEGER,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_created
            ON experiments(created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_mutation
            ON experiments(mutation_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_decision
            ON experiments(promotion_decision)
        """)
        self.conn.commit()

    def log_experiment(self, record: ExperimentRecord) -> int:
        """Insert an experiment record into the journal.

        Parameters
        ----------
        record : ExperimentRecord
            The experiment to log. The ``id`` field is ignored (auto-assigned).

        Returns
        -------
        int
            The auto-generated experiment ID.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO experiments
            (created_at, hypothesis, mutation_type, config_diff, results,
             champion_metrics, promotion_decision, promotion_reason,
             run_id, model_version, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.created_at,
                record.hypothesis,
                record.mutation_type,
                json.dumps(record.config_diff),
                json.dumps(record.results),
                json.dumps(record.champion_metrics) if record.champion_metrics is not None else None,
                record.promotion_decision,
                record.promotion_reason,
                record.run_id,
                record.model_version,
                json.dumps(record.tags),
                json.dumps(record.metadata),
            ),
        )
        self.conn.commit()
        experiment_id = cursor.lastrowid
        logger.info(
            "experiment_logged",
            experiment_id=experiment_id,
            hypothesis=record.hypothesis,
            mutation_type=record.mutation_type,
            promotion_decision=record.promotion_decision,
        )
        return experiment_id

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        """Retrieve a single experiment by its ID.

        Parameters
        ----------
        experiment_id : int
            The experiment ID to look up.

        Returns
        -------
        ExperimentRecord | None
            The experiment record, or None if not found.
        """
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def query(
        self,
        tags: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        mutation_type: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
    ) -> list[ExperimentRecord]:
        """Query experiments with AND-combined filters.

        Parameters
        ----------
        tags : list[str] | None
            Filter to experiments containing ALL specified tags.
        date_from : str | None
            Inclusive lower bound on created_at (ISO 8601).
        date_to : str | None
            Inclusive upper bound on created_at (ISO 8601).
        mutation_type : str | None
            Exact match on mutation_type.
        outcome : str | None
            Exact match on promotion_decision.
        limit : int
            Maximum number of results (default 100).

        Returns
        -------
        list[ExperimentRecord]
            Matching experiments ordered by created_at DESC.
        """
        conditions: list[str] = []
        params: list[object] = []

        if tags:
            for tag in tags:
                conditions.append('tags LIKE ?')
                params.append(f'%"{tag}"%')

        if date_from is not None:
            conditions.append("created_at >= ?")
            params.append(date_from)

        if date_to is not None:
            conditions.append("created_at <= ?")
            params.append(date_to)

        if mutation_type is not None:
            conditions.append("mutation_type = ?")
            params.append(mutation_type)

        if outcome is not None:
            conditions.append("promotion_decision = ?")
            params.append(outcome)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM experiments
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Return the total number of experiments in the journal.

        Returns
        -------
        int
            Total experiment count.
        """
        row = self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        return row[0]

    def _row_to_record(self, row: tuple) -> ExperimentRecord:
        """Deserialize a database row into an ExperimentRecord.

        Parameters
        ----------
        row : tuple
            A row from the experiments table with all columns in schema order.

        Returns
        -------
        ExperimentRecord
            The deserialized experiment record.
        """
        return ExperimentRecord(
            id=row[0],
            created_at=row[1],
            hypothesis=row[2],
            mutation_type=row[3],
            config_diff=json.loads(row[4]),
            results=json.loads(row[5]),
            champion_metrics=json.loads(row[6]) if row[6] is not None else None,
            promotion_decision=row[7],
            promotion_reason=row[8],
            run_id=row[9],
            model_version=row[10],
            tags=json.loads(row[11]) if row[11] else [],
            metadata=json.loads(row[12]) if row[12] else {},
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("experiment_journal_closed", db_path=str(self.db_path))
