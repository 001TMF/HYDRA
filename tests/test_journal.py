"""Tests for ExperimentJournal logging and querying.

Covers: logging, retrieval, all 4 query filter types (tag, date range,
mutation type, outcome), combined filters, count, and metadata extensibility.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord


@pytest.fixture
def journal(tmp_path):
    """Create an ExperimentJournal backed by a temp SQLite database."""
    j = ExperimentJournal(tmp_path / "journal.db")
    yield j
    j.close()


def make_record(**overrides) -> ExperimentRecord:
    """Factory that returns an ExperimentRecord with sensible defaults."""
    defaults = dict(
        hypothesis="Test hypothesis",
        mutation_type="hyperparameter",
        config_diff={"lr": 0.05},
        results={"sharpe": 0.8},
        promotion_decision="rejected",
        tags=["test"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


class TestExperimentJournal:
    """Tests for ExperimentJournal."""

    def test_log_and_retrieve(self, journal: ExperimentJournal):
        """Log a record, retrieve by id, assert all fields match."""
        record = make_record(
            hypothesis="Increase learning rate",
            mutation_type="hyperparameter",
            config_diff={"lr": 0.1, "old_lr": 0.05},
            results={"sharpe": 1.2, "max_dd": 0.15},
            champion_metrics={"sharpe": 0.9},
            promotion_decision="promoted",
            promotion_reason="Higher Sharpe ratio",
            run_id="mlflow-run-123",
            model_version=3,
            tags=["lr_experiment", "urgent"],
            metadata={"source": "agent_v2"},
        )
        exp_id = journal.log_experiment(record)
        assert exp_id is not None
        assert exp_id >= 1

        retrieved = journal.get_experiment(exp_id)
        assert retrieved is not None
        assert retrieved.id == exp_id
        assert retrieved.hypothesis == "Increase learning rate"
        assert retrieved.mutation_type == "hyperparameter"
        assert retrieved.config_diff == {"lr": 0.1, "old_lr": 0.05}
        assert retrieved.results == {"sharpe": 1.2, "max_dd": 0.15}
        assert retrieved.champion_metrics == {"sharpe": 0.9}
        assert retrieved.promotion_decision == "promoted"
        assert retrieved.promotion_reason == "Higher Sharpe ratio"
        assert retrieved.run_id == "mlflow-run-123"
        assert retrieved.model_version == 3
        assert retrieved.tags == ["lr_experiment", "urgent"]
        assert retrieved.metadata == {"source": "agent_v2"}

    def test_query_by_mutation_type(self, journal: ExperimentJournal):
        """Log 3 records: 2 hyperparameter, 1 feature_add. Query by mutation_type."""
        journal.log_experiment(make_record(mutation_type="hyperparameter"))
        journal.log_experiment(make_record(mutation_type="hyperparameter"))
        journal.log_experiment(make_record(mutation_type="feature_add"))

        results = journal.query(mutation_type="hyperparameter")
        assert len(results) == 2
        assert all(r.mutation_type == "hyperparameter" for r in results)

    def test_query_by_outcome(self, journal: ExperimentJournal):
        """Log 3 records: 1 promoted, 2 rejected. Query by outcome."""
        journal.log_experiment(make_record(promotion_decision="promoted"))
        journal.log_experiment(make_record(promotion_decision="rejected"))
        journal.log_experiment(make_record(promotion_decision="rejected"))

        results = journal.query(outcome="promoted")
        assert len(results) == 1
        assert results[0].promotion_decision == "promoted"

    def test_query_by_date_range(self, journal: ExperimentJournal):
        """Log 3 records with distinct dates. Query by date range."""
        journal.log_experiment(make_record(created_at="2024-01-01T00:00:00"))
        journal.log_experiment(make_record(created_at="2024-06-15T12:00:00"))
        journal.log_experiment(make_record(created_at="2024-12-31T23:59:59"))

        results = journal.query(
            date_from="2024-06-01T00:00:00",
            date_to="2024-07-01T00:00:00",
        )
        assert len(results) == 1
        assert results[0].created_at == "2024-06-15T12:00:00"

    def test_query_by_tag(self, journal: ExperimentJournal):
        """Log 2 records with different tags. Query by tag."""
        journal.log_experiment(make_record(tags=["drift", "urgent"]))
        journal.log_experiment(make_record(tags=["routine"]))

        results = journal.query(tags=["drift"])
        assert len(results) == 1
        assert "drift" in results[0].tags

    def test_query_combined_filters(self, journal: ExperimentJournal):
        """Query with mutation_type + outcome combined for AND logic."""
        # Record 1: hyperparameter + promoted
        journal.log_experiment(
            make_record(mutation_type="hyperparameter", promotion_decision="promoted")
        )
        # Record 2: hyperparameter + rejected
        journal.log_experiment(
            make_record(mutation_type="hyperparameter", promotion_decision="rejected")
        )
        # Record 3: feature_add + promoted
        journal.log_experiment(
            make_record(mutation_type="feature_add", promotion_decision="promoted")
        )
        # Record 4: feature_add + rejected
        journal.log_experiment(
            make_record(mutation_type="feature_add", promotion_decision="rejected")
        )

        # AND: hyperparameter AND promoted
        results = journal.query(mutation_type="hyperparameter", outcome="promoted")
        assert len(results) == 1
        assert results[0].mutation_type == "hyperparameter"
        assert results[0].promotion_decision == "promoted"

    def test_count(self, journal: ExperimentJournal):
        """Log 5 records and assert count returns 5."""
        for i in range(5):
            journal.log_experiment(make_record(hypothesis=f"Experiment {i}"))

        assert journal.count() == 5

    def test_metadata_escape_hatch(self, journal: ExperimentJournal):
        """Log record with metadata dict and verify round-trip."""
        metadata = {"autonomy_level": "supervised", "parent_id": 42}
        record = make_record(metadata=metadata)
        exp_id = journal.log_experiment(record)

        retrieved = journal.get_experiment(exp_id)
        assert retrieved is not None
        assert retrieved.metadata == {"autonomy_level": "supervised", "parent_id": 42}

    def test_get_nonexistent_experiment(self, journal: ExperimentJournal):
        """Querying a nonexistent ID returns None."""
        assert journal.get_experiment(999) is None

    def test_query_no_filters_returns_all(self, journal: ExperimentJournal):
        """Query with no filters returns all experiments."""
        journal.log_experiment(make_record())
        journal.log_experiment(make_record())
        journal.log_experiment(make_record())

        results = journal.query()
        assert len(results) == 3
