"""Tests for the point-in-time correct feature store.

The critical test in this module is ``test_point_in_time_prevents_lookahead``:
COT data collected on Tuesday with available_at=Friday must NOT be visible
when querying on Wednesday.  This is the foundational invariant that prevents
lookahead bias in every downstream backtest.
"""

from datetime import datetime, timezone

import pytest

from hydra.data.store.feature_store import FeatureStore


@pytest.fixture
def store(tmp_feature_db) -> FeatureStore:
    """Create a FeatureStore instance backed by a temporary database."""
    fs = FeatureStore(tmp_feature_db)
    yield fs
    fs.close()


class TestFeatureStoreBasic:
    """Basic write/read roundtrip and storage verification."""

    def test_write_and_read_feature(self, store: FeatureStore) -> None:
        """A feature written can be read back at or after its available_at time."""
        as_of = datetime(2026, 2, 10, tzinfo=timezone.utc)      # Tuesday
        available_at = datetime(2026, 2, 10, tzinfo=timezone.utc)  # same day

        store.write_feature(
            market="HE",
            feature_name="futures_close",
            as_of=as_of,
            available_at=available_at,
            value=65.50,
        )

        # Query at exactly available_at -- should be returned
        features = store.get_features_at(
            market="HE",
            query_time=datetime(2026, 2, 10, tzinfo=timezone.utc),
        )
        assert "futures_close" in features
        assert features["futures_close"] == 65.50

    def test_quality_flag_stored(self, store: FeatureStore) -> None:
        """The quality flag is persisted and retrievable via history."""
        as_of = datetime(2026, 2, 10, tzinfo=timezone.utc)
        available_at = datetime(2026, 2, 10, tzinfo=timezone.utc)

        store.write_feature(
            market="HE",
            feature_name="atm_iv",
            as_of=as_of,
            available_at=available_at,
            value=0.32,
            quality="degraded",
        )

        history = store.get_feature_history(
            market="HE",
            feature_name="atm_iv",
            start=as_of,
            end=as_of,
        )
        assert len(history) == 1
        assert history[0]["quality"] == "degraded"
        assert history[0]["value"] == 0.32


class TestFeatureStoreLookaheadPrevention:
    """Verify that the feature store prevents lookahead bias.

    This is the most important test class in the entire project.
    """

    def test_point_in_time_prevents_lookahead(
        self, store: FeatureStore
    ) -> None:
        """COT data: as_of=Tuesday, available_at=Friday.

        Querying on Wednesday must NOT return it.
        Querying on Saturday must return it.
        """
        tuesday = datetime(2026, 2, 10, tzinfo=timezone.utc)
        friday = datetime(2026, 2, 13, tzinfo=timezone.utc)
        wednesday = datetime(2026, 2, 11, tzinfo=timezone.utc)
        saturday = datetime(2026, 2, 14, tzinfo=timezone.utc)

        store.write_feature(
            market="HE",
            feature_name="cot_managed_money_net",
            as_of=tuesday,
            available_at=friday,
            value=12345.0,
        )

        # Wednesday: data exists but is NOT yet available
        features_wed = store.get_features_at(market="HE", query_time=wednesday)
        assert "cot_managed_money_net" not in features_wed, (
            "COT data with available_at=Friday must NOT be returned on Wednesday!"
        )

        # Saturday: data IS available
        features_sat = store.get_features_at(market="HE", query_time=saturday)
        assert "cot_managed_money_net" in features_sat
        assert features_sat["cot_managed_money_net"] == 12345.0

    def test_unavailable_before_available_at(
        self, store: FeatureStore
    ) -> None:
        """Feature is invisible at any time before its available_at."""
        as_of = datetime(2026, 2, 10, tzinfo=timezone.utc)
        available_at = datetime(2026, 2, 13, 15, 30, tzinfo=timezone.utc)

        store.write_feature(
            market="HE",
            feature_name="cot_producer_net",
            as_of=as_of,
            available_at=available_at,
            value=9999.0,
        )

        # One second before release
        before = datetime(2026, 2, 13, 15, 29, 59, tzinfo=timezone.utc)
        assert "cot_producer_net" not in store.get_features_at(
            market="HE", query_time=before
        )

        # At exact release time
        at = available_at
        assert "cot_producer_net" in store.get_features_at(
            market="HE", query_time=at
        )


class TestFeatureStoreLatestValue:
    """Verify that the store returns the latest available value per feature."""

    def test_latest_value_returned(self, store: FeatureStore) -> None:
        """When multiple as_of values exist, the most recent available is returned."""
        # Week 1: as_of=Feb 3, available_at=Feb 6
        store.write_feature(
            market="HE",
            feature_name="cot_managed_money_net",
            as_of=datetime(2026, 2, 3, tzinfo=timezone.utc),
            available_at=datetime(2026, 2, 6, tzinfo=timezone.utc),
            value=10000.0,
        )

        # Week 2: as_of=Feb 10, available_at=Feb 13
        store.write_feature(
            market="HE",
            feature_name="cot_managed_money_net",
            as_of=datetime(2026, 2, 10, tzinfo=timezone.utc),
            available_at=datetime(2026, 2, 13, tzinfo=timezone.utc),
            value=15000.0,
        )

        # Query on Feb 7 (Saturday after week 1): should see week 1 value
        features_feb7 = store.get_features_at(
            market="HE",
            query_time=datetime(2026, 2, 7, tzinfo=timezone.utc),
        )
        assert features_feb7["cot_managed_money_net"] == 10000.0

        # Query on Feb 14 (Saturday after week 2): should see week 2 value
        features_feb14 = store.get_features_at(
            market="HE",
            query_time=datetime(2026, 2, 14, tzinfo=timezone.utc),
        )
        assert features_feb14["cot_managed_money_net"] == 15000.0

    def test_multiple_features_returned(self, store: FeatureStore) -> None:
        """get_features_at returns all features available at query_time."""
        now = datetime(2026, 2, 10, tzinfo=timezone.utc)

        store.write_feature(
            market="HE",
            feature_name="futures_close",
            as_of=now,
            available_at=now,
            value=65.50,
        )
        store.write_feature(
            market="HE",
            feature_name="atm_iv",
            as_of=now,
            available_at=now,
            value=0.28,
        )

        features = store.get_features_at(market="HE", query_time=now)
        assert len(features) == 2
        assert features["futures_close"] == 65.50
        assert features["atm_iv"] == 0.28
