"""Tests for MarketReplayEngine: bar-by-bar replay with volume-adaptive slippage.

Covers:
- Basic replay with synthetic data
- Volume-adaptive slippage variation (SBOX-01 core requirement)
- Observer callback system
- Empty data edge case
- Metric reasonableness
"""

import numpy as np
import pytest

from hydra.sandbox.replay import MarketReplayEngine, ReplayResult, TradeEvent


class MockModel:
    """Simple mock model for testing replay engine."""

    def __init__(self, proba: float = 0.6):
        self._proba = proba

    def predict_proba(self, X):
        return np.full(len(X), self._proba)


class TestReplayBasicRun:
    """Test basic replay execution with synthetic data."""

    def test_replay_basic_run(self):
        """Run replay with 100 bars of synthetic data, verify result structure."""
        np.random.seed(42)
        n_bars = 100
        n_features = 5

        features = np.random.randn(n_bars, n_features)
        # Random walk prices starting at 100
        price_changes = np.random.randn(n_bars) * 0.01
        prices = 100.0 * np.cumprod(1 + price_changes)
        volumes = np.full(n_bars, 5000.0)
        spreads = np.full(n_bars, 0.5)

        model = MockModel(proba=0.6)
        engine = MarketReplayEngine()
        result = engine.replay(model, features, prices, volumes, spreads)

        assert isinstance(result, ReplayResult)
        assert result.n_trades > 0
        assert len(result.equity_curve) == n_bars
        assert len(result.daily_returns) == n_bars
        assert len(result.trade_log) == result.n_trades
        assert result.total_return != 0.0  # should have some return


class TestSlippageVariation:
    """Test that slippage varies with volume (SBOX-01 core requirement)."""

    def test_slippage_varies_with_volume(self):
        """Low-volume run should have higher slippage than high-volume run."""
        np.random.seed(42)
        n_bars = 50
        n_features = 5

        features = np.random.randn(n_bars, n_features)
        price_changes = np.random.randn(n_bars) * 0.01
        prices = 100.0 * np.cumprod(1 + price_changes)
        spreads = np.full(n_bars, 0.5)

        model = MockModel(proba=0.6)

        # High volume run
        volumes_high = np.full(n_bars, 10000.0)
        engine_high = MarketReplayEngine()
        result_high = engine_high.replay(model, features, prices, volumes_high, spreads)

        # Low volume run (same seed, same features/prices/spreads)
        volumes_low = np.full(n_bars, 100.0)
        engine_low = MarketReplayEngine()
        result_low = engine_low.replay(model, features, prices, volumes_low, spreads)

        # Both runs should produce trades
        assert result_high.n_trades > 0, "High-volume run should produce trades"
        # Low volume may have fewer trades due to volume cap,
        # but if it has trades, slippage should be higher per contract
        if result_low.n_trades > 0:
            avg_slippage_high = np.mean(
                [t.slippage_per_contract for t in result_high.trade_log]
            )
            avg_slippage_low = np.mean(
                [t.slippage_per_contract for t in result_low.trade_log]
            )
            assert avg_slippage_low > avg_slippage_high, (
                f"Low volume slippage ({avg_slippage_low:.6f}) should be > "
                f"high volume slippage ({avg_slippage_high:.6f})"
            )

        # Also verify total slippage difference if both have trades
        if result_low.n_trades > 0 and result_high.n_trades > 0:
            total_slippage_high = sum(
                t.slippage_per_contract * t.n_contracts
                for t in result_high.trade_log
            )
            total_slippage_low = sum(
                t.slippage_per_contract * t.n_contracts
                for t in result_low.trade_log
            )
            # Per-contract slippage should be higher for low volume
            per_contract_high = total_slippage_high / sum(
                t.n_contracts for t in result_high.trade_log
            )
            per_contract_low = total_slippage_low / sum(
                t.n_contracts for t in result_low.trade_log
            )
            assert per_contract_low > per_contract_high


class TestCallbackSystem:
    """Test observer callback system."""

    def test_callback_receives_events(self):
        """Registered callback should receive all trade events."""
        np.random.seed(42)
        n_bars = 50
        n_features = 5

        features = np.random.randn(n_bars, n_features)
        price_changes = np.random.randn(n_bars) * 0.01
        prices = 100.0 * np.cumprod(1 + price_changes)
        volumes = np.full(n_bars, 5000.0)
        spreads = np.full(n_bars, 0.5)

        model = MockModel(proba=0.6)
        engine = MarketReplayEngine()

        # Register callback
        received_events: list[TradeEvent] = []
        engine.add_callback(lambda event: received_events.append(event))

        result = engine.replay(model, features, prices, volumes, spreads)

        # Callback list length should equal n_trades
        assert len(received_events) == result.n_trades

        # Each event should have all required fields
        for event in received_events:
            assert isinstance(event, TradeEvent)
            assert isinstance(event.bar_idx, int)
            assert event.direction in (1, -1)
            assert isinstance(event.n_contracts, int)
            assert event.n_contracts > 0
            assert isinstance(event.price, float)
            assert isinstance(event.volume, float)
            assert isinstance(event.spread, float)
            assert isinstance(event.slippage_per_contract, float)
            assert event.slippage_per_contract >= 0
            assert isinstance(event.raw_return, float)
            assert isinstance(event.net_return, float)
            assert isinstance(event.capital_after, float)
            assert event.capital_after > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_replay_empty_data(self):
        """Empty arrays should return zero-trade result."""
        model = MockModel(proba=0.6)
        engine = MarketReplayEngine()

        result = engine.replay(
            model,
            features=np.array([]).reshape(0, 5),
            prices=np.array([]),
            volumes=np.array([]),
            spreads=np.array([]),
        )

        assert result.n_trades == 0
        assert result.sharpe_ratio == 0.0
        assert result.total_return == 0.0
        assert result.max_drawdown == 0.0
        assert result.hit_rate == 0.0
        assert len(result.equity_curve) == 0
        assert len(result.trade_log) == 0
        assert len(result.daily_returns) == 0


class TestMetricsReasonableness:
    """Test that computed metrics are within reasonable bounds."""

    def test_replay_metrics_reasonable(self):
        """Run with 200+ bars, verify metrics are reasonable."""
        np.random.seed(42)
        n_bars = 250
        n_features = 5

        features = np.random.randn(n_bars, n_features)
        price_changes = np.random.randn(n_bars) * 0.005
        prices = 100.0 * np.cumprod(1 + price_changes)
        volumes = np.random.uniform(3000, 8000, n_bars)
        spreads = np.random.uniform(0.3, 0.8, n_bars)

        # Use higher proba and aggressive sizing to ensure enough trades
        # for meaningful metric computation
        model = MockModel(proba=0.7)
        engine = MarketReplayEngine(config={
            "max_volume_pct": 0.05,
            "max_position_pct": 0.20,
            "kelly_fraction": 0.8,
        })
        result = engine.replay(model, features, prices, volumes, spreads)

        # Sharpe ratio should be finite
        assert np.isfinite(result.sharpe_ratio), (
            f"Sharpe ratio should be finite, got {result.sharpe_ratio}"
        )

        # Max drawdown should be <= 0 (negative or zero)
        assert result.max_drawdown <= 0, (
            f"Max drawdown should be <= 0, got {result.max_drawdown}"
        )

        # Hit rate should be between 0 and 1
        assert 0 <= result.hit_rate <= 1, (
            f"Hit rate should be in [0, 1], got {result.hit_rate}"
        )

        # Equity curve should have correct length
        assert len(result.equity_curve) == n_bars

        # Total return should be finite
        assert np.isfinite(result.total_return), (
            f"Total return should be finite, got {result.total_return}"
        )

        # Trade log should have trades (Kelly sizing + volume cap may limit count)
        assert result.n_trades > 0, (
            f"Expected trades with 250 bars, got {result.n_trades}"
        )
