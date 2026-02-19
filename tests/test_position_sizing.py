"""Tests for fractional Kelly position sizing and volume-capped positions."""

import pytest

from hydra.risk.position_sizing import fractional_kelly, volume_capped_position


class TestFractionalKelly:
    """Tests for fractional_kelly function."""

    def test_negative_kelly_returns_zero(self):
        """win_prob < loss_prob with small b -> kelly <= 0 -> return 0."""
        result = fractional_kelly(
            win_prob=0.3,
            avg_win=1.0,
            avg_loss=1.0,
        )
        assert result == 0.0

    def test_half_kelly_default(self):
        """fraction=0.5 (default) produces half of full Kelly."""
        # Full Kelly: f* = (p*b - q) / b
        # p=0.6, avg_win=2.0, avg_loss=1.0: b=2.0, q=0.4
        # f* = (0.6*2.0 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.8/2.0 = 0.4
        # Half Kelly: 0.5 * 0.4 = 0.2
        result = fractional_kelly(
            win_prob=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            fraction=0.5,
            max_position_pct=1.0,  # uncapped to test formula
        )
        assert result == pytest.approx(0.2)

    def test_capped_at_max_position_pct(self):
        """Even large edge capped at max_position_pct (default 10%)."""
        # Very high edge: p=0.9, avg_win=10, avg_loss=1 -> b=10, q=0.1
        # f* = (0.9*10 - 0.1)/10 = 8.9/10 = 0.89
        # Half Kelly: 0.445 -- exceeds 10% cap
        result = fractional_kelly(
            win_prob=0.9,
            avg_win=10.0,
            avg_loss=1.0,
            fraction=0.5,
            max_position_pct=0.10,
        )
        assert result == pytest.approx(0.10)

    def test_zero_win_prob_returns_zero(self):
        """win_prob=0 -> 0."""
        result = fractional_kelly(
            win_prob=0.0,
            avg_win=2.0,
            avg_loss=1.0,
        )
        assert result == 0.0

    def test_full_kelly(self):
        """fraction=1.0 returns full Kelly (uncapped if below max)."""
        # p=0.55, avg_win=1.5, avg_loss=1.0: b=1.5, q=0.45
        # f* = (0.55*1.5 - 0.45)/1.5 = (0.825 - 0.45)/1.5 = 0.375/1.5 = 0.25
        result = fractional_kelly(
            win_prob=0.55,
            avg_win=1.5,
            avg_loss=1.0,
            fraction=1.0,
            max_position_pct=1.0,  # uncapped to test formula
        )
        assert result == pytest.approx(0.25)

    def test_zero_avg_loss_returns_zero(self):
        """avg_loss=0 (edge case) -> return 0 (avoid division by zero)."""
        result = fractional_kelly(
            win_prob=0.6,
            avg_win=2.0,
            avg_loss=0.0,
        )
        assert result == 0.0


class TestVolumeCappedPosition:
    """Tests for volume_capped_position function."""

    def test_volume_capped_position_integer(self):
        """Returns integer contracts, rounded down."""
        result = volume_capped_position(
            kelly_pct=0.05,
            capital=100_000,
            contract_value=1_000,
            avg_daily_volume=500,
        )
        assert isinstance(result, int)
        # kelly_contracts = int(0.05 * 100000 / 1000) = int(5.0) = 5
        # volume_cap = int(0.02 * 500) = int(10.0) = 10
        # min(5, 10) = 5
        assert result == 5

    def test_volume_cap_limits_position(self):
        """If kelly_contracts > volume_cap, use volume_cap."""
        result = volume_capped_position(
            kelly_pct=0.10,
            capital=1_000_000,
            contract_value=1_000,
            avg_daily_volume=100,
        )
        # kelly_contracts = int(0.10 * 1_000_000 / 1_000) = 100
        # volume_cap = int(0.02 * 100) = 2
        # min(100, 2) = 2
        assert result == 2

    def test_zero_kelly_returns_zero(self):
        """kelly_pct=0 -> 0 contracts."""
        result = volume_capped_position(
            kelly_pct=0.0,
            capital=100_000,
            contract_value=1_000,
            avg_daily_volume=500,
        )
        assert result == 0
