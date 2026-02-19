"""Tests for the volume-adaptive slippage model.

Square-root impact model:
    slippage = spread/2 + k * sigma * sqrt(V_order / V_daily)
"""

import math

import pytest

from hydra.risk.slippage import estimate_slippage


class TestEstimateSlippage:
    """Tests for estimate_slippage function."""

    def test_zero_order_size(self):
        """order_size=0 -> slippage = spread/2 only (no market impact)."""
        result = estimate_slippage(
            order_size=0,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
            impact_coefficient=0.1,
        )
        assert result == pytest.approx(1.0)  # spread/2 = 2.0/2 = 1.0

    def test_slippage_increases_with_order_size(self):
        """Larger orders produce higher slippage."""
        small = estimate_slippage(
            order_size=10,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
        )
        large = estimate_slippage(
            order_size=500,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
        )
        assert large > small

    def test_slippage_scales_with_volatility(self):
        """Higher daily_volatility increases market impact."""
        low_vol = estimate_slippage(
            order_size=100,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.01,
        )
        high_vol = estimate_slippage(
            order_size=100,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.04,
        )
        assert high_vol > low_vol

    def test_slippage_with_custom_k(self):
        """k=0.2 produces higher slippage than k=0.1 for same inputs."""
        low_k = estimate_slippage(
            order_size=100,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
            impact_coefficient=0.1,
        )
        high_k = estimate_slippage(
            order_size=100,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
            impact_coefficient=0.2,
        )
        assert high_k > low_k

    def test_slippage_formula_exact(self):
        """Verify exact formula for known inputs.

        order_size=100, daily_volume=5000, spread=2.0, sigma=0.02, k=0.1
        Expected: spread/2 + k * sigma * sqrt(100/5000)
                = 1.0 + 0.1 * 0.02 * sqrt(0.02)
                = 1.0 + 0.002 * 0.14142...
                = 1.0 + 0.00028284...
                = 1.00028284...
        """
        result = estimate_slippage(
            order_size=100,
            daily_volume=5000,
            spread=2.0,
            daily_volatility=0.02,
            impact_coefficient=0.1,
        )
        participation_rate = 100 / 5000
        expected = 2.0 / 2.0 + 0.1 * 0.02 * math.sqrt(participation_rate)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_zero_daily_volume_no_crash(self):
        """daily_volume=0 should not crash (guarded with max(daily_volume, 1))."""
        result = estimate_slippage(
            order_size=10,
            daily_volume=0,
            spread=2.0,
            daily_volatility=0.02,
        )
        assert result > 0  # Should still return a valid positive number
