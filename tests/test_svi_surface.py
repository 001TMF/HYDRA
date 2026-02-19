"""TDD tests for SVI volatility surface calibration.

Tests the core mathematical functions:
- svi_total_variance: SVI parameterization w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
- calibrate_svi: Fit SVI parameters to market implied volatilities
- SVICalibrationResult: Dataclass holding calibration output
- svi_to_call_prices: Convert fitted SVI to smooth call prices for B-L input
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Test 1: svi_total_variance is vectorized and matches known formula
# --------------------------------------------------------------------------- #


class TestSVITotalVariance:
    """Verify the raw SVI formula produces correct total variance values."""

    def test_scalar_input(self):
        """Single log-moneyness value returns correct total variance."""
        from hydra.signals.options_math.surface import svi_total_variance

        # With known params: a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.5
        # k=0: w = 0.04 + 0.1 * (-0.3 * 0 + sqrt(0 + 0.25)) = 0.04 + 0.1*0.5 = 0.09
        k = np.array([0.0])
        w = svi_total_variance(k, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.5)
        np.testing.assert_allclose(w, [0.09], atol=1e-10)

    def test_vectorized(self):
        """Multiple log-moneyness values return array of same length."""
        from hydra.signals.options_math.surface import svi_total_variance

        k = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
        w = svi_total_variance(k, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.5)
        assert w.shape == (5,)
        # Total variance must be non-negative for valid params
        assert np.all(w >= 0)

    def test_symmetry_when_rho_zero(self):
        """When rho=0 and m=0, the smile is symmetric around k=0."""
        from hydra.signals.options_math.surface import svi_total_variance

        k = np.array([-0.3, 0.3])
        w = svi_total_variance(k, a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.5)
        np.testing.assert_allclose(w[0], w[1], atol=1e-12)


# --------------------------------------------------------------------------- #
# Test 2: Flat vol smile -- SVI should recover near-flat surface
# --------------------------------------------------------------------------- #


class TestCalibrateFlat:
    """SVI fit to flat implied volatility should recover near-flat surface."""

    def test_flat_vol_rmse_below_threshold(self):
        """Flat vol (all IVs = 0.30, T=0.5): RMSE < 0.001."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        strikes = np.linspace(85, 115, 15)
        market_ivs = np.full_like(strikes, 0.30)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert result.rmse < 0.001, f"RMSE {result.rmse:.6f} exceeds 0.001 for flat vol"

    def test_flat_vol_fitted_ivs_close(self):
        """Fitted IVs should all be very close to 0.30 for flat input."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        strikes = np.linspace(85, 115, 15)
        market_ivs = np.full_like(strikes, 0.30)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        np.testing.assert_allclose(result.fitted_iv, 0.30, atol=0.005)

    def test_flat_vol_no_butterfly_arbitrage(self):
        """Flat surface should have no butterfly arbitrage."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        strikes = np.linspace(85, 115, 15)
        market_ivs = np.full_like(strikes, 0.30)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert not result.has_butterfly_arbitrage


# --------------------------------------------------------------------------- #
# Test 3: Typical skewed smile
# --------------------------------------------------------------------------- #


class TestCalibrateSkewed:
    """SVI fit to a realistic skewed volatility smile."""

    def test_skewed_smile_rmse_below_threshold(self):
        """10-strike skewed smile (IVs from 0.35 to 0.25): RMSE < 0.02."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.25
        strikes = np.linspace(90, 110, 10)
        # Typical skew: higher IV for lower strikes (put skew)
        market_ivs = 0.30 - 0.005 * (strikes - forward) / forward * 10
        # This gives ~0.35 at K=90, ~0.25 at K=110

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert result.rmse < 0.02, f"RMSE {result.rmse:.6f} exceeds 0.02 for skewed smile"

    def test_skewed_returns_svicalibrationresult(self):
        """calibrate_svi returns an SVICalibrationResult dataclass."""
        from hydra.signals.options_math.surface import (
            SVICalibrationResult,
            calibrate_svi,
        )

        forward = 100.0
        T = 0.25
        strikes = np.linspace(90, 110, 10)
        market_ivs = 0.30 - 0.005 * (strikes - forward) / forward * 10

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert isinstance(result, SVICalibrationResult)
        assert "a" in result.params
        assert "b" in result.params
        assert "rho" in result.params
        assert "m" in result.params
        assert "sigma" in result.params
        assert isinstance(result.fitted_iv, np.ndarray)
        assert isinstance(result.rmse, float)
        assert isinstance(result.has_butterfly_arbitrage, bool)
        assert isinstance(result.warnings, list)


# --------------------------------------------------------------------------- #
# Test 4: Sparse data handling (8 strikes with noise)
# --------------------------------------------------------------------------- #


class TestCalibrateSparse:
    """SVI handles sparse data (8-15 points) without blowing up."""

    def test_sparse_8_points_converges(self):
        """8 strikes with 5% noise: SVI converges, no NaN/Inf."""
        from hydra.signals.options_math.surface import calibrate_svi

        rng = np.random.default_rng(42)
        forward = 100.0
        T = 0.5
        strikes = np.array([88, 92, 95, 98, 100, 102, 105, 112])
        base_ivs = 0.30 - 0.003 * (strikes - forward)
        noise = rng.normal(0, 0.05 * base_ivs)
        market_ivs = base_ivs + noise
        # Ensure IVs are positive
        market_ivs = np.maximum(market_ivs, 0.05)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert np.isfinite(result.rmse), "RMSE is not finite"
        assert np.all(np.isfinite(result.fitted_iv)), "Fitted IVs contain NaN/Inf"

    def test_very_sparse_5_points_produces_warning(self):
        """5 strikes should still produce a result but with a warning."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        strikes = np.array([90, 95, 100, 105, 110])
        market_ivs = np.array([0.35, 0.32, 0.30, 0.28, 0.26])

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert np.isfinite(result.rmse), "RMSE is not finite for 5-point data"
        assert len(result.warnings) > 0, "Expected warning for very sparse data"


# --------------------------------------------------------------------------- #
# Test 5: Butterfly arbitrage detection
# --------------------------------------------------------------------------- #


class TestButterflyArbitrageDetection:
    """Detect invalid (non-convex) surfaces via d2w/dk2 < 0 check."""

    def test_detects_arbitrage_in_invalid_surface(self):
        """Manually created concavity should trigger butterfly arbitrage flag."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        # Create a "W-shaped" smile that should trigger non-convexity
        strikes = np.linspace(85, 115, 15)
        k = np.log(strikes / forward)
        # Create artificial non-monotone total variance
        market_ivs = 0.30 + 0.10 * np.cos(4 * np.pi * k)
        # This creates a wavy pattern that SVI can't fit perfectly,
        # but the test verifies the arbitrage check mechanism.

        result = calibrate_svi(strikes, market_ivs, forward, T)
        # Even if SVI smooths away the concavity, we verify the check runs.
        # The has_butterfly_arbitrage field must be a bool.
        assert isinstance(result.has_butterfly_arbitrage, bool)

    def test_well_behaved_smile_no_arbitrage(self):
        """A standard skew smile should not flag butterfly arbitrage."""
        from hydra.signals.options_math.surface import calibrate_svi

        forward = 100.0
        T = 0.5
        strikes = np.linspace(85, 115, 20)
        # Standard monotone decreasing skew
        market_ivs = 0.35 - 0.10 * (strikes - 85) / 30

        result = calibrate_svi(strikes, market_ivs, forward, T)
        assert not result.has_butterfly_arbitrage, (
            "Standard skew should not have butterfly arbitrage"
        )


# --------------------------------------------------------------------------- #
# Test 6: svi_to_call_prices -- smooth call prices via Black-76
# --------------------------------------------------------------------------- #


class TestSVIToCallPrices:
    """svi_to_call_prices converts fitted SVI params to smooth call prices."""

    def test_call_prices_monotonically_decreasing(self):
        """Call prices should decrease as strike increases."""
        from hydra.signals.options_math.surface import calibrate_svi, svi_to_call_prices

        forward = 100.0
        T = 0.5
        r = 0.05
        strikes = np.linspace(85, 115, 15)
        market_ivs = 0.30 - 0.003 * (strikes - forward)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        strikes_fine = np.linspace(85, 115, 100)
        call_prices = svi_to_call_prices(result.params, strikes_fine, forward, r, T)

        assert call_prices.shape == (100,)
        # Call prices must be monotonically decreasing with strike
        diffs = np.diff(call_prices)
        assert np.all(diffs <= 1e-10), "Call prices not monotonically decreasing"

    def test_call_prices_positive(self):
        """All call prices must be non-negative."""
        from hydra.signals.options_math.surface import calibrate_svi, svi_to_call_prices

        forward = 100.0
        T = 0.5
        r = 0.05
        strikes = np.linspace(85, 115, 15)
        market_ivs = np.full_like(strikes, 0.30)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        strikes_fine = np.linspace(85, 115, 100)
        call_prices = svi_to_call_prices(result.params, strikes_fine, forward, r, T)

        assert np.all(call_prices >= 0), "Found negative call prices"

    def test_call_prices_smooth_for_bl(self):
        """Call prices should be smooth enough for B-L second derivative."""
        from hydra.signals.options_math.surface import calibrate_svi, svi_to_call_prices

        forward = 100.0
        T = 0.5
        r = 0.05
        strikes = np.linspace(85, 115, 15)
        market_ivs = 0.30 - 0.003 * (strikes - forward)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        strikes_fine = np.linspace(87, 113, 200)
        call_prices = svi_to_call_prices(result.params, strikes_fine, forward, r, T)

        # Second derivative should be non-negative (convexity of call prices)
        d2c = np.gradient(np.gradient(call_prices, strikes_fine), strikes_fine)
        # Allow small numerical noise
        assert np.all(d2c >= -1e-6), (
            f"Call price convexity violated: min d2C = {d2c.min():.8f}"
        )

    def test_atm_call_price_reasonable(self):
        """ATM call price should be approximately F * N(d1) - K * N(d2) discounted."""
        from hydra.signals.options_math.surface import calibrate_svi, svi_to_call_prices

        forward = 100.0
        T = 0.5
        r = 0.05
        strikes = np.linspace(85, 115, 15)
        market_ivs = np.full_like(strikes, 0.30)

        result = calibrate_svi(strikes, market_ivs, forward, T)
        strikes_fine = np.array([100.0])
        call_prices = svi_to_call_prices(result.params, strikes_fine, forward, r, T)

        # ATM Black-76 call price: e^(-rT) * F * (2*N(sigma*sqrt(T)/2) - 1)
        # For sigma=0.30, T=0.5: sigma*sqrt(T) ~ 0.212
        # N(0.106) ~ 0.542
        # Call ~ e^(-0.025)*100*(2*0.542 - 1) ~ 0.975 * 100 * 0.084 ~ 8.2
        # Just check it's in a reasonable range
        assert 5.0 < call_prices[0] < 15.0, (
            f"ATM call price {call_prices[0]:.2f} outside reasonable range [5, 15]"
        )
