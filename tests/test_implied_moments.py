"""TDD tests for implied moments computation from B-L density.

Tests the moments calculator:
- compute_moments: Computes mean, variance, skew, kurtosis from density
- ImpliedMoments: Dataclass holding moment output with quality
- Degraded input: Returns None for moments except atm_iv
- Log-normal benchmark: Moments match analytic log-normal within 5%
- Stability: Slightly different inputs produce moments within 10%
"""

import math

import numpy as np
import pytest
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helper: generate synthetic density results
# ---------------------------------------------------------------------------

def _make_density_result(
    quality="full",
    forward=100.0,
    sigma=0.30,
    T=0.25,
    n_points=200,
    strike_range=(70.0, 140.0),
    atm_iv=0.30,
):
    """Create a synthetic ImpliedDensityResult for testing moments.

    For FULL quality: generates a log-normal density.
    For DEGRADED: returns trivial density/strikes.
    """
    from hydra.signals.options_math.density import DataQuality, ImpliedDensityResult

    if quality == "degraded":
        return ImpliedDensityResult(
            strikes=np.array([forward]),
            density=np.array([0.0]),
            quality=DataQuality.DEGRADED,
            liquid_strike_count=3,
            atm_iv=atm_iv,
            warnings=["Insufficient liquid strikes"],
        )

    # Generate log-normal density analytically
    strikes = np.linspace(strike_range[0], strike_range[1], n_points)

    # Log-normal density: f(K) = 1/(K*sigma*sqrt(T)) * phi(d)
    # where d = (ln(K/F) - (r - 0.5*sigma^2)*T) / (sigma*sqrt(T))
    # For risk-neutral density centered at forward:
    mu = np.log(forward) - 0.5 * sigma**2 * T
    s = sigma * np.sqrt(T)

    density = np.zeros_like(strikes)
    for i, K in enumerate(strikes):
        if K > 0:
            d = (np.log(K) - mu) / s
            density[i] = norm.pdf(d) / (K * s)

    # Normalize
    integral = np.trapezoid(density, strikes)
    if integral > 0:
        density = density / integral

    return ImpliedDensityResult(
        strikes=strikes,
        density=density,
        quality=DataQuality.FULL,
        liquid_strike_count=20,
        atm_iv=atm_iv,
        warnings=[],
    )


# ===========================================================================
# Test 1: ImpliedMoments dataclass
# ===========================================================================


class TestImpliedMomentsDataclass:
    """Verify ImpliedMoments is properly defined with expected fields."""

    def test_implied_moments_fields(self):
        """ImpliedMoments has mean, variance, skew, kurtosis, atm_iv, quality, warnings."""
        from hydra.signals.options_math.moments import ImpliedMoments
        from hydra.signals.options_math.density import DataQuality

        moments = ImpliedMoments(
            mean=100.0,
            variance=225.0,
            skew=-0.5,
            kurtosis=3.0,
            atm_iv=0.30,
            quality=DataQuality.FULL,
            warnings=[],
        )
        assert moments.mean == 100.0
        assert moments.variance == 225.0
        assert moments.skew == -0.5
        assert moments.kurtosis == 3.0
        assert moments.atm_iv == 0.30
        assert moments.quality.value == "full"

    def test_degraded_moments_have_none(self):
        """Degraded quality moments have None for mean/variance/skew/kurtosis."""
        from hydra.signals.options_math.moments import ImpliedMoments
        from hydra.signals.options_math.density import DataQuality

        moments = ImpliedMoments(
            mean=None,
            variance=None,
            skew=None,
            kurtosis=None,
            atm_iv=0.30,
            quality=DataQuality.DEGRADED,
            warnings=["Insufficient data for moment computation"],
        )
        assert moments.mean is None
        assert moments.variance is None
        assert moments.skew is None
        assert moments.kurtosis is None
        assert moments.atm_iv == 0.30


# ===========================================================================
# Test 2: Full quality moments computation
# ===========================================================================


class TestFullQualityMoments:
    """Moments computed from full-quality density."""

    def test_mean_near_forward(self):
        """Mean of density should be close to forward price."""
        from hydra.signals.options_math.moments import compute_moments

        density_result = _make_density_result(quality="full", forward=100.0)
        moments = compute_moments(density_result)

        assert moments.mean is not None
        assert abs(moments.mean - 100.0) / 100.0 < 0.05  # within 5%

    def test_variance_positive(self):
        """Variance should be positive for any non-degenerate density."""
        from hydra.signals.options_math.moments import compute_moments

        density_result = _make_density_result(quality="full")
        moments = compute_moments(density_result)

        assert moments.variance is not None
        assert moments.variance > 0

    def test_kurtosis_above_zero(self):
        """Kurtosis should be positive (raw kurtosis)."""
        from hydra.signals.options_math.moments import compute_moments

        density_result = _make_density_result(quality="full")
        moments = compute_moments(density_result)

        assert moments.kurtosis is not None
        assert moments.kurtosis > 0

    def test_quality_is_full(self):
        """Full quality density produces full quality moments."""
        from hydra.signals.options_math.moments import compute_moments
        from hydra.signals.options_math.density import DataQuality

        density_result = _make_density_result(quality="full")
        moments = compute_moments(density_result)

        assert moments.quality == DataQuality.FULL


# ===========================================================================
# Test 3: Degraded density input
# ===========================================================================


class TestDegradedMoments:
    """Degraded density input returns None for moments."""

    def test_degraded_returns_none_moments(self):
        """Degraded density result => moments all None except atm_iv."""
        from hydra.signals.options_math.moments import compute_moments
        from hydra.signals.options_math.density import DataQuality

        density_result = _make_density_result(quality="degraded", atm_iv=0.28)
        moments = compute_moments(density_result)

        assert moments.mean is None
        assert moments.variance is None
        assert moments.skew is None
        assert moments.kurtosis is None
        assert moments.atm_iv == 0.28
        assert moments.quality == DataQuality.DEGRADED

    def test_degraded_has_warning(self):
        """Degraded moments include warning about insufficient data."""
        from hydra.signals.options_math.moments import compute_moments

        density_result = _make_density_result(quality="degraded")
        moments = compute_moments(density_result)

        assert len(moments.warnings) > 0


# ===========================================================================
# Test 4: Log-normal benchmark -- analytic moment verification
# ===========================================================================


class TestLogNormalBenchmark:
    """For log-normal density, moments should match analytic values within 5%."""

    def test_variance_matches_lognormal(self):
        """Variance should match analytic log-normal variance within 5%."""
        from hydra.signals.options_math.moments import compute_moments

        F = 100.0
        sigma = 0.30
        T = 0.25

        density_result = _make_density_result(
            quality="full", forward=F, sigma=sigma, T=T,
            n_points=500, strike_range=(50.0, 180.0),
        )
        moments = compute_moments(density_result)

        # Analytic log-normal variance: F^2 * (exp(sigma^2*T) - 1)
        analytic_var = F**2 * (np.exp(sigma**2 * T) - 1)

        assert moments.variance is not None
        pct_err = abs(moments.variance - analytic_var) / analytic_var
        assert pct_err < 0.05, (
            f"Variance {moments.variance:.2f} vs analytic {analytic_var:.2f} "
            f"(error: {pct_err:.4f})"
        )

    def test_skew_positive_for_lognormal(self):
        """Log-normal distribution has positive skew."""
        from hydra.signals.options_math.moments import compute_moments

        density_result = _make_density_result(
            quality="full", forward=100.0, sigma=0.30, T=0.25,
            n_points=500, strike_range=(50.0, 180.0),
        )
        moments = compute_moments(density_result)

        assert moments.skew is not None
        # Log-normal skew is positive
        assert moments.skew > 0, f"Expected positive skew, got {moments.skew:.4f}"


# ===========================================================================
# Test 5: Stability -- slight perturbation produces similar moments
# ===========================================================================


class TestMomentsStability:
    """Two slightly different inputs produce moments within 10% of each other."""

    def test_perturbed_vol_produces_similar_moments(self):
        """sigma=0.30 vs sigma=0.31: moments within 10%."""
        from hydra.signals.options_math.moments import compute_moments

        result1 = _make_density_result(quality="full", sigma=0.30, n_points=500,
                                        strike_range=(50.0, 180.0))
        result2 = _make_density_result(quality="full", sigma=0.31, n_points=500,
                                        strike_range=(50.0, 180.0))

        moments1 = compute_moments(result1)
        moments2 = compute_moments(result2)

        # Mean should be very similar
        assert moments1.mean is not None
        assert moments2.mean is not None
        mean_diff = abs(moments1.mean - moments2.mean) / moments1.mean
        assert mean_diff < 0.10, f"Mean diff {mean_diff:.4f} exceeds 10%"

        # Variance should be within 10%
        assert moments1.variance is not None
        assert moments2.variance is not None
        var_diff = abs(moments1.variance - moments2.variance) / moments1.variance
        assert var_diff < 0.10, f"Variance diff {var_diff:.4f} exceeds 10%"
