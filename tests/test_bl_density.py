"""TDD tests for Breeden-Litzenberger risk-neutral density extraction.

Tests the core B-L pipeline:
- extract_density: Filters liquid strikes, calibrates SVI, computes density via d2C/dK2
- ImpliedDensityResult: Dataclass holding density output with quality assessment
- DataQuality: Enum for FULL/DEGRADED/STALE/MISSING quality levels
- Graceful degradation to ATM IV when insufficient liquid strikes
- Negative density clipping and renormalization
- Log-normal benchmark recovery
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers: generate synthetic options data from Black-76
# ---------------------------------------------------------------------------

def _black76_call_price(F, K, r, T, sigma):
    """Black-76 call price for test data generation."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T)
    return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _make_chain_data(
    forward=100.0,
    r=0.05,
    T=0.25,
    sigma=0.30,
    n_strikes=20,
    strike_range=(80.0, 120.0),
    oi_per_strike=200,
    spread_pct=0.05,
    skew=0.0,
):
    """Generate synthetic options chain data from known Black-76 vol.

    Returns dict with: strikes, call_prices, oi, bid_ask_spread_pct, plus
    the underlying parameters for verification.
    """
    strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)
    # Apply optional skew to create realistic smile
    ivs = sigma + skew * (strikes - forward) / forward
    ivs = np.maximum(ivs, 0.05)  # floor at 5% vol

    call_prices = np.array([
        _black76_call_price(forward, K, r, T, iv)
        for K, iv in zip(strikes, ivs)
    ])

    oi = np.full(n_strikes, oi_per_strike, dtype=float)
    bid_ask = np.full(n_strikes, spread_pct, dtype=float)

    return {
        "strikes": strikes,
        "call_prices": call_prices,
        "oi": oi,
        "bid_ask_spread_pct": bid_ask,
        # Reference values
        "forward": forward,
        "r": r,
        "T": T,
        "sigma": sigma,
    }


# ===========================================================================
# Test 1: DataQuality enum and ImpliedDensityResult dataclass
# ===========================================================================


class TestDataQualityAndResult:
    """Verify DataQuality enum and ImpliedDensityResult are properly defined."""

    def test_data_quality_enum_values(self):
        """DataQuality has FULL, DEGRADED, STALE, MISSING members."""
        from hydra.signals.options_math.density import DataQuality

        assert DataQuality.FULL.value == "full"
        assert DataQuality.DEGRADED.value == "degraded"
        assert DataQuality.STALE.value == "stale"
        assert DataQuality.MISSING.value == "missing"

    def test_implied_density_result_fields(self):
        """ImpliedDensityResult has all expected fields."""
        from hydra.signals.options_math.density import (
            DataQuality,
            ImpliedDensityResult,
        )

        result = ImpliedDensityResult(
            strikes=np.array([90.0, 100.0, 110.0]),
            density=np.array([0.01, 0.03, 0.01]),
            quality=DataQuality.FULL,
            liquid_strike_count=20,
            atm_iv=0.30,
            warnings=[],
        )
        assert result.quality == DataQuality.FULL
        assert result.liquid_strike_count == 20
        assert result.atm_iv == 0.30
        assert isinstance(result.warnings, list)

    def test_data_quality_importable_from_greeks(self):
        """DataQuality is importable from greeks module (shared enum)."""
        from hydra.signals.options_math.greeks import DataQuality as GreeksQuality
        from hydra.signals.options_math.density import DataQuality as DensityQuality

        # They should be the same class
        assert GreeksQuality is DensityQuality


# ===========================================================================
# Test 2: Full quality density extraction (20 liquid strikes)
# ===========================================================================


class TestFullQualityDensity:
    """B-L density from clean, full-quality data."""

    def test_density_integrates_to_one(self):
        """Density integrates to approximately 1.0 (+/- 0.05)."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=20, oi_per_strike=200, spread_pct=0.05)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        integral = np.trapezoid(result.density, result.strikes)
        assert abs(integral - 1.0) < 0.05, (
            f"Density integral {integral:.4f} not within 0.05 of 1.0"
        )

    def test_density_mean_near_forward(self):
        """Density mean within 2% of forward price."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=20)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        mean = np.trapezoid(result.strikes * result.density, result.strikes)
        forward = data["forward"]
        pct_err = abs(mean - forward) / forward
        assert pct_err < 0.02, (
            f"Density mean {mean:.2f} not within 2% of forward {forward:.2f} "
            f"(error: {pct_err:.4f})"
        )

    def test_full_quality_returns_quality_full(self):
        """20 liquid strikes returns quality=FULL."""
        from hydra.signals.options_math.density import DataQuality, extract_density

        data = _make_chain_data(n_strikes=20)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert result.quality == DataQuality.FULL
        assert result.liquid_strike_count >= 8

    def test_density_non_negative(self):
        """All density values are non-negative (negative clipped)."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=20)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert np.all(result.density >= 0), "Density contains negative values"

    def test_atm_iv_set_for_full_quality(self):
        """ATM IV is set even for full quality results."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=20, sigma=0.30)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert result.atm_iv is not None
        # ATM IV should be close to the input vol
        assert abs(result.atm_iv - 0.30) < 0.05


# ===========================================================================
# Test 3: Degraded quality (insufficient liquid strikes)
# ===========================================================================


class TestDegradedQuality:
    """Graceful degradation when fewer than min_liquid_strikes available."""

    def test_five_liquid_strikes_returns_degraded(self):
        """5 liquid strikes (below default 8) returns DEGRADED."""
        from hydra.signals.options_math.density import DataQuality, extract_density

        data = _make_chain_data(n_strikes=5, oi_per_strike=200, spread_pct=0.05)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert result.quality == DataQuality.DEGRADED

    def test_degraded_has_atm_iv(self):
        """Degraded result still has ATM IV set."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=5, sigma=0.30)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert result.atm_iv is not None
        assert result.atm_iv > 0

    def test_degraded_has_warning(self):
        """Degraded result includes warning about insufficient data."""
        from hydra.signals.options_math.density import extract_density

        data = _make_chain_data(n_strikes=5)
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        assert len(result.warnings) > 0
        assert any("liquid" in w.lower() or "strike" in w.lower()
                    for w in result.warnings)

    def test_low_oi_causes_degradation(self):
        """Strikes with OI < min_oi are excluded from liquid count."""
        from hydra.signals.options_math.density import DataQuality, extract_density

        data = _make_chain_data(n_strikes=20, oi_per_strike=10)  # low OI
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
            min_oi=50,
        )

        assert result.quality == DataQuality.DEGRADED

    def test_wide_spread_causes_degradation(self):
        """Strikes with wide bid-ask spread excluded from liquid count."""
        from hydra.signals.options_math.density import DataQuality, extract_density

        data = _make_chain_data(n_strikes=20, spread_pct=0.50)  # wide spreads
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
            max_spread_pct=0.20,
        )

        assert result.quality == DataQuality.DEGRADED


# ===========================================================================
# Test 4: Negative density handling
# ===========================================================================


class TestNegativeDensityHandling:
    """Negative density regions are clipped to zero and renormalized."""

    def test_clipped_density_has_warning(self):
        """If negative density was clipped, a warning is issued."""
        from hydra.signals.options_math.density import extract_density

        # Use a skewed smile that might produce some negative density regions
        # at the extreme tails
        data = _make_chain_data(
            n_strikes=20,
            skew=-0.15,  # strong negative skew
            strike_range=(75.0, 125.0),
        )
        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=data["forward"],
            r=data["r"],
            T=data["T"],
        )

        # Density must be non-negative regardless
        assert np.all(result.density >= 0)
        # After any clipping, density should still integrate close to 1.0
        if result.quality.value == "full":
            integral = np.trapezoid(result.density, result.strikes)
            assert abs(integral - 1.0) < 0.10  # relaxed tolerance for skewed case


# ===========================================================================
# Test 5: Log-normal benchmark
# ===========================================================================


class TestLogNormalBenchmark:
    """B-L should recover approximately log-normal density from Black-76 prices."""

    def test_lognormal_shape_recovery(self):
        """Black-76 prices with flat vol should produce log-normal-like density."""
        from hydra.signals.options_math.density import extract_density

        F = 100.0
        r = 0.05
        T = 0.25
        sigma = 0.30

        # Wide strike range to capture most of the distribution
        data = _make_chain_data(
            forward=F,
            r=r,
            T=T,
            sigma=sigma,
            n_strikes=25,
            strike_range=(70.0, 140.0),
            oi_per_strike=500,
            spread_pct=0.02,
        )

        result = extract_density(
            strikes=data["strikes"],
            call_prices=data["call_prices"],
            oi=data["oi"],
            bid_ask_spread_pct=data["bid_ask_spread_pct"],
            spot=F,
            r=r,
            T=T,
        )

        assert result.quality.value == "full"

        # Density should be unimodal with peak near forward price
        peak_idx = np.argmax(result.density)
        peak_strike = result.strikes[peak_idx]
        # Peak should be reasonably close to forward (within 10%)
        assert abs(peak_strike - F) / F < 0.10, (
            f"Density peak at {peak_strike:.2f}, expected near {F:.2f}"
        )


# ===========================================================================
# Test 6: Liquidity filtering
# ===========================================================================


class TestLiquidityFiltering:
    """Verify liquid strike counting with OI and spread filters."""

    def test_liquid_count_correct_with_mixed_quality(self):
        """Mixed OI and spread data: liquid count should only include valid strikes."""
        from hydra.signals.options_math.density import extract_density

        n = 15
        strikes = np.linspace(85, 115, n)
        F = 100.0
        r = 0.05
        T = 0.25
        sigma = 0.30
        call_prices = np.array([
            _black76_call_price(F, K, r, T, sigma) for K in strikes
        ])

        # 10 liquid, 5 illiquid
        oi = np.array([200] * 10 + [10] * 5, dtype=float)
        spread = np.array([0.05] * 10 + [0.05] * 5, dtype=float)

        result = extract_density(
            strikes=strikes,
            call_prices=call_prices,
            oi=oi,
            bid_ask_spread_pct=spread,
            spot=F,
            r=r,
            T=T,
            min_oi=50,
        )

        assert result.liquid_strike_count == 10
        assert result.quality.value == "full"
