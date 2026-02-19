"""TDD tests for Black-76 Greeks computation and flow aggregation.

Tests cover:
- Individual Black-76 Greeks (gamma, vanna, charm, delta, vega)
- Aggregated flow metrics (GEX, vanna flow, charm flow)
- Graceful degradation with insufficient liquid strikes
- Edge cases (zero OI, expired options, near-zero vol/time)
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from hydra.signals.options_math.greeks import (
    GreeksFlowResult,
    black76_greeks,
    compute_greeks_flow,
)


# ---------------------------------------------------------------------------
# Helper: analytic Black-76 Greeks reference values
# ---------------------------------------------------------------------------

def _analytic_gamma(F, K, r, T, sigma):
    """Reference Black-76 gamma from analytic formula."""
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    discount = math.exp(-r * T)
    return discount * norm.pdf(d1) / (F * sigma * math.sqrt(T))


def _analytic_vanna(F, K, r, T, sigma):
    """Reference Black-76 vanna from analytic formula."""
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    return -discount * norm.pdf(d1) * d2 / sigma


def _analytic_charm(F, K, r, T, sigma):
    """Reference Black-76 charm from analytic formula."""
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    return -discount * norm.pdf(d1) * (
        2 * r * T - d2 * sigma * math.sqrt(T)
    ) / (2 * T * sigma * math.sqrt(T))


# ===========================================================================
# Test Suite 1: Individual Black-76 Greeks
# ===========================================================================

class TestBlack76Greeks:
    """Tests for the black76_greeks function."""

    def test_atm_call_gamma_matches_analytic(self):
        """ATM call: gamma matches analytic Black-76 formula within 1e-6."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        expected_gamma = _analytic_gamma(F, K, r, T, sigma)
        assert abs(result["gamma"] - expected_gamma) < 1e-6
        assert result["gamma"] > 0  # gamma is always positive

    def test_atm_call_vanna_matches_analytic(self):
        """ATM call: vanna matches analytic formula."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        expected_vanna = _analytic_vanna(F, K, r, T, sigma)
        assert abs(result["vanna"] - expected_vanna) < 1e-6

    def test_atm_call_charm_matches_analytic(self):
        """ATM call: charm matches analytic formula."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        expected_charm = _analytic_charm(F, K, r, T, sigma)
        assert abs(result["charm"] - expected_charm) < 1e-6

    def test_atm_call_delta_positive(self):
        """ATM call delta should be approximately 0.5 (discounted)."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        discount = math.exp(-r * T)
        # For ATM, d1 > 0 slightly, so delta slightly > 0.5 * discount
        assert 0.4 < result["delta"] < 0.6
        assert result["delta"] > 0  # call delta positive

    def test_atm_put_delta_negative(self):
        """ATM put delta should be negative."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=False)

        assert result["delta"] < 0  # put delta negative

    def test_call_and_put_gamma_equal(self):
        """Gamma is the same for calls and puts at the same strike."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        call = black76_greeks(F, K, r, T, sigma, is_call=True)
        put = black76_greeks(F, K, r, T, sigma, is_call=False)

        assert abs(call["gamma"] - put["gamma"]) < 1e-10

    def test_vega_positive(self):
        """Vega should be positive for both calls and puts."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.30
        call = black76_greeks(F, K, r, T, sigma, is_call=True)
        put = black76_greeks(F, K, r, T, sigma, is_call=False)

        assert call["vega"] > 0
        assert put["vega"] > 0

    def test_deep_otm_put_gamma_near_zero(self):
        """Deep OTM put (K << F): gamma should be near zero."""
        F, K, r, T, sigma = 100.0, 60.0, 0.05, 0.25, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=False)

        assert result["gamma"] < 0.001  # near zero for deep OTM

    def test_near_expiry_atm_gamma_high(self):
        """Near expiry ATM: gamma blowup (very high gamma)."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.01, 0.30  # T=0.01 ~ 2.5 days
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        # Gamma should be much higher near expiry for ATM
        far_result = black76_greeks(F, K, r, 0.25, sigma, is_call=True)
        assert result["gamma"] > far_result["gamma"] * 3  # at least 3x higher

    def test_zero_vol_returns_zeros_no_crash(self):
        """Zero vol (sigma -> 0): should not crash, returns zeros."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.25, 0.0
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        assert result["gamma"] == 0.0
        assert result["vanna"] == 0.0
        assert result["charm"] == 0.0
        assert math.isfinite(result["delta"])
        assert math.isfinite(result["vega"])

    def test_zero_time_returns_zeros_no_crash(self):
        """T near zero: should not crash, returns zeros."""
        F, K, r, T, sigma = 100.0, 100.0, 0.05, 0.0, 0.30
        result = black76_greeks(F, K, r, T, sigma, is_call=True)

        assert result["gamma"] == 0.0
        assert result["vanna"] == 0.0
        assert result["charm"] == 0.0

    def test_returns_all_expected_keys(self):
        """Result dict contains all expected Greek keys."""
        result = black76_greeks(100.0, 100.0, 0.05, 0.25, 0.30, True)
        expected_keys = {"gamma", "vanna", "charm", "delta", "vega"}
        assert set(result.keys()) >= expected_keys

    def test_known_value_gamma(self):
        """Verify gamma against hand-calculated known value."""
        F, K, r, T, sigma = 100.0, 105.0, 0.05, 0.5, 0.25
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        discount = math.exp(-r * T)
        expected = discount * norm.pdf(d1) / (F * sigma * math.sqrt(T))

        result = black76_greeks(F, K, r, T, sigma, is_call=True)
        assert abs(result["gamma"] - expected) < 1e-10


# ===========================================================================
# Test Suite 2: Aggregated Greeks Flow
# ===========================================================================

def _make_chain(
    strikes,
    call_ivs,
    put_ivs,
    call_oi,
    put_oi,
    expiries_T,
    bid_ask_spread_pct=None,
):
    """Helper to construct a chain dict for compute_greeks_flow."""
    n = len(strikes)
    if bid_ask_spread_pct is None:
        bid_ask_spread_pct = [0.05] * n  # tight spreads by default
    return {
        "strikes": strikes,
        "call_ivs": call_ivs,
        "put_ivs": put_ivs,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "expiries_T": expiries_T,
        "bid_ask_spread_pct": bid_ask_spread_pct,
    }


class TestComputeGreeksFlow:
    """Tests for the compute_greeks_flow aggregation function."""

    def test_simple_two_strike_gex_matches_hand_calc(self):
        """Simple 2-strike chain: verify GEX = sum of per-strike contributions."""
        spot = 100.0
        r = 0.05
        T = 0.25
        multiplier = 400.0  # lean hog contract multiplier

        strikes = [95.0, 105.0]
        call_ivs = [0.30, 0.28]
        put_ivs = [0.32, 0.29]
        call_oi = [100, 200]
        put_oi = [150, 100]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [T, T])

        # Use min_liquid_strikes=2 to bypass degradation for this unit test
        result = compute_greeks_flow(chain, spot, r, multiplier,
                                     min_liquid_strikes=2)

        # Hand-calculate expected GEX
        expected_gex = 0.0
        for i, K in enumerate(strikes):
            # Call contribution (dealer short calls -> sign = +1)
            cg = _analytic_gamma(spot, K, r, T, call_ivs[i])
            expected_gex += 1 * cg * call_oi[i] * multiplier * spot**2 / 100

            # Put contribution (dealer short puts -> sign = -1)
            pg = _analytic_gamma(spot, K, r, T, put_ivs[i])
            expected_gex += -1 * pg * put_oi[i] * multiplier * spot**2 / 100

        assert abs(result.gex - expected_gex) < 1e-4

    def test_full_chain_20_strikes_non_nan(self):
        """Full chain (20 strikes): GEX, vanna, charm all non-NaN."""
        n = 20
        spot = 100.0
        strikes = [80.0 + 2.0 * i for i in range(n)]
        call_ivs = [0.35 - 0.005 * i for i in range(n)]  # skew
        put_ivs = [0.36 - 0.005 * i for i in range(n)]
        call_oi = [100 + 10 * i for i in range(n)]
        put_oi = [120 + 8 * i for i in range(n)]
        T = 0.25

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [T] * n)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0)

        assert math.isfinite(result.gex)
        assert math.isfinite(result.vanna_flow)
        assert math.isfinite(result.charm_flow)
        assert not math.isnan(result.gex)
        assert not math.isnan(result.vanna_flow)
        assert not math.isnan(result.charm_flow)

    def test_sparse_chain_returns_degraded(self):
        """Sparse chain (5 liquid strikes): returns quality=DEGRADED, flows=0.0."""
        n = 5
        spot = 100.0
        strikes = [90.0 + 5.0 * i for i in range(n)]
        call_ivs = [0.30] * n
        put_ivs = [0.30] * n
        call_oi = [100] * n
        put_oi = [100] * n

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * n)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0,
                                     min_liquid_strikes=8)

        assert result.quality.value == "degraded"
        assert result.gex == 0.0
        assert result.vanna_flow == 0.0
        assert result.charm_flow == 0.0
        assert result.liquid_strike_count == 5
        assert len(result.warnings) > 0

    def test_zero_oi_strikes_skipped(self):
        """Strikes with zero OI should be skipped without error."""
        spot = 100.0
        strikes = [95.0, 100.0, 105.0, 110.0, 115.0,
                   120.0, 125.0, 130.0, 135.0, 140.0]
        call_ivs = [0.30] * 10
        put_ivs = [0.30] * 10
        call_oi = [0, 100, 0, 200, 0, 100, 200, 100, 100, 100]  # some zero
        put_oi = [100, 0, 200, 0, 100, 100, 100, 100, 100, 100]  # some zero

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * 10)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0)

        # Should complete without error, flows should be finite
        assert math.isfinite(result.gex)
        assert math.isfinite(result.vanna_flow)
        assert math.isfinite(result.charm_flow)

    def test_expired_options_skipped(self):
        """Options with T <= 0 are skipped without error."""
        spot = 100.0
        strikes = [95.0, 100.0, 105.0, 110.0, 115.0,
                   120.0, 125.0, 130.0, 135.0, 140.0]
        call_ivs = [0.30] * 10
        put_ivs = [0.30] * 10
        call_oi = [100] * 10
        put_oi = [100] * 10
        # Mix of expired (T<=0), valid, and too-far-out (T>2.0)
        expiries = [-0.1, 0.0, 0.25, 0.25, 0.25,
                    0.25, 0.25, 0.25, 0.25, 2.5]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            expiries)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0)

        # Should complete without error
        assert math.isfinite(result.gex)

    def test_sign_convention_calls_positive_gex(self):
        """Dealer short calls -> positive GEX contribution."""
        spot = 100.0
        # Chain with ONLY calls (no puts)
        strikes = [95.0, 97.0, 99.0, 101.0, 103.0,
                   105.0, 107.0, 109.0, 111.0, 113.0]
        call_ivs = [0.30] * 10
        put_ivs = [0.0] * 10  # no put IVs
        call_oi = [100] * 10
        put_oi = [0] * 10  # no put OI

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * 10)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0)

        # GEX should be positive (dealer short calls -> positive gamma)
        assert result.gex > 0

    def test_sign_convention_puts_negative_gex(self):
        """Dealer short puts -> negative GEX contribution."""
        spot = 100.0
        # Chain with ONLY puts (no calls)
        strikes = [87.0, 89.0, 91.0, 93.0, 95.0,
                   97.0, 99.0, 101.0, 103.0, 105.0]
        call_ivs = [0.0] * 10  # no call IVs
        put_ivs = [0.30] * 10
        call_oi = [0] * 10  # no call OI
        put_oi = [100] * 10

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * 10)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0)

        # GEX should be negative (dealer short puts -> negative gamma)
        assert result.gex < 0

    def test_returns_greeks_flow_result_dataclass(self):
        """Result should be a GreeksFlowResult dataclass with expected fields."""
        chain = _make_chain(
            strikes=[95.0, 97.0, 99.0, 101.0, 103.0,
                     105.0, 107.0, 109.0, 111.0, 113.0],
            call_ivs=[0.30] * 10,
            put_ivs=[0.30] * 10,
            call_oi=[100] * 10,
            put_oi=[100] * 10,
            expiries_T=[0.25] * 10,
        )

        result = compute_greeks_flow(chain, 100.0, 0.05, 400.0)

        assert isinstance(result, GreeksFlowResult)
        assert hasattr(result, "gex")
        assert hasattr(result, "vanna_flow")
        assert hasattr(result, "charm_flow")
        assert hasattr(result, "quality")
        assert hasattr(result, "liquid_strike_count")
        assert hasattr(result, "warnings")

    def test_wide_spread_strikes_excluded_from_liquid_count(self):
        """Strikes with bid-ask spread > max_spread_pct excluded from liquid count."""
        n = 10
        spot = 100.0
        strikes = [90.0 + 3.0 * i for i in range(n)]
        call_ivs = [0.30] * n
        put_ivs = [0.30] * n
        call_oi = [100] * n
        put_oi = [100] * n
        # 6 have tight spreads, 4 have wide spreads
        spreads = [0.05, 0.05, 0.05, 0.30, 0.30, 0.30, 0.30, 0.05, 0.05, 0.05]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * n, bid_ask_spread_pct=spreads)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0,
                                     min_liquid_strikes=8, max_spread_pct=0.20)

        # Only 6 liquid strikes (< 8 required), so should be DEGRADED
        assert result.quality.value == "degraded"
        assert result.liquid_strike_count == 6

    def test_low_oi_strikes_excluded_from_liquid_count(self):
        """Strikes with OI < min_oi excluded from liquid count."""
        n = 10
        spot = 100.0
        strikes = [90.0 + 3.0 * i for i in range(n)]
        call_ivs = [0.30] * n
        put_ivs = [0.30] * n
        # 5 with sufficient OI, 5 with low OI
        call_oi = [100, 100, 100, 10, 10, 10, 10, 10, 100, 100]
        put_oi = [100, 100, 100, 10, 10, 10, 10, 10, 100, 100]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [0.25] * n)

        result = compute_greeks_flow(chain, spot, 0.05, 400.0,
                                     min_liquid_strikes=8, min_oi=50)

        # Only 5 strikes have OI >= 50 (< 8), so DEGRADED
        assert result.quality.value == "degraded"
        assert result.liquid_strike_count == 5

    def test_two_strike_vanna_matches_hand_calc(self):
        """Simple 2-strike chain: verify vanna flow matches hand calculation."""
        spot = 100.0
        r = 0.05
        T = 0.25
        multiplier = 400.0

        strikes = [95.0, 105.0]
        call_ivs = [0.30, 0.28]
        put_ivs = [0.32, 0.29]
        call_oi = [100, 200]
        put_oi = [150, 100]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [T, T])

        # Use min_liquid_strikes=2 to bypass degradation for this unit test
        result = compute_greeks_flow(chain, spot, r, multiplier,
                                     min_liquid_strikes=2)

        # Hand-calculate expected vanna flow
        expected_vanna = 0.0
        for i, K in enumerate(strikes):
            cv = _analytic_vanna(spot, K, r, T, call_ivs[i])
            expected_vanna += 1 * cv * call_oi[i] * multiplier * spot

            pv = _analytic_vanna(spot, K, r, T, put_ivs[i])
            expected_vanna += -1 * pv * put_oi[i] * multiplier * spot

        assert abs(result.vanna_flow - expected_vanna) < 1e-2

    def test_two_strike_charm_matches_hand_calc(self):
        """Simple 2-strike chain: verify charm flow matches hand calculation."""
        spot = 100.0
        r = 0.05
        T = 0.25
        multiplier = 400.0

        strikes = [95.0, 105.0]
        call_ivs = [0.30, 0.28]
        put_ivs = [0.32, 0.29]
        call_oi = [100, 200]
        put_oi = [150, 100]

        chain = _make_chain(strikes, call_ivs, put_ivs, call_oi, put_oi,
                            [T, T])

        # Use min_liquid_strikes=2 to bypass degradation for this unit test
        result = compute_greeks_flow(chain, spot, r, multiplier,
                                     min_liquid_strikes=2)

        # Hand-calculate expected charm flow
        expected_charm = 0.0
        for i, K in enumerate(strikes):
            cc = _analytic_charm(spot, K, r, T, call_ivs[i])
            expected_charm += 1 * cc * call_oi[i] * multiplier

            pc = _analytic_charm(spot, K, r, T, put_ivs[i])
            expected_charm += -1 * pc * put_oi[i] * multiplier

        assert abs(result.charm_flow - expected_charm) < 1e-2
