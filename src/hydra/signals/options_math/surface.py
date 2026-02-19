"""SVI volatility surface calibration and interpolation.

Implements Gatheral's Stochastic Volatility Inspired (SVI) parameterization
for fitting implied volatility smiles from thin-market options chains with
sparse data (8-15 strikes).

The SVI total variance formula:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where:
    k = ln(K/F) is log-moneyness
    w = iv^2 * T is total implied variance

References:
    Gatheral, "The Volatility Surface" (2006)
    Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class SVICalibrationResult:
    """Result of SVI calibration to market implied volatilities.

    Attributes:
        params: Dict with SVI parameters {a, b, rho, m, sigma}.
        fitted_iv: Fitted implied volatilities at the input strikes.
        rmse: Root mean squared error between fitted and market IVs.
        has_butterfly_arbitrage: True if d2w/dk2 < 0 detected on fine grid.
        warnings: List of warning messages (e.g. sparse data, high RMSE).
    """

    params: dict
    fitted_iv: np.ndarray
    rmse: float
    has_butterfly_arbitrage: bool
    warnings: list[str] = field(default_factory=list)


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Compute SVI total implied variance w(k).

    Parameters:
        k: Log-moneyness array, k = ln(K/F).
        a: Vertical translation (overall variance level).
        b: Slope of the wings (must be >= 0).
        rho: Rotation parameter, controls skew (-1 < rho < 1).
        m: Horizontal translation (smile center).
        sigma: Curvature at the vertex (must be > 0).

    Returns:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    k = np.asarray(k, dtype=np.float64)
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff**2 + sigma**2))


def _check_butterfly_arbitrage(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    k_min: float,
    k_max: float,
    n_points: int = 500,
) -> bool:
    """Check for butterfly arbitrage by verifying d2w/dk2 >= 0.

    Butterfly arbitrage exists when the total variance surface is locally
    concave (second derivative negative), which implies negative probability
    density in the Breeden-Litzenberger framework.

    Returns:
        True if butterfly arbitrage is detected (bad), False if clean.
    """
    k_fine = np.linspace(k_min - 0.5, k_max + 0.5, n_points)
    w_fine = svi_total_variance(k_fine, a, b, rho, m, sigma)
    d2w = np.gradient(np.gradient(w_fine, k_fine), k_fine)
    return bool(np.any(d2w < -1e-10))


def calibrate_svi(
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    forward: float,
    T: float,
) -> SVICalibrationResult:
    """Fit SVI parameters to market implied volatilities for one expiry.

    Converts strikes and IVs to log-moneyness and total variance, then
    minimizes the sum of squared differences between model and market
    total variance using L-BFGS-B.

    Parameters:
        strikes: Strike prices array.
        market_ivs: Market implied volatilities at each strike.
        forward: Forward price F.
        T: Time to expiry in years.

    Returns:
        SVICalibrationResult with fitted parameters, IVs, RMSE, and
        butterfly arbitrage flag.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    market_ivs = np.asarray(market_ivs, dtype=np.float64)

    warnings: list[str] = []

    n_strikes = len(strikes)
    if n_strikes < 8:
        warnings.append(
            f"Very sparse data: {n_strikes} strikes (recommended >= 8). "
            "Calibration may be unreliable."
        )

    # Convert to log-moneyness and total variance
    k = np.log(strikes / forward)
    market_w = market_ivs**2 * T

    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        model_w = svi_total_variance(k, a, b, rho, m, sigma)
        return float(np.sum((model_w - market_w) ** 2))

    # Parameter bounds
    bounds = [
        (-1.0, 1.0),       # a: vertical level
        (0.0, 5.0),        # b: wing slope (non-negative)
        (-0.99, 0.99),     # rho: skew
        (-2.0, 2.0),       # m: horizontal shift
        (0.01, 5.0),       # sigma: curvature (positive)
    ]

    # Initial guess based on market data
    x0 = np.array([
        float(np.mean(market_w)),  # a: start at mean total variance
        0.1,                        # b: moderate wing slope
        -0.3,                       # rho: typical negative skew
        0.0,                        # m: centered
        0.5,                        # sigma: moderate curvature
    ])

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-14},
    )

    a, b, rho, m, sigma = result.x

    # Compute fitted values
    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)

    # Guard against negative total variance (can happen with poor fits)
    fitted_w_safe = np.maximum(fitted_w, 1e-12)
    fitted_iv = np.sqrt(fitted_w_safe / T)

    # RMSE in implied vol space
    rmse = float(np.sqrt(np.mean((fitted_iv - market_ivs) ** 2)))

    # Check butterfly arbitrage
    has_arb = _check_butterfly_arbitrage(a, b, rho, m, sigma, k.min(), k.max())

    if has_arb:
        warnings.append(
            "Butterfly arbitrage detected: d2w/dk2 < 0 in fitted surface. "
            "The implied density may have negative regions."
        )

    if rmse > 0.05:
        warnings.append(
            f"High RMSE: {rmse:.4f}. Fit quality may be insufficient for "
            "reliable density extraction."
        )

    params_dict = {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma),
    }

    return SVICalibrationResult(
        params=params_dict,
        fitted_iv=fitted_iv,
        rmse=rmse,
        has_butterfly_arbitrage=has_arb,
        warnings=warnings,
    )


def svi_to_call_prices(
    svi_params: dict,
    strikes: np.ndarray,
    forward: float,
    r: float,
    T: float,
) -> np.ndarray:
    """Convert fitted SVI parameters to smooth call prices via Black-76.

    This produces the smooth call price curve needed as input to
    Breeden-Litzenberger density extraction (Plan 04). Using SVI-smoothed
    IVs rather than raw market prices avoids the numerical instability
    that plagues B-L in thin markets.

    Parameters:
        svi_params: Dict with SVI parameters {a, b, rho, m, sigma}.
        strikes: Fine grid of strike prices for output.
        forward: Forward price F.
        r: Risk-free rate.
        T: Time to expiry in years.

    Returns:
        Array of Black-76 call prices at each strike.
    """
    strikes = np.asarray(strikes, dtype=np.float64)

    # Compute log-moneyness
    k = np.log(strikes / forward)

    # Get total variance from SVI
    w = svi_total_variance(
        k,
        svi_params["a"],
        svi_params["b"],
        svi_params["rho"],
        svi_params["m"],
        svi_params["sigma"],
    )

    # Convert total variance to implied vol
    w_safe = np.maximum(w, 1e-12)
    iv = np.sqrt(w_safe / T)

    # Black-76 call price: C = e^(-rT) * [F*N(d1) - K*N(d2)]
    discount = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    d1 = (np.log(forward / strikes) + 0.5 * iv**2 * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    call_prices = discount * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))

    # Ensure non-negative prices
    call_prices = np.maximum(call_prices, 0.0)

    return call_prices
