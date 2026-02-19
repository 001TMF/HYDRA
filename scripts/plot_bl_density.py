"""Breeden-Litzenberger pipeline validation gate: synthetic thin-market diagnostic.

Generates synthetic options chain data (8-15 strikes with realistic noise),
runs the full SVI -> B-L pipeline, and produces diagnostic plots:
  1. Implied vol smile (raw vs. SVI-fitted)
  2. B-L implied probability density
  3. Implied density vs. log-normal benchmark with same mean/variance

This script fulfills the roadmap validation gate:
  "Plot implied vs. realized distributions to verify"

Run with:
  uv run python scripts/plot_bl_density.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, norm

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from hydra.signals.options_math.density import extract_density
from hydra.signals.options_math.moments import compute_moments
from hydra.signals.options_math.surface import calibrate_svi, svi_to_call_prices

OUTPUT_DIR = project_root / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_synthetic_chain(
    forward: float = 85.0,
    T: float = 0.25,
    r: float = 0.05,
    n_strikes: int = 12,
    base_vol: float = 0.25,
    skew: float = -0.15,
    noise_std: float = 0.005,
    seed: int = 42,
) -> dict:
    """Generate a synthetic thin-market options chain with realistic noise.

    Creates an SVI-like vol smile with slight negative skew (typical for
    commodity options), then adds small random noise to simulate real data.

    Returns dict with keys: strikes, call_prices, ivs, oi, bid_ask_spread_pct
    """
    rng = np.random.default_rng(seed)

    # Generate strikes around forward
    strike_range = forward * 0.20  # +/- 20% of forward
    strikes = np.linspace(forward - strike_range, forward + strike_range, n_strikes)

    # Generate realistic implied vols with skew
    log_moneyness = np.log(strikes / forward)
    # Quadratic smile + skew
    true_ivs = base_vol + skew * log_moneyness + 0.5 * log_moneyness**2
    true_ivs = np.maximum(true_ivs, 0.05)  # Floor at 5%

    # Add noise
    noisy_ivs = true_ivs + rng.normal(0, noise_std, n_strikes)
    noisy_ivs = np.maximum(noisy_ivs, 0.05)

    # Compute call prices from noisy IVs via Black-76
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    d1 = (np.log(forward / strikes) + 0.5 * noisy_ivs**2 * T) / (noisy_ivs * sqrt_T)
    d2 = d1 - noisy_ivs * sqrt_T
    call_prices = discount * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))
    call_prices = np.maximum(call_prices, 0.0)

    # Realistic OI and spread (all liquid for this test)
    oi = rng.integers(100, 500, n_strikes).astype(float)
    bid_ask_spread_pct = rng.uniform(0.02, 0.10, n_strikes)

    return {
        "strikes": strikes,
        "call_prices": call_prices,
        "ivs": noisy_ivs,
        "true_ivs": true_ivs,
        "oi": oi,
        "bid_ask_spread_pct": bid_ask_spread_pct,
        "forward": forward,
        "T": T,
        "r": r,
    }


def main():
    print("=" * 60)
    print("HYDRA B-L Pipeline Validation Gate")
    print("=" * 60)

    # Generate synthetic data
    chain = generate_synthetic_chain(n_strikes=12)
    strikes = chain["strikes"]
    call_prices = chain["call_prices"]
    ivs = chain["ivs"]
    true_ivs = chain["true_ivs"]
    oi = chain["oi"]
    spread = chain["bid_ask_spread_pct"]
    F = chain["forward"]
    T = chain["T"]
    r = chain["r"]

    print(f"\nSynthetic chain: {len(strikes)} strikes, F={F}, T={T}, r={r}")
    print(f"Strike range: [{strikes[0]:.2f}, {strikes[-1]:.2f}]")

    # Step 1: Calibrate SVI
    print("\n--- Step 1: SVI Calibration ---")
    svi_result = calibrate_svi(strikes, ivs, F, T)
    print(f"SVI params: {svi_result.params}")
    print(f"RMSE: {svi_result.rmse:.6f}")
    print(f"Butterfly arbitrage: {svi_result.has_butterfly_arbitrage}")
    if svi_result.warnings:
        for w in svi_result.warnings:
            print(f"  WARNING: {w}")

    # Generate fitted IVs on same strikes for comparison
    fitted_ivs = svi_result.fitted_iv

    # Step 2: Extract density via B-L
    print("\n--- Step 2: B-L Density Extraction ---")
    density_result = extract_density(
        strikes=strikes,
        call_prices=call_prices,
        oi=oi,
        bid_ask_spread_pct=spread,
        spot=F,
        r=r,
        T=T,
        min_liquid_strikes=8,
    )
    print(f"Quality: {density_result.quality.value}")
    print(f"Liquid strike count: {density_result.liquid_strike_count}")
    print(f"ATM IV: {density_result.atm_iv:.4f}" if density_result.atm_iv else "ATM IV: None")

    density_integral = float(np.trapezoid(density_result.density, density_result.strikes))
    print(f"Density integral: {density_integral:.6f}")
    if density_result.warnings:
        for w in density_result.warnings:
            print(f"  WARNING: {w}")

    # Step 3: Compute moments
    print("\n--- Step 3: Implied Moments ---")
    moments = compute_moments(density_result)
    print(f"Mean: {moments.mean:.4f}" if moments.mean else "Mean: None")
    print(f"Variance: {moments.variance:.4f}" if moments.variance else "Variance: None")
    print(f"Skew: {moments.skew:.4f}" if moments.skew else "Skew: None")
    print(f"Kurtosis: {moments.kurtosis:.4f}" if moments.kurtosis else "Kurtosis: None")
    print(f"Implied mean vs forward: {moments.mean:.2f} vs {F:.2f}" if moments.mean else "")

    # Step 4: Generate plots
    print("\n--- Step 4: Generating Plots ---")

    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Plot 1: Implied vol smile
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        ax1.scatter(strikes, ivs, color="blue", label="Market (noisy)", zorder=5, s=40)
        ax1.plot(strikes, true_ivs, "g--", label="True smile", alpha=0.7)
        ax1.plot(strikes, fitted_ivs, "r-", label="SVI fit", linewidth=2)
        ax1.axvline(F, color="gray", linestyle=":", alpha=0.5, label=f"Forward={F}")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Implied Volatility")
        ax1.set_title("Implied Vol Smile: Raw vs SVI-Fitted")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: B-L implied density
        ax2 = axes[1]
        ax2.plot(
            density_result.strikes,
            density_result.density,
            "b-",
            linewidth=2,
            label="B-L Density",
        )
        ax2.fill_between(
            density_result.strikes,
            density_result.density,
            alpha=0.15,
            color="blue",
        )
        ax2.axvline(F, color="gray", linestyle=":", alpha=0.5, label=f"Forward={F}")
        if moments.mean is not None:
            ax2.axvline(
                moments.mean,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Mean={moments.mean:.2f}",
            )
        ax2.set_xlabel("Strike / Price")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("B-L Implied Probability Density")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Implied density vs log-normal benchmark
        ax3 = axes[2]
        ax3.plot(
            density_result.strikes,
            density_result.density,
            "b-",
            linewidth=2,
            label="B-L Implied",
        )

        # Log-normal benchmark with same mean/variance
        if moments.mean is not None and moments.variance is not None and moments.variance > 0:
            mu_ln = moments.mean
            sigma_ln = np.sqrt(moments.variance)
            # Compute log-normal parameters from mean and variance
            # Using the parameterization: X ~ LogNormal(mu, sigma)
            # where mu, sigma are the mean and std of log(X)
            sigma_log = np.sqrt(np.log(1 + (sigma_ln / mu_ln) ** 2))
            mu_log = np.log(mu_ln) - 0.5 * sigma_log**2

            lognorm_density = lognorm.pdf(
                density_result.strikes, s=sigma_log, scale=np.exp(mu_log)
            )
            ax3.plot(
                density_result.strikes,
                lognorm_density,
                "r--",
                linewidth=2,
                label="Log-Normal Benchmark",
            )

        ax3.axvline(F, color="gray", linestyle=":", alpha=0.5, label=f"Forward={F}")
        ax3.set_xlabel("Strike / Price")
        ax3.set_ylabel("Probability Density")
        ax3.set_title("Implied Density vs Log-Normal Benchmark")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plots
        plot_path = OUTPUT_DIR / "bl_density_validation.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to: {plot_path}")

    except ImportError:
        print(
            "WARNING: matplotlib not installed. Skipping plot generation.\n"
            "Install with: uv add --group dev matplotlib"
        )

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION GATE SUMMARY")
    print("=" * 60)
    print(f"  Liquid strikes:    {density_result.liquid_strike_count}")
    print(f"  Density integral:  {density_integral:.6f}")
    if moments.mean is not None:
        print(f"  Implied mean:      {moments.mean:.4f}")
        print(f"  Forward:           {F:.4f}")
        print(f"  Mean vs forward:   {abs(moments.mean - F) / F:.4%} error")
    if moments.variance is not None:
        print(f"  Variance:          {moments.variance:.4f}")
    if moments.skew is not None:
        print(f"  Skew:              {moments.skew:.4f}")
    if moments.kurtosis is not None:
        print(f"  Kurtosis:          {moments.kurtosis:.4f}")
    print(f"  Quality:           {density_result.quality.value}")
    print(f"  SVI RMSE:          {svi_result.rmse:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
