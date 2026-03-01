"""Options feature pipeline: reads from Parquet lake, writes to feature store.

Transforms raw options chain data (stored in Parquet) into 8 features:
  - implied_mean, implied_variance, implied_skew, implied_kurtosis, atm_iv
    (from Breeden-Litzenberger density via moments.py)
  - gex, vanna_flow, charm_flow
    (from Black-76 Greeks aggregation via greeks.py)

This pipeline is fully synchronous — it performs no IB or network calls.
It reads from the ParquetLake, computes math, and writes to FeatureStore.
"""

from __future__ import annotations

import structlog
from datetime import datetime, date
from typing import Optional

import numpy as np

from hydra.data.ingestion.base import IngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake
from hydra.signals.options_math.density import (
    DataQuality,
    extract_density,
    _implied_vol_from_price,
)
from hydra.signals.options_math.moments import compute_moments
from hydra.signals.options_math.greeks import compute_greeks_flow

logger = structlog.get_logger()


class OptionsFeaturePipeline(IngestPipeline):
    """Compute options-derived features from a stored Parquet options chain.

    Reads the latest options chain from ParquetLake, computes density moments
    and Greeks flow metrics, then writes 8 features to the FeatureStore.

    Parameters
    ----------
    parquet_lake : ParquetLake
        Data lake containing raw options chains under data_type="options".
    feature_store : FeatureStore
        Point-in-time feature store where computed features are written.
    risk_free_rate : float
        Annualised risk-free rate (default 0.05).
    multiplier : float
        Contract dollar multiplier for the market (default 400 for HE).
    """

    def __init__(
        self,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
        risk_free_rate: float = 0.05,
        multiplier: float = 400,
    ) -> None:
        super().__init__(parquet_lake, feature_store)
        self.risk_free_rate = risk_free_rate
        self.multiplier = multiplier

    # ------------------------------------------------------------------
    # Abstract stubs — not used by this pipeline's run() override
    # ------------------------------------------------------------------

    def fetch(self, market: str, date: datetime) -> dict:
        raise NotImplementedError("OptionsFeaturePipeline uses run() directly")

    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        raise NotImplementedError("OptionsFeaturePipeline uses run() directly")

    def persist(self, data: dict, market: str, date: datetime) -> None:
        raise NotImplementedError("OptionsFeaturePipeline uses run() directly")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, market: str, date: datetime) -> bool:  # type: ignore[override]
        """Execute the full options feature pipeline.

        Steps:
          1. Read options chain from ParquetLake.
          2. Get spot price from FeatureStore.
          3. Find the front-month expiry and filter rows.
          4. Build sorted parallel arrays (strikes, call prices, OI, spread).
          5. Compute T (time to expiry in years).
          6. Call extract_density → compute_moments → write 5 features.
          7. Invert IVs for each call/put mid price via Black-76.
          8. Call compute_greeks_flow → write 3 features.

        Parameters
        ----------
        market : str
            Market identifier (e.g., "HE").
        date : datetime
            Reference datetime (timezone-aware UTC).

        Returns
        -------
        bool
            True on success, False on any unhandled exception.
        """
        log = logger.bind(pipeline="OptionsFeaturePipeline", market=market, date=str(date))
        try:
            log.info("options_features_started")

            # Step 1: Read options chain
            table = self.parquet_lake.read("options", market)
            if len(table) == 0:
                log.warning("options_features_no_data")
                return False

            records = table.to_pydict()

            # Step 2: Get spot price
            features = self.feature_store.get_features_at(market, date)
            spot_key = f"futures_close_{market}"
            spot = features.get(spot_key)
            if spot is None or spot <= 0:
                log.warning("options_features_no_spot", key=spot_key)
                return False

            # Step 3: Find front-month expiry (nearest date with valid T > 0)
            expiries = records.get("expiry", [])
            if not expiries:
                log.warning("options_features_no_expiry_column")
                return False

            ref_date = date.date() if isinstance(date, datetime) else date
            unique_expiries = sorted(set(expiries))
            front_expiry = self._select_front_expiry(unique_expiries, ref_date)
            if front_expiry is None:
                log.warning("options_features_no_valid_expiry")
                return False

            expiry_date = self._parse_expiry(front_expiry)
            T = (expiry_date - ref_date).days / 365.0
            if T <= 0:
                log.warning("options_features_expired", expiry=front_expiry, T=T)
                return False

            # Step 4: Filter to front-month expiry, group by strike
            rows_for_expiry = [
                i for i, e in enumerate(expiries) if e == front_expiry
            ]

            call_by_strike: dict[float, dict] = {}
            put_by_strike: dict[float, dict] = {}

            for i in rows_for_expiry:
                strike = float(records["strike"][i])
                bid = float(records["bid"][i] or 0)
                ask = float(records["ask"][i] or 0)
                oi = float(records["oi"][i] or 0)
                is_call = bool(records["is_call"][i])

                mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
                spread_pct = (ask - bid) / mid if mid > 1e-12 else 1.0

                row_data = {"mid": mid, "oi": oi, "spread_pct": spread_pct, "bid": bid, "ask": ask}
                if is_call:
                    call_by_strike[strike] = row_data
                else:
                    put_by_strike[strike] = row_data

            # Only use strikes where we have both a call and a put
            common_strikes = sorted(set(call_by_strike) & set(put_by_strike))
            if len(common_strikes) < 3:
                log.warning("options_features_too_few_strikes", count=len(common_strikes))
                return False

            strikes_arr = np.array(common_strikes, dtype=np.float64)
            call_mids = np.array([call_by_strike[k]["mid"] for k in common_strikes], dtype=np.float64)
            call_oi = np.array([call_by_strike[k]["oi"] for k in common_strikes], dtype=np.float64)
            put_oi = np.array([put_by_strike[k]["oi"] for k in common_strikes], dtype=np.float64)
            spread_pct_arr = np.array([call_by_strike[k]["spread_pct"] for k in common_strikes], dtype=np.float64)

            r = self.risk_free_rate

            # Step 5 & 6: Density → moments → write 5 features
            density_result = extract_density(
                strikes=strikes_arr,
                call_prices=call_mids,
                oi=call_oi,
                bid_ask_spread_pct=spread_pct_arr,
                spot=spot,
                r=r,
                T=T,
            )

            for w in density_result.warnings:
                log.warning("density_warning", detail=w)

            moments = compute_moments(density_result)
            for w in moments.warnings:
                log.warning("moments_warning", detail=w)

            quality_str = density_result.quality.value
            feature_pairs = [
                ("implied_mean", moments.mean),
                ("implied_variance", moments.variance),
                ("implied_skew", moments.skew),
                ("implied_kurtosis", moments.kurtosis),
                ("atm_iv", moments.atm_iv),
            ]
            for fname, fval in feature_pairs:
                if fval is not None:
                    self.feature_store.write_feature(
                        market=market,
                        feature_name=fname,
                        as_of=date,
                        available_at=date,
                        value=fval,
                        quality=quality_str,
                    )

            # Step 7: Invert IVs for greeks chain dict
            expiries_T_arr = np.full(len(common_strikes), T, dtype=np.float64)

            call_ivs = []
            for k in common_strikes:
                mid = call_by_strike[k]["mid"]
                iv = _implied_vol_from_price(mid, spot, k, r, T) if mid > 0 else None
                call_ivs.append(iv if iv is not None else 0.0)

            put_ivs = []
            for k in common_strikes:
                mid = put_by_strike[k]["mid"]
                # Use put-call parity to get synthetic call price, then invert
                # Or invert put directly — compute_greeks_flow accepts put IVs (both use same gamma formula)
                iv = self._invert_put_iv(mid, spot, k, r, T)
                put_ivs.append(iv if iv is not None else 0.0)

            chain = {
                "strikes": list(strikes_arr),
                "call_ivs": call_ivs,
                "put_ivs": put_ivs,
                "call_oi": list(call_oi),
                "put_oi": list(put_oi),
                "expiries_T": list(expiries_T_arr),
                "bid_ask_spread_pct": list(spread_pct_arr),
            }

            # Step 8: Greeks flow → write 3 features
            greeks_result = compute_greeks_flow(
                chain=chain,
                spot=spot,
                r=r,
                contract_multiplier=self.multiplier,
            )

            for w in greeks_result.warnings:
                log.warning("greeks_warning", detail=w)

            greeks_quality = greeks_result.quality.value
            greeks_features = [
                ("gex", greeks_result.gex),
                ("vanna_flow", greeks_result.vanna_flow),
                ("charm_flow", greeks_result.charm_flow),
            ]
            for fname, fval in greeks_features:
                self.feature_store.write_feature(
                    market=market,
                    feature_name=fname,
                    as_of=date,
                    available_at=date,
                    value=fval,
                    quality=greeks_quality,
                )

            log.info(
                "options_features_complete",
                liquid_strikes=density_result.liquid_strike_count,
                density_quality=quality_str,
                greeks_quality=greeks_quality,
                T=round(T, 4),
            )
            return True

        except Exception as e:
            log.error("options_features_failed", error=str(e), exc_info=True)
            return False

    async def run_async(self, market: str, date: datetime) -> bool:
        """Async wrapper so the runner can call this uniformly.

        Delegates synchronously to ``run()`` since this pipeline performs
        no async I/O.
        """
        return self.run(market, date)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_front_expiry(
        self, unique_expiries: list[str], ref_date: date
    ) -> Optional[str]:
        """Return the nearest expiry strictly after ref_date."""
        for exp_str in unique_expiries:
            try:
                exp_date = self._parse_expiry(exp_str)
                if exp_date > ref_date:
                    return exp_str
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _parse_expiry(expiry_str: str) -> date:
        """Parse an expiry string to a date.

        Supports ISO-8601 (YYYY-MM-DD) and YYYYMMDD formats.
        """
        expiry_str = expiry_str.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(expiry_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unrecognised expiry format: {expiry_str!r}")

    @staticmethod
    def _invert_put_iv(
        put_price: float,
        spot: float,
        strike: float,
        r: float,
        T: float,
    ) -> Optional[float]:
        """Invert a put mid-price to implied volatility via put-call parity.

        Converts the put price to a synthetic call price using:
            C = P + F*e^{-rT} - K*e^{-rT}
        where F ≈ spot, then calls _implied_vol_from_price on the synthetic call.
        """
        if put_price <= 0 or T <= 0:
            return None
        discount = np.exp(-r * T)
        synthetic_call = put_price + discount * (spot - strike)
        if synthetic_call <= 0:
            return None
        return _implied_vol_from_price(synthetic_call, spot, strike, r, T)
