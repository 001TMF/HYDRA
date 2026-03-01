"""Tests for the multi-market configuration registry.

Verifies that all 14 commodity futures markets are registered with correct
metadata (exchange, CFTC code, multiplier, tier) and that the filtering
helper returns appropriate subsets.
"""

from __future__ import annotations

import pytest

from hydra.config.markets import MARKETS, MarketConfig, get_active_markets


# ---------------------------------------------------------------------------
# Registry completeness and structure
# ---------------------------------------------------------------------------


class TestMarketRegistry:
    """Verify all 14 markets are present with valid structural fields."""

    EXPECTED_SYMBOLS = {
        "HE", "LE", "GF", "DC",
        "ZC", "ZW", "ZM", "ZL", "KE",
        "CC", "KC", "CT", "SB", "OJ",
    }

    def test_all_14_markets_present(self):
        """MARKETS dict must contain exactly 14 entries."""
        assert len(MARKETS) == 14

    def test_expected_symbols_match(self):
        """MARKETS must contain exactly the expected set of symbols."""
        assert set(MARKETS.keys()) == self.EXPECTED_SYMBOLS

    def test_each_market_is_market_config(self):
        """Every value in MARKETS must be a MarketConfig instance."""
        for sym, cfg in MARKETS.items():
            assert isinstance(cfg, MarketConfig), (
                f"{sym!r} maps to {type(cfg).__name__}, expected MarketConfig"
            )

    def test_each_config_symbol_matches_key(self):
        """MarketConfig.symbol must equal its MARKETS dict key."""
        for sym, cfg in MARKETS.items():
            assert cfg.symbol == sym, (
                f"MARKETS[{sym!r}].symbol == {cfg.symbol!r}"
            )

    def test_each_exchange_is_valid(self):
        """Exchange must be one of GLOBEX, CBOT, or NYBOT."""
        valid_exchanges = {"GLOBEX", "CBOT", "NYBOT"}
        for sym, cfg in MARKETS.items():
            assert cfg.exchange in valid_exchanges, (
                f"{sym}: unexpected exchange {cfg.exchange!r}"
            )

    def test_cftc_codes_are_six_digits(self):
        """Each CFTC code must be exactly 6 characters (zero-padded)."""
        for sym, cfg in MARKETS.items():
            assert len(cfg.cftc_code) == 6, (
                f"{sym}: cftc_code {cfg.cftc_code!r} is not 6 characters"
            )
            assert cfg.cftc_code.isdigit(), (
                f"{sym}: cftc_code {cfg.cftc_code!r} is not numeric"
            )

    def test_cftc_codes_unique(self):
        """No two markets may share the same CFTC code."""
        codes = [cfg.cftc_code for cfg in MARKETS.values()]
        assert len(codes) == len(set(codes)), "Duplicate CFTC codes detected"

    def test_multiplier_positive(self):
        """Every contract multiplier must be > 0."""
        for sym, cfg in MARKETS.items():
            assert cfg.multiplier > 0, f"{sym}: multiplier {cfg.multiplier} <= 0"

    def test_tier_in_valid_range(self):
        """Tier must be in {1, 2, 3}."""
        for sym, cfg in MARKETS.items():
            assert cfg.tier in {1, 2, 3}, (
                f"{sym}: tier {cfg.tier} not in {{1, 2, 3}}"
            )

    def test_has_options_is_bool(self):
        """has_options must be a bool, not an int truthy value."""
        for sym, cfg in MARKETS.items():
            assert isinstance(cfg.has_options, bool), (
                f"{sym}: has_options is {type(cfg.has_options).__name__}, not bool"
            )

    def test_frozen_dataclass_immutable(self):
        """MarketConfig must be frozen — mutation must raise an error."""
        cfg = MARKETS["HE"]
        with pytest.raises((AttributeError, TypeError)):
            cfg.symbol = "XX"  # type: ignore[misc]

    def test_strike_range_positive(self):
        """strike_range must be > 0 for all markets."""
        for sym, cfg in MARKETS.items():
            assert cfg.strike_range > 0, (
                f"{sym}: strike_range {cfg.strike_range} is not positive"
            )


# ---------------------------------------------------------------------------
# Per-market spot-checks against the authoritative table
# ---------------------------------------------------------------------------


class TestMarketSpotChecks:
    """Verify exact field values for representative markets in each exchange."""

    # --- GLOBEX livestock ---

    def test_he_config(self):
        """HE (Lean Hogs): GLOBEX, CFTC 054642, multiplier 400, tier 1."""
        he = MARKETS["HE"]
        assert he.exchange == "GLOBEX"
        assert he.cftc_code == "054642"
        assert he.multiplier == 400
        assert he.tier == 1

    def test_le_config(self):
        """LE (Live Cattle): GLOBEX, CFTC 057642, multiplier 400, tier 1."""
        le = MARKETS["LE"]
        assert le.exchange == "GLOBEX"
        assert le.cftc_code == "057642"
        assert le.multiplier == 400
        assert le.tier == 1

    def test_gf_config(self):
        """GF (Feeder Cattle): GLOBEX, CFTC 061641, multiplier 500, tier 1."""
        gf = MARKETS["GF"]
        assert gf.exchange == "GLOBEX"
        assert gf.cftc_code == "061641"
        assert gf.multiplier == 500
        assert gf.tier == 1

    def test_dc_config(self):
        """DC (Class III Milk): GLOBEX, CFTC 052641, multiplier 2000, tier 2."""
        dc = MARKETS["DC"]
        assert dc.exchange == "GLOBEX"
        assert dc.cftc_code == "052641"
        assert dc.multiplier == 2000
        assert dc.tier == 2

    # --- CBOT grains ---

    def test_zc_config(self):
        """ZC (Corn): CBOT, CFTC 002602, multiplier 50, tier 1."""
        zc = MARKETS["ZC"]
        assert zc.exchange == "CBOT"
        assert zc.cftc_code == "002602"
        assert zc.multiplier == 50
        assert zc.tier == 1

    def test_zw_config(self):
        """ZW (Chicago Wheat): CBOT, CFTC 001602, multiplier 50, tier 1."""
        zw = MARKETS["ZW"]
        assert zw.exchange == "CBOT"
        assert zw.cftc_code == "001602"
        assert zw.multiplier == 50
        assert zw.tier == 1

    def test_zm_config(self):
        """ZM (Soybean Meal): CBOT, CFTC 026603, multiplier 100, tier 2."""
        zm = MARKETS["ZM"]
        assert zm.exchange == "CBOT"
        assert zm.cftc_code == "026603"
        assert zm.multiplier == 100
        assert zm.tier == 2

    def test_zl_config(self):
        """ZL (Soybean Oil): CBOT, CFTC 007601, multiplier 600, tier 2."""
        zl = MARKETS["ZL"]
        assert zl.exchange == "CBOT"
        assert zl.cftc_code == "007601"
        assert zl.multiplier == 600
        assert zl.tier == 2

    def test_ke_config(self):
        """KE (KC HRW Wheat): CBOT, CFTC 001612, multiplier 50, tier 2."""
        ke = MARKETS["KE"]
        assert ke.exchange == "CBOT"
        assert ke.cftc_code == "001612"
        assert ke.multiplier == 50
        assert ke.tier == 2

    # --- NYBOT softs ---

    def test_cc_config(self):
        """CC (Cocoa): NYBOT, CFTC 073732, multiplier 10, tier 2."""
        cc = MARKETS["CC"]
        assert cc.exchange == "NYBOT"
        assert cc.cftc_code == "073732"
        assert cc.multiplier == 10
        assert cc.tier == 2

    def test_kc_config(self):
        """KC (Coffee): NYBOT, CFTC 083731, multiplier 375, tier 2."""
        kc = MARKETS["KC"]
        assert kc.exchange == "NYBOT"
        assert kc.cftc_code == "083731"
        assert kc.multiplier == 375
        assert kc.tier == 2

    def test_ct_config(self):
        """CT (Cotton): NYBOT, CFTC 033661, multiplier 500, tier 2."""
        ct = MARKETS["CT"]
        assert ct.exchange == "NYBOT"
        assert ct.cftc_code == "033661"
        assert ct.multiplier == 500
        assert ct.tier == 2

    def test_sb_config(self):
        """SB (Sugar #11): NYBOT, CFTC 080732, multiplier 1120, tier 2."""
        sb = MARKETS["SB"]
        assert sb.exchange == "NYBOT"
        assert sb.cftc_code == "080732"
        assert sb.multiplier == 1120
        assert sb.tier == 2

    def test_oj_config(self):
        """OJ (Orange Juice): NYBOT, CFTC 040701, multiplier 150, tier 3."""
        oj = MARKETS["OJ"]
        assert oj.exchange == "NYBOT"
        assert oj.cftc_code == "040701"
        assert oj.multiplier == 150
        assert oj.tier == 3


# ---------------------------------------------------------------------------
# get_active_markets filtering
# ---------------------------------------------------------------------------


class TestGetActiveMarkets:
    """Verify tier filtering logic of get_active_markets."""

    def test_tier1_only(self):
        """get_active_markets([1]) returns exactly tier-1 markets."""
        tier1 = get_active_markets(tiers=[1])
        syms = {m.symbol for m in tier1}
        assert syms == {"HE", "LE", "GF", "ZC", "ZW"}

    def test_tier1_count(self):
        """get_active_markets([1]) returns 5 markets."""
        assert len(get_active_markets(tiers=[1])) == 5

    def test_tier12_count(self):
        """get_active_markets([1, 2]) returns 13 markets (5 + 8)."""
        result = get_active_markets(tiers=[1, 2])
        assert len(result) == 13

    def test_tier12_excludes_oj(self):
        """get_active_markets([1, 2]) must not include OJ (tier 3)."""
        syms = {m.symbol for m in get_active_markets(tiers=[1, 2])}
        assert "OJ" not in syms

    def test_tier123_all_markets(self):
        """get_active_markets([1, 2, 3]) returns all 14 markets."""
        result = get_active_markets(tiers=[1, 2, 3])
        assert len(result) == 14

    def test_tier3_only(self):
        """get_active_markets([3]) returns only OJ."""
        tier3 = get_active_markets(tiers=[3])
        assert len(tier3) == 1
        assert tier3[0].symbol == "OJ"

    def test_returns_list_of_market_configs(self):
        """Return value must be a list of MarketConfig instances."""
        result = get_active_markets(tiers=[1])
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, MarketConfig)

    def test_empty_tiers_returns_empty(self):
        """get_active_markets([]) returns an empty list."""
        result = get_active_markets(tiers=[])
        assert result == []
