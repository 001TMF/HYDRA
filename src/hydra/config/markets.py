"""Market registry: canonical configuration for all 14 supported markets.

Each market has a unique symbol, exchange, CFTC code, contract multiplier,
strike range, options availability flag, and tier. Tier 1 markets are
prioritized for live trading; Tier 2 and 3 are monitored but not
immediately activated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    """Immutable configuration for a single futures market.

    Attributes
    ----------
    symbol : str
        Ticker symbol used throughout the system (e.g., "HE").
    exchange : str
        Exchange identifier for IB routing (e.g., "GLOBEX", "CBOT").
    cftc_code : str
        CFTC commitment of traders report code (zero-padded string).
    multiplier : float
        Contract dollar multiplier (e.g., 400 for lean hogs).
    strike_range : float
        Options strike range step size (market-specific).
    has_options : bool
        Whether options are available and tracked for this market.
    tier : int
        Priority tier: 1=live, 2=monitor, 3=watch.
    """

    symbol: str
    exchange: str
    cftc_code: str
    multiplier: float
    strike_range: float
    has_options: bool
    tier: int


MARKETS: dict[str, MarketConfig] = {
    "HE": MarketConfig(
        symbol="HE",
        exchange="GLOBEX",
        cftc_code="054642",
        multiplier=400,
        strike_range=15,
        has_options=True,
        tier=1,
    ),
    "LE": MarketConfig(
        symbol="LE",
        exchange="GLOBEX",
        cftc_code="057642",
        multiplier=400,
        strike_range=15,
        has_options=True,
        tier=1,
    ),
    "GF": MarketConfig(
        symbol="GF",
        exchange="GLOBEX",
        cftc_code="061641",
        multiplier=500,
        strike_range=15,
        has_options=True,
        tier=1,
    ),
    "DC": MarketConfig(
        symbol="DC",
        exchange="GLOBEX",
        cftc_code="052641",
        multiplier=2000,
        strike_range=5,
        has_options=True,
        tier=2,
    ),
    "ZC": MarketConfig(
        symbol="ZC",
        exchange="CBOT",
        cftc_code="002602",
        multiplier=50,
        strike_range=50,
        has_options=True,
        tier=1,
    ),
    "ZW": MarketConfig(
        symbol="ZW",
        exchange="CBOT",
        cftc_code="001602",
        multiplier=50,
        strike_range=50,
        has_options=True,
        tier=1,
    ),
    "ZM": MarketConfig(
        symbol="ZM",
        exchange="CBOT",
        cftc_code="026603",
        multiplier=100,
        strike_range=30,
        has_options=True,
        tier=2,
    ),
    "ZL": MarketConfig(
        symbol="ZL",
        exchange="CBOT",
        cftc_code="007601",
        multiplier=600,
        strike_range=5,
        has_options=True,
        tier=2,
    ),
    "KE": MarketConfig(
        symbol="KE",
        exchange="CBOT",
        cftc_code="001612",
        multiplier=50,
        strike_range=50,
        has_options=True,
        tier=2,
    ),
    "CC": MarketConfig(
        symbol="CC",
        exchange="NYBOT",
        cftc_code="073732",
        multiplier=10,
        strike_range=500,
        has_options=True,
        tier=2,
    ),
    "KC": MarketConfig(
        symbol="KC",
        exchange="NYBOT",
        cftc_code="083731",
        multiplier=375,
        strike_range=30,
        has_options=True,
        tier=2,
    ),
    "CT": MarketConfig(
        symbol="CT",
        exchange="NYBOT",
        cftc_code="033661",
        multiplier=500,
        strike_range=10,
        has_options=True,
        tier=2,
    ),
    "SB": MarketConfig(
        symbol="SB",
        exchange="NYBOT",
        cftc_code="080732",
        multiplier=1120,
        strike_range=5,
        has_options=True,
        tier=2,
    ),
    "OJ": MarketConfig(
        symbol="OJ",
        exchange="NYBOT",
        cftc_code="040701",
        multiplier=150,
        strike_range=20,
        has_options=True,
        tier=3,
    ),
}


def get_active_markets(tiers: list[int]) -> list[MarketConfig]:
    """Return all markets belonging to the specified tiers.

    Parameters
    ----------
    tiers : list[int]
        List of tier numbers to include (e.g., [1, 2]).

    Returns
    -------
    list[MarketConfig]
        Sorted by tier then symbol for deterministic ordering.
    """
    return sorted(
        [m for m in MARKETS.values() if m.tier in tiers],
        key=lambda m: (m.tier, m.symbol),
    )
