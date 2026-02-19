"""Fractional Kelly position sizing with volume cap.

Implements:
    - fractional_kelly: Computes position size as fraction of capital using
      the Kelly criterion, capped at a configurable maximum.
    - volume_capped_position: Converts Kelly percentage to integer contracts,
      further capped by a percentage of average daily volume.

Used by the walk-forward backtester (Plan 02-05) and runtime risk layer.
"""


def fractional_kelly(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
    max_position_pct: float = 0.10,
) -> float:
    """Compute fractional Kelly position size as fraction of capital.

    Kelly formula: f* = (p * b - q) / b
    where b = avg_win / avg_loss, p = win_prob, q = 1 - win_prob.

    Returns fraction * f*, capped at max_position_pct.
    Negative Kelly (negative edge) returns 0.0.

    Args:
        win_prob: Probability of winning trade (0 to 1).
        avg_win: Average win amount (positive).
        avg_loss: Average loss amount (positive, representing magnitude).
        fraction: Kelly fraction to use (default 0.5 = half-Kelly).
        max_position_pct: Maximum position as fraction of capital (default 10%).

    Returns:
        Position size as fraction of capital, in [0, max_position_pct].
    """
    if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = avg_win / avg_loss  # win/loss ratio
    q = 1.0 - win_prob
    kelly = (win_prob * b - q) / b

    # Negative Kelly means expected value is negative -- don't trade
    if kelly <= 0:
        return 0.0

    sized = fraction * kelly
    return min(sized, max_position_pct)


def volume_capped_position(
    kelly_pct: float,
    capital: float,
    contract_value: float,
    avg_daily_volume: float,
    max_volume_pct: float = 0.02,
) -> int:
    """Convert Kelly fraction to integer contracts, capped by volume.

    Args:
        kelly_pct: Position size as fraction of capital (from fractional_kelly).
        capital: Total capital available.
        contract_value: Value per contract in currency units.
        avg_daily_volume: 20-day average daily volume in contracts.
        max_volume_pct: Maximum fraction of daily volume (default 2%).

    Returns:
        Number of contracts to trade (integer, rounded down).
    """
    kelly_contracts = int(kelly_pct * capital / contract_value)
    volume_cap = int(max_volume_pct * avg_daily_volume)
    return max(0, min(kelly_contracts, volume_cap))
