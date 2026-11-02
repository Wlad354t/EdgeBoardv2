"""Odds math and Kelly sizing (push-aware)."""
from __future__ import annotations

import math


def american_to_decimal(odds: float) -> float:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        odds = -110
    return 1 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def no_vig_home(home_odds, away_odds, default: int = -110):
    """No-vig probability that home covers (pushes excluded). Vectorised."""
    import numpy as np
    h = np.where(np.isnan(np.asarray(home_odds, float)), default, home_odds).astype(float)
    a = np.where(np.isnan(np.asarray(away_odds, float)), default, away_odds).astype(float)
    dec = lambda o: np.where(o > 0, 1 + o / 100.0, 1 + 100.0 / -o)
    ih, ia = 1 / dec(h), 1 / dec(a)
    return ih / (ih + ia)


def breakeven(odds: float) -> float:
    """Win rate (excluding pushes) needed to break even at these odds."""
    return 1.0 / american_to_decimal(odds)


def expected_value(p_win: float, p_loss: float, odds: float) -> float:
    """Expected profit per $1 staked. Pushes return the stake (EV 0)."""
    b = american_to_decimal(odds) - 1
    return p_win * b - p_loss


def kelly_fraction(p_win: float, p_loss: float, odds: float) -> float:
    """Full-Kelly bankroll fraction with three outcomes (win/push/loss).

    Maximises p_w*log(1+f*b) + p_l*log(1-f)  ->  f = (p_w*b - p_l) / (b*(p_w+p_l))
    """
    b = american_to_decimal(odds) - 1
    denom = b * (p_win + p_loss)
    if denom <= 0:
        return 0.0
    return max(0.0, (p_win * b - p_loss) / denom)
