"""Margin-Elo: a points-based rating system (sport-agnostic).

Each team carries a rating in points relative to league average.
    predicted home margin = r_home - r_away + hfa (0 at neutral sites)
After the game, both ratings move by k * (actual - predicted), with the error
capped at +/- mov_cap so one blowout can't swing a rating too far.
At each new season, ratings shrink toward 0 by `regress`.

Every prediction is made BEFORE that game's result is used, so the output
column `model_margin` is free of look-ahead by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_margin_elo(df: pd.DataFrame, k: float, hfa: float, regress: float,
                   mov_cap: float) -> tuple[np.ndarray, dict]:
    """Return pre-game predicted home margins for every row, plus final ratings."""
    ratings: dict[str, float] = {}
    preds = np.full(len(df), np.nan)
    season_prev = None

    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    seasons = df["season"].to_numpy()
    neutral = df["neutral"].to_numpy()
    results = df["result"].to_numpy(dtype=float)

    for i in range(len(df)):
        s = seasons[i]
        if s != season_prev:
            if season_prev is not None:
                for t in ratings:
                    ratings[t] *= (1.0 - regress)
            season_prev = s
        h, a = homes[i], aways[i]
        rh = ratings.get(h, 0.0)
        ra = ratings.get(a, 0.0)
        pred = rh - ra + (0.0 if neutral[i] else hfa)
        preds[i] = pred
        res = results[i]
        if not np.isnan(res):
            err = max(-mov_cap, min(mov_cap, res - pred))
            ratings[h] = rh + k * err
            ratings[a] = ra - k * err
    return preds, ratings
