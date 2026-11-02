"""Parameter tuning on TRAINING seasons only.

Rating parameters are chosen to minimise the squared error of predicted
margins (a forecasting objective, not a betting one: far harder to overfit
than picking whatever made the most money). The edge threshold is then chosen
on training-season bets. The holdout seasons are never looked at here.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .backtest import BetRules, run_backtest
from .config import SportConfig
from .model import fit_walk_forward
from .ratings import run_margin_elo
from .stats import summarize


def tune_ratings(df: pd.DataFrame, cfg: SportConfig, train_seasons: list[int]) -> tuple[dict, pd.DataFrame]:
    mask = (df["played"] & df["season"].isin(train_seasons)).to_numpy()
    y = df["result"].to_numpy()[mask]
    rows = []
    keys = list(cfg.grid)
    for combo in itertools.product(*[cfg.grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        preds, _ = run_margin_elo(df, **params)
        rmse = float(np.sqrt(np.mean((y - preds[mask]) ** 2)))
        rows.append({**params, "rmse": rmse})
    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best = {k: float(table.loc[0, k]) for k in keys}
    return best, table


def tune_threshold(df: pd.DataFrame, cfg: SportConfig, params: dict, train_seasons: list[int],
                   rules: BetRules, grid=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08),
                   min_bets: int = 150) -> tuple[float, pd.DataFrame]:
    model_df, _ = fit_walk_forward(df, cfg, params)
    rows = []
    for t in grid:
        r = BetRules(**{**rules.__dict__, "min_edge": t})
        _, bets = run_backtest(model_df, r, train_seasons)
        n = len(bets)
        roi = bets["units"].mean() if n else np.nan
        s = summarize(bets) if n else {"p_value": np.nan, "win_rate": np.nan}
        rows.append({"min_edge": t, "bets": n, "win_rate": s["win_rate"], "roi": roi,
                     "p_value": s["p_value"]})
    table = pd.DataFrame(rows)
    # A threshold is only adopted if its training record beats break-even with
    # p < 0.05 (Bonferroni-corrected for the number of thresholds tried).
    # Otherwise there's no edge to tune, and picking the best-looking row would
    # just be fitting noise, so fall back to the sport's pre-set default.
    alpha = 0.05 / len(grid)
    ok = table[(table["bets"] >= min_bets) & (table["p_value"] < alpha)]
    best = float(ok.sort_values("roi", ascending=False).iloc[0]["min_edge"]) if len(ok) else cfg.min_edge
    table["chosen"] = table["min_edge"] == best
    return best, table
