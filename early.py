"""Early-line analysis: where a ratings model can realistically have an edge.

Closing lines are the market's most informed price; a public-data ratings
model rarely beats them. Early lines (posted mid-week) are softer. Two tests:

1. Line-movement test: regress (closing - early) on (model - early). If
   b > 0 with a large t-stat, the market *later moves toward the model*:
   the model knows things before the market prices them. This is the
   statistical basis of closing-line value (CLV).
2. Walk-forward ATS test vs. early lines: fit the trust coefficient on past
   seasons only, bet the current season at the early line, grade on result,
   and record CLV (did the line move our way by kickoff?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import BetRules, grade, price_games
from .model import ols
from .probability import cover_probs, implied_mean
from .staking import no_vig_home


def line_movement_test(model_df: pd.DataFrame) -> dict:
    d = model_df[model_df["early_line"].notna() & model_df["spread_line"].notna()
                 & model_df["model_margin"].notna()]
    if len(d) < 100:
        return {}
    move = (d["spread_line"] - d["early_line"]).to_numpy(dtype=float)
    edge = (d["model_margin"] - d["early_line"]).to_numpy(dtype=float)
    beta, se, _ = ols(np.column_stack([np.ones(len(d)), edge]), move)
    t = beta[1] / se[1]
    return {"n": len(d), "b": beta[1], "se": se[1], "t": t,
            "p": 2 * (1 - stats.t.cdf(abs(t), df=len(d) - 2)),
            "seasons": f"{d['season'].min()}-{d['season'].max()}"}


def early_walk_forward(model_df: pd.DataFrame, fits_by_season: dict, rules: BetRules,
                       min_train_seasons: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward bets at early lines (assumed -110 both sides).

    fits_by_season supplies sigma and key-number weights (from the closing-line
    model, which only uses earlier seasons). Returns (priced games, coefficients).
    """
    d = model_df[model_df["early_line"].notna() & model_df["model_margin"].notna()].copy()
    seasons = sorted(d["season"].unique())
    rows, coefs = [], []
    for i, S in enumerate(seasons):
        if i < min_train_seasons or S not in fits_by_season:
            continue
        f = fits_by_season[S]
        train = d[(d["season"] < S) & d["played"]]
        cur = d[d["season"] == S].copy()
        even = lambda x: np.full(len(x), 0.5)   # -110/-110 -> no-vig 50%
        mu0_tr = implied_mean(train["early_line"].to_numpy(float), even(train), f.sigma, f.key_w)
        y = train["result"].to_numpy(float) - mu0_tr
        x = (train["model_margin"] - train["early_line"]).to_numpy(float)
        beta, se, _ = ols(x[:, None], y)          # no intercept: trust coefficient only
        b = max(0.0, float(beta[0]))              # never bet *against* the model's own view
        coefs.append({"season": S, "n_train": len(train), "b_trust": float(beta[0]),
                      "t": float(beta[0] / se[0])})
        mu0 = implied_mean(cur["early_line"].to_numpy(float), even(cur), f.sigma, f.key_w)
        mu = mu0 + b * (cur["model_margin"] - cur["early_line"]).to_numpy(float)
        probs = np.array([cover_probs(m, f.sigma, L, f.key_w)
                          for m, L in zip(mu, cur["early_line"].to_numpy(float))])
        cur["mu"], cur["market_mu"] = mu, mu0
        cur[["p_home", "p_push", "p_away"]] = probs
        rows.append(cur)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(coefs)
    allp = pd.concat(rows)
    # Price and grade AGAINST THE EARLY LINE at standard -110.
    tmp = allp.copy()
    tmp["close_line"] = tmp["spread_line"]
    tmp["spread_line"] = tmp["early_line"]
    tmp["home_odds"] = tmp["away_odds"] = -110.0
    priced = grade(price_games(tmp[tmp["played"]], rules))
    side = np.where(priced["side"] == "HOME", 1, -1)
    # CLV in points: how far the close moved in our favour.
    priced["clv_pts"] = (priced["close_line"] - priced["early_line"]) * side
    return priced, pd.DataFrame(coefs)


def trust_coefficient(model_df: pd.DataFrame, fit) -> float:
    """Trust coefficient for live early-line bets, fit on ALL completed seasons
    with early lines (all strictly in the past relative to upcoming games)."""
    d = model_df[model_df["early_line"].notna() & model_df["played"] & model_df["model_margin"].notna()]
    if len(d) < 300:
        return 0.0
    mu0 = implied_mean(d["early_line"].to_numpy(float), np.full(len(d), 0.5), fit.sigma, fit.key_w)
    y = d["result"].to_numpy(float) - mu0
    x = (d["model_margin"] - d["early_line"]).to_numpy(float)
    beta, _, _ = ols(x[:, None], y)
    return max(0.0, float(beta[0]))


def price_with_trust(games: pd.DataFrame, line_col: str, b: float, fit, rules: BetRules) -> pd.DataFrame:
    """Price upcoming games at a user-chosen line with a given trust coefficient."""
    g = games.copy()
    g["spread_line"] = g[line_col].astype(float)
    p_fair = no_vig_home(g["home_odds"].to_numpy(float), g["away_odds"].to_numpy(float), rules.default_odds)
    mu0 = implied_mean(g["spread_line"].to_numpy(float), p_fair, fit.sigma, fit.key_w)
    mu = mu0 + b * (g["model_margin"] - g["spread_line"]).to_numpy(float)
    g["market_mu"], g["mu"] = mu0, mu
    probs = np.array([cover_probs(m, fit.sigma, L, fit.key_w) for m, L in zip(mu, g["spread_line"])])
    g[["p_home", "p_push", "p_away"]] = probs
    return price_games(g, rules)
