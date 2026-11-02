"""Walk-forward spread model.

Stage 1  Margin-Elo ratings -> model_margin (pre-game, no look-ahead).
Stage 2  Market baseline: mu0 = the mean margin implied by the line and the
         no-vig odds (see probability.implied_mean). The market alone
         produces exactly zero edge.
Stage 3  For each season S, fit on seasons < S only:

            result - mu0 = b0 + b1*(model_margin - line) + b2*rest_diff + e

         b1 is the key number: "when my ratings disagree with the line by
         1 point, how many points does reality move my way?" b1 = 0 means
         the model adds nothing beyond the market, and the program will then
         (correctly) find almost no bets.
Stage 4  mu = mu0 + fitted adjustment -> P(home cover / push / away cover).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import SportConfig
from .probability import cover_probs, implied_mean, learn_key_weights
from .ratings import run_margin_elo
from .staking import no_vig_home

FEATURES = ["intercept", "model_edge_pts", "rest_diff"]


@dataclass
class SeasonFit:
    season: int
    n_train: int
    beta: np.ndarray
    se: np.ndarray
    sigma: float
    key_w: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return self.beta / self.se


def _design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), d["model_edge_pts"].to_numpy(dtype=float),
                            d["rest_diff"].to_numpy(dtype=float)])


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), float(np.sqrt(s2))


def _market_mu(d: pd.DataFrame, sigma: float, key_w: np.ndarray, default_odds: int) -> np.ndarray:
    p_fair = no_vig_home(d["home_odds"].to_numpy(dtype=float),
                         d["away_odds"].to_numpy(dtype=float), default_odds)
    return implied_mean(d["spread_line"].to_numpy(dtype=float), p_fair, sigma, key_w)


def fit_walk_forward(df: pd.DataFrame, cfg: SportConfig, params: dict,
                     train_start: int | None = None, min_train_games: int = 400,
                     shrink: float = 1.0) -> tuple[pd.DataFrame, list[SeasonFit]]:
    """Add model columns to df and return per-season fits.

    train_start: first season allowed in training (skips rating warm-up).
    shrink: multiply the fitted adjustment by this (0..1) for extra caution.
    """
    out = df.copy()
    preds, _ = run_margin_elo(out, **params)
    out["model_margin"] = preds
    out["model_edge_pts"] = out["model_margin"] - out["spread_line"]
    for c in ["market_mu", "mu", "p_home", "p_push", "p_away"]:
        out[c] = np.nan

    line_seasons = sorted(out.loc[out["has_line"], "season"].unique())
    if train_start is None:
        train_start = line_seasons[1] if len(line_seasons) > 1 else line_seasons[0]

    fits: list[SeasonFit] = []
    usable = out["has_line"] & out["model_edge_pts"].notna()
    for S in line_seasons:
        train = out[usable & out["played"] & (out["season"] < S) & (out["season"] >= train_start)]
        if len(train) < min_train_games:
            continue
        key_w = (learn_key_weights(train["result"].to_numpy(dtype=float))
                 if cfg.use_key_numbers else np.ones(81))
        sigma = float(np.std(train["result"] - train["spread_line"])) or cfg.sigma_prior
        mu0_train = _market_mu(train, sigma, key_w, cfg.default_odds)
        y = train["result"].to_numpy(dtype=float) - mu0_train
        beta, se, _ = ols(_design(train), y)
        fits.append(SeasonFit(int(S), len(train), beta, se, sigma, key_w))

        idx = out.index[usable & (out["season"] == S)]
        cur = out.loc[idx]
        mu0 = _market_mu(cur, sigma, key_w, cfg.default_odds)
        mu = mu0 + shrink * (_design(cur) @ beta)
        lines = cur["spread_line"].to_numpy(dtype=float)
        probs = np.array([cover_probs(m, sigma, L, key_w) for m, L in zip(mu, lines)])
        out.loc[idx, "market_mu"] = mu0
        out.loc[idx, "mu"] = mu
        out.loc[idx, ["p_home", "p_push", "p_away"]] = probs
    return out, fits


def market_value_test(df: pd.DataFrame, seasons: list[int]) -> dict:
    """Pooled out-of-sample test on the given seasons.

    Regresses (result - market-implied mean) on the model's pre-game edge.
    If the ratings carry information the closing line doesn't, b1 > 0 with
    t > ~2 (p < 0.05). This is the single most important number in the app.
    """
    d = df[df["played"] & df["season"].isin(seasons) & df["market_mu"].notna()]
    if len(d) < 50:
        return {}
    y = (d["result"] - d["market_mu"]).to_numpy(dtype=float)
    beta, se, sigma = ols(_design(d), y)
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=len(d) - 3))
    return {"n": len(d), "b0": beta[0], "t0": t[0], "b1": beta[1], "se1": se[1], "t1": t[1],
            "p1": p[1], "b_rest": beta[2], "t_rest": t[2], "p_rest": p[2], "sigma": sigma}


def coefficient_history(fits: list[SeasonFit]) -> pd.DataFrame:
    return pd.DataFrame([{
        "season": f.season, "n_train": f.n_train, "intercept": f.beta[0],
        "b1_model_edge": f.beta[1], "t_b1": f.t[1], "b2_rest": f.beta[2], "t_b2": f.t[2],
        "sigma": f.sigma} for f in fits])
