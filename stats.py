"""Statistical evaluation: is the edge real or noise?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .staking import breakeven


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bootstrap_roi(units: np.ndarray, n_boot: int = 10000, seed: int = 7) -> tuple[float, float]:
    if len(units) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    rois = units[idx].mean(axis=1)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def bets_needed(edge_winrate: float, be: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Bets needed to detect a true win rate `edge_winrate` vs break-even `be`
    (one-sided test) with the given power."""
    if edge_winrate <= be:
        return -1
    za, zb = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    n = ((za * np.sqrt(be * (1 - be)) + zb * np.sqrt(edge_winrate * (1 - edge_winrate)))
         / (edge_winrate - be)) ** 2
    return int(np.ceil(n))


def max_drawdown(path: np.ndarray) -> float:
    if len(path) == 0:
        return 0.0
    peak = np.maximum.accumulate(path)
    return float(((path - peak) / peak).min())


def summarize(bets: pd.DataFrame, start_bankroll: float = 1000.0) -> dict:
    g = bets[bets["outcome"].notna()]
    w, l, p = (g["outcome"] == "W").sum(), (g["outcome"] == "L").sum(), (g["outcome"] == "P").sum()
    n = w + l
    be = float(np.mean([breakeven(o) for o in g["odds"]])) if len(g) else breakeven(-110)
    win_rate = w / n if n else np.nan
    lo, hi = wilson_ci(int(w), int(n))
    pval = stats.binomtest(int(w), int(n), be, alternative="greater").pvalue if n else np.nan
    units = g["units"].to_numpy()
    roi = units.mean() if len(units) else np.nan
    roi_lo, roi_hi = bootstrap_roi(units)
    out = {
        "bets": int(len(g)), "wins": int(w), "losses": int(l), "pushes": int(p),
        "win_rate": win_rate, "win_ci_lo": lo, "win_ci_hi": hi, "breakeven": be,
        "p_value": pval, "units": float(units.sum()), "roi": roi,
        "roi_ci_lo": roi_lo, "roi_ci_hi": roi_hi,
        "avg_ev_claimed": float(g["ev"].mean()) if len(g) else np.nan,
    }
    if "bankroll" in g.columns and len(g):
        path = np.concatenate([[start_bankroll], g.sort_values("date")["bankroll"].to_numpy()])
        out["final_bankroll"] = float(path[-1])
        out["max_drawdown"] = max_drawdown(path)
    return out


def probability_scores(priced: pd.DataFrame) -> dict:
    """Brier / log-loss of P(home covers | no push) vs. the market's 50/50."""
    d = priced[priced["outcome"].notna() & (priced["result"] != priced["spread_line"])]
    if not len(d):
        return {}
    p = (d["p_home"] / (d["p_home"] + d["p_away"])).clip(1e-6, 1 - 1e-6).to_numpy()
    y = (d["result"] > d["spread_line"]).astype(float).to_numpy()
    brier = np.mean((p - y) ** 2)
    brier_mkt = np.mean((0.5 - y) ** 2)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"n": len(d), "brier": brier, "brier_market": brier_mkt,
            "brier_skill": 1 - brier / brier_mkt, "log_loss": ll, "log_loss_market": np.log(2)}


def calibration_table(bets: pd.DataFrame, bins=(0.5, 0.53, 0.55, 0.57, 0.6, 0.65, 1.0)) -> pd.DataFrame:
    """Predicted vs. actual win rate (pushes excluded) for the side we bet."""
    g = bets[bets["outcome"].isin(["W", "L"])].copy()
    if not len(g):
        return pd.DataFrame()
    g["p_win_np"] = g["p_win"] / (g["p_win"] + g["p_loss"])
    g["bucket"] = pd.cut(g["p_win_np"], bins=list(bins), include_lowest=True)
    t = g.groupby("bucket", observed=True).agg(
        bets=("outcome", "size"),
        predicted=("p_win_np", "mean"),
        actual=("outcome", lambda s: (s == "W").mean()),
    ).reset_index()
    t["bucket"] = t["bucket"].astype(str)
    return t


def by_season(bets: pd.DataFrame) -> pd.DataFrame:
    g = bets[bets["outcome"].notna()]
    if not len(g):
        return pd.DataFrame()
    t = g.groupby("season").agg(
        bets=("outcome", "size"),
        wins=("outcome", lambda s: (s == "W").sum()),
        losses=("outcome", lambda s: (s == "L").sum()),
        pushes=("outcome", lambda s: (s == "P").sum()),
        units=("units", "sum"),
    ).reset_index()
    t["win_rate"] = t["wins"] / (t["wins"] + t["losses"])
    t["roi"] = t["units"] / t["bets"]
    return t
