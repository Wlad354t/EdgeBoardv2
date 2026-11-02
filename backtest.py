"""Bet selection and bankroll simulation on walk-forward probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .staking import american_to_decimal, expected_value, kelly_fraction


@dataclass
class BetRules:
    min_edge: float = 0.03        # minimum EV per $1
    max_edge: float = 0.25        # ignore "edges" this big: almost always stale/bad data
    kelly_mult: float = 0.25      # quarter Kelly
    max_stake_pct: float = 0.03   # never risk more than 3% on one bet
    default_odds: int = -110
    start_bankroll: float = 1000.0


def price_games(df: pd.DataFrame, rules: BetRules) -> pd.DataFrame:
    """For each game with probabilities, pick the better side and its EV."""
    d = df[df["p_home"].notna()].copy()
    ho = d["home_odds"].fillna(rules.default_odds).to_numpy(dtype=float)
    ao = d["away_odds"].fillna(rules.default_odds).to_numpy(dtype=float)
    ph, pp, pa = d["p_home"].to_numpy(), d["p_push"].to_numpy(), d["p_away"].to_numpy()
    ev_h = np.array([expected_value(w, l, o) for w, l, o in zip(ph, pa, ho)])
    ev_a = np.array([expected_value(w, l, o) for w, l, o in zip(pa, ph, ao)])
    home_side = ev_h >= ev_a
    d["side"] = np.where(home_side, "HOME", "AWAY")
    d["pick"] = np.where(home_side, d["home_team"], d["away_team"])
    d["odds"] = np.where(home_side, ho, ao)
    d["p_win"] = np.where(home_side, ph, pa)
    d["p_loss"] = np.where(home_side, pa, ph)
    d["ev"] = np.where(home_side, ev_h, ev_a)
    d["kelly_full"] = [kelly_fraction(w, l, o) for w, l, o in zip(d["p_win"], d["p_loss"], d["odds"])]
    # Pick's line in sportsbook notation (negative = favourite)
    d["pick_line"] = np.where(home_side, -d["spread_line"], d["spread_line"])
    d["bet"] = (d["ev"] >= rules.min_edge) & (d["ev"] <= rules.max_edge)
    return d


def grade(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    margin_vs_line = d["result"] - d["spread_line"]           # >0 home covered
    side_sign = np.where(d["side"] == "HOME", 1, -1)
    m = margin_vs_line * side_sign
    d["outcome"] = np.select([m > 0, m < 0, m == 0], ["W", "L", "P"], default=None)
    d.loc[d["result"].isna(), "outcome"] = None
    dec = d["odds"].apply(american_to_decimal)
    d["units"] = np.select([d["outcome"] == "W", d["outcome"] == "L"], [dec - 1, -1.0], default=0.0)
    return d


def simulate_bankroll(bets: pd.DataFrame, rules: BetRules) -> pd.DataFrame:
    """Fractional-Kelly bankroll path. Bets on the same date are sized off the
    bankroll at the start of that date (they're simultaneous in real life)."""
    bets = bets.sort_values("date").copy()
    bank = rules.start_bankroll
    stakes, banks = [], []
    for _, day in bets.groupby("date", sort=True):
        day_start = bank
        for i, r in day.iterrows():
            frac = min(rules.kelly_mult * r["kelly_full"], rules.max_stake_pct)
            stake = day_start * frac
            bank += stake * r["units"]
            stakes.append((i, stake))
        for i, _ in day.iterrows():
            banks.append((i, bank))
    bets["stake"] = pd.Series(dict(stakes))
    bets["bankroll"] = pd.Series(dict(banks))
    return bets


def run_backtest(model_df: pd.DataFrame, rules: BetRules, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all priced games in those seasons, placed bets with bankroll)."""
    priced = grade(price_games(model_df[model_df["season"].isin(seasons) & model_df["played"]], rules))
    bets = priced[priced["bet"]].copy()
    if len(bets):
        bets = simulate_bankroll(bets, rules)
    return priced, bets
