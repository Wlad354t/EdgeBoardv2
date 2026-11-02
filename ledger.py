"""Paper-trade ledger: log picks before games, grade them and measure CLV after.

CLV (closing-line value) is the fastest honest scoreboard: over a few hundred
bets it says far more about a real edge than win/loss record does.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .staking import american_to_decimal

LEDGER = DATA_DIR / "ledger.csv"
COLS = ["logged_at", "sport", "mode", "game_id", "date", "away_team", "home_team", "pick", "side",
        "line_taken", "pick_line", "odds", "p_win", "ev", "stake"]


def load(path: Path = LEDGER) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path)


def append(picks: pd.DataFrame, sport: str, mode: str, path: Path = LEDGER) -> int:
    led = load(path)
    new = picks.assign(logged_at=datetime.now().isoformat(timespec="seconds"), sport=sport, mode=mode,
                       line_taken=picks["spread_line"])
    new = new[[c for c in COLS if c in new.columns]]
    # One entry per game/side: re-logging replaces the old entry.
    if len(led):
        key = led["game_id"].astype(str) + led["side"]
        led = led[~key.isin(new["game_id"].astype(str) + new["side"])]
    out = pd.concat([led, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(new)


def grade(ledger: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if not len(ledger):
        return ledger
    g = games[["game_id", "result", "spread_line"]].rename(columns={"spread_line": "close_line"})
    d = ledger.merge(g, on="game_id", how="left")
    s = np.where(d["side"] == "HOME", 1, -1)
    m = (d["result"] - d["line_taken"]) * s
    d["outcome"] = np.select([m > 0, m < 0, m == 0], ["W", "L", "P"], default="")
    d.loc[d["result"].isna(), "outcome"] = "pending"
    dec = d["odds"].apply(american_to_decimal)
    d["units"] = np.select([d["outcome"] == "W", d["outcome"] == "L"], [dec - 1, -1.0], default=0.0)
    d["profit"] = d["units"] * d["stake"].fillna(0)
    d["clv_pts"] = (d["close_line"] - d["line_taken"]) * s
    return d
