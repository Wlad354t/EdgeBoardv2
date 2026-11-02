"""Data loading and validation.

Canonical schema (one row per game, any sport):

    date          game date (YYYY-MM-DD)
    season        season label (int); derived from date if missing
    home_team     str
    away_team     str
    home_score    float, blank for games not yet played
    away_score    float, blank for games not yet played
    spread_line   HOME-perspective line, POSITIVE when home is favored
                  (nflverse convention: home -3.5 at the book -> 3.5 here)
    home_odds     American odds on home spread (optional, default -110)
    away_odds     American odds on away spread (optional, default -110)
    neutral       1 if neutral site (optional)
    rest_diff     home rest days minus away rest days (optional)

Sign convention used everywhere: result = home_score - away_score.
Home covers when result > spread_line; push when equal.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NFLVERSE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
SC_URL = "https://github.com/nflverse/nfldata/raw/master/data/sc_lines.csv"
DATA_DIR = Path(__file__).resolve().parent.parent
NFL_CACHE = DATA_DIR / "nfl_games.csv"
SC_CACHE = DATA_DIR / "nfl_sc_lines.csv"   # Westgate SuperContest lines (set mid-week), 2013-2020

REQUIRED = ["date", "home_team", "away_team", "home_score", "away_score", "spread_line"]


def refresh_nfl(timeout: int = 30) -> pd.DataFrame:
    """Download the latest nflverse games file (scores + lines) and cache it."""
    with urllib.request.urlopen(NFLVERSE_URL, timeout=timeout) as r:
        raw = r.read()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NFL_CACHE.write_bytes(raw)
    try:
        with urllib.request.urlopen(SC_URL, timeout=timeout) as r:
            SC_CACHE.write_bytes(r.read())
    except Exception:
        pass
    return pd.read_csv(io.BytesIO(raw))


def load_nfl(path: Path | str = NFL_CACHE) -> pd.DataFrame:
    g = pd.read_csv(path)
    df = pd.DataFrame({
        "game_id": g["game_id"],
        "date": pd.to_datetime(g["gameday"]),
        "season": g["season"].astype(int),
        "week": g["week"],
        "game_type": g["game_type"],
        "home_team": g["home_team"],
        "away_team": g["away_team"],
        "home_score": g["home_score"],
        "away_score": g["away_score"],
        "spread_line": g["spread_line"],
        "home_odds": g["home_spread_odds"],
        "away_odds": g["away_spread_odds"],
        "neutral": (g["location"] == "Neutral").astype(int),
        "rest_diff": (g["home_rest"] - g["away_rest"]).clip(-7, 7),
        "div_game": g["div_game"].fillna(0).astype(int),
        "home_qb": g.get("home_qb_name"),
        "away_qb": g.get("away_qb_name"),
    })
    # Early (mid-week) lines, home perspective, where available.
    df["early_line"] = np.nan
    if SC_CACHE.exists():
        sc = pd.read_csv(SC_CACHE)
        sc = sc[sc["side"] == sc["home_team"]][["game_id", "line"]]
        sc_map = dict(zip(sc["game_id"].astype(str), -sc["line"]))
        df["early_line"] = g["old_game_id"].astype(str).map(sc_map).to_numpy()
    return finalize(df)


def load_generic(file, convention: str = "home_positive") -> pd.DataFrame:
    """Load a user CSV in the canonical schema.

    convention="sportsbook" means spread_line is the home team's *posted* spread
    (negative when favored, e.g. -3.5); it is flipped to the internal convention.
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    if "season" not in df.columns:
        # Season = year the season started; works for fall-spring leagues if
        # the season starts after July.
        df["season"] = np.where(df["date"].dt.month >= 8, df["date"].dt.year, df["date"].dt.year - 1)
    if convention == "sportsbook":
        df["spread_line"] = -df["spread_line"]
    if convention == "sportsbook" and "early_line" in df.columns:
        df["early_line"] = -df["early_line"]
    for c, default in [("home_odds", np.nan), ("away_odds", np.nan), ("neutral", 0),
                       ("rest_diff", 0.0), ("div_game", 0), ("early_line", np.nan)]:
        if c not in df.columns:
            df[c] = default
    if "game_id" not in df.columns:
        df["game_id"] = (df["date"].dt.strftime("%Y%m%d") + "_" + df["away_team"].astype(str)
                         + "_" + df["home_team"].astype(str))
    return finalize(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    df["result"] = df["home_score"] - df["away_score"]
    df["played"] = df["result"].notna()
    df["has_line"] = df["spread_line"].notna()
    df["rest_diff"] = df["rest_diff"].fillna(0.0)
    df["neutral"] = df["neutral"].fillna(0).astype(int)
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    return df
