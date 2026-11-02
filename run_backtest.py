"""Command-line backtest + this week's picks.

    python run_backtest.py                       # NFL, cached data
    python run_backtest.py --refresh             # pull latest nflverse data first
    python run_backtest.py --csv nba.csv --sport NBA --train 2015-2020 --test 2021-2024
"""
import argparse

import pandas as pd

from edgeboard import (SPORTS, BetRules, by_season, calibration_table, fit_walk_forward,
                       load_generic, load_nfl, market_value_test, price_games,
                       probability_scores, refresh_nfl, run_backtest, summarize,
                       tune_ratings, tune_threshold)


def rng(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="NFL", choices=list(SPORTS))
    ap.add_argument("--csv", help="generic CSV (see edgeboard/data.py for schema)")
    ap.add_argument("--sportsbook-lines", action="store_true",
                    help="CSV spread_line is the posted home spread (negative = home favoured)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--train", default="2003-2012")
    ap.add_argument("--test", default="2013-2025")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    a = ap.parse_args()

    cfg = SPORTS[a.sport]
    if a.csv:
        df = load_generic(a.csv, "sportsbook" if a.sportsbook_lines else "home_positive")
    else:
        if a.refresh:
            refresh_nfl()
        df = load_nfl()
    train, test = rng(a.train), rng(a.test)

    params, _ = tune_ratings(df, cfg, train)
    print("Tuned rating params (train only):", params)
    rules = BetRules(start_bankroll=a.bankroll, default_odds=cfg.default_odds)
    thr, thr_table = tune_threshold(df, cfg, params, train, rules)
    rules.min_edge = thr
    print("\nThreshold search on TRAIN seasons:\n", thr_table.to_string(index=False))
    print(f"Chosen min EV: {thr:.2%}")

    model_df, fits = fit_walk_forward(df, cfg, params)

    mv = market_value_test(model_df, test)
    print(f"\nDoes the model know something the closing line doesn't? (holdout {a.test})")
    print(f"  b1 = {mv['b1']:.3f}  (SE {mv['se1']:.3f}, t = {mv['t1']:.2f}, p = {mv['p1']:.3f}, n = {mv['n']})")

    priced, bets = run_backtest(model_df, rules, test)
    s = summarize(bets, a.bankroll)
    print(f"\nHOLDOUT {a.test}: {s['wins']}-{s['losses']}-{s['pushes']}  win {s['win_rate']:.1%} "
          f"[95% CI {s['win_ci_lo']:.1%}-{s['win_ci_hi']:.1%}]  break-even {s['breakeven']:.1%}  "
          f"p = {s['p_value']:.3f}")
    print(f"  ROI {s['roi']:+.2%} [95% CI {s['roi_ci_lo']:+.2%} to {s['roi_ci_hi']:+.2%}]  "
          f"units {s['units']:+.1f}  claimed avg EV {s['avg_ev_claimed']:+.2%}")
    if "final_bankroll" in s:
        print(f"  Bankroll ${a.bankroll:,.0f} -> ${s['final_bankroll']:,.0f}  max DD {s['max_drawdown']:.1%}")
    print("\nProbability quality:", {k: round(v, 4) for k, v in probability_scores(priced).items()})
    print("\nBy season:\n", by_season(bets).to_string(index=False))
    print("\nCalibration:\n", calibration_table(bets).to_string(index=False))

    from edgeboard import early
    lm = early.line_movement_test(model_df)
    if lm:
        print(f"\nEarly lines {lm['seasons']}: closing line moves {lm['b']:+.3f} pts toward the model per "
              f"point of disagreement (t = {lm['t']:.2f}, p = {lm['p']:.4f}, n = {lm['n']})")
        ep, _ = early.early_walk_forward(model_df, {f.season: f for f in fits}, rules)
        eb = ep[ep["bet"]] if len(ep) else ep
        if len(eb):
            es = summarize(eb)
            print(f"  Walk-forward bets at early lines: {es['wins']}-{es['losses']}-{es['pushes']} "
                  f"ROI {es['roi']:+.2%}  avg CLV {eb['clv_pts'].mean():+.2f} pts")

    upcoming = model_df[~model_df["played"] & model_df["p_home"].notna()]
    if len(upcoming):
        up = price_games(upcoming, rules)
        cols = ["date", "away_team", "home_team", "spread_line", "model_margin", "mu",
                "pick", "pick_line", "odds", "p_win", "ev", "bet"]
        print("\nUpcoming:\n", up[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
