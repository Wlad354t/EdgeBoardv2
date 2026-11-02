"""EdgeBoard v2: statistically tested spread betting model.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from edgeboard import SPORTS, BetRules, load_generic, load_nfl, refresh_nfl
from edgeboard import early as E
from edgeboard import ledger as L
from edgeboard.backtest import run_backtest
from edgeboard.model import coefficient_history, fit_walk_forward, market_value_test
from edgeboard.stats import (bets_needed, by_season, calibration_table, probability_scores,
                             summarize)
from edgeboard.tune import tune_ratings, tune_threshold

st.set_page_config(page_title="EdgeBoard v2", page_icon="📈", layout="wide")

# ----------------------------------------------------------------- data ----
@st.cache_data(show_spinner=False)
def get_nfl(stamp: int) -> pd.DataFrame:
    return load_nfl()


@st.cache_data(show_spinner=False)
def get_csv(raw: bytes, convention: str) -> pd.DataFrame:
    return load_generic(io.BytesIO(raw), convention)


@st.cache_data(show_spinner="Tuning on training seasons and running walk-forward…")
def pipeline(df: pd.DataFrame, sport: str, train: tuple, test: tuple, rules_d: dict,
             auto_edge: bool, tune: bool):
    cfg = SPORTS[sport]
    if tune:
        params, grid = tune_ratings(df, cfg, list(train))
    else:
        params = {"k": cfg.k, "hfa": cfg.hfa, "regress": cfg.regress, "mov_cap": cfg.mov_cap}
        grid = pd.DataFrame()
    rules = BetRules(**rules_d)
    thr_table = pd.DataFrame()
    if auto_edge:
        thr, thr_table = tune_threshold(df, cfg, params, list(train), rules)
        rules.min_edge = thr
    model_df, fits = fit_walk_forward(df, cfg, params)
    priced, bets = run_backtest(model_df, rules, list(test))
    return dict(params=params, grid=grid, rules=rules, thr_table=thr_table, model_df=model_df,
                fits=fits, priced=priced, bets=bets,
                mv=market_value_test(model_df, list(test)),
                coefs=coefficient_history(fits))


# -------------------------------------------------------------- sidebar ----
st.sidebar.title("EdgeBoard v2")
source = st.sidebar.radio("Data", ["NFL (nflverse, built in)", "Upload CSV (any sport)"])

if source.startswith("NFL"):
    sport = "NFL"
    if "stamp" not in st.session_state:
        st.session_state.stamp = 0
    if st.sidebar.button("Refresh NFL data (scores + lines)"):
        with st.spinner("Downloading latest nflverse data…"):
            try:
                refresh_nfl()
                st.session_state.stamp += 1
                st.sidebar.success("Updated.")
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")
    df = get_nfl(st.session_state.stamp)
else:
    sport = st.sidebar.selectbox("Sport settings", [s for s in SPORTS if s != "NFL"] + ["NFL"])
    conv = st.sidebar.radio("spread_line column is…",
                            ["Home perspective, + = home favored", "Posted home spread (-3.5 = home favored)"])
    up = st.sidebar.file_uploader("Games CSV", type="csv")
    st.sidebar.caption("Columns: date, home_team, away_team, home_score, away_score, spread_line "
                       "(+ optional season, home_odds, away_odds, neutral, rest_diff, early_line). "
                       "Leave scores blank for upcoming games.")
    if up is None:
        st.info("Upload a games CSV to start, or switch to the built-in NFL data.")
        st.stop()
    try:
        df = get_csv(up.getvalue(), "sportsbook" if conv.startswith("Posted") else "home_positive")
    except Exception as e:
        st.error(str(e))
        st.stop()

line_seasons = sorted(df.loc[df["has_line"] & df["played"], "season"].unique().tolist())
if len(line_seasons) < 4:
    st.error("Need at least 4 seasons of games with lines and results for an honest walk-forward test.")
    st.stop()

st.sidebar.subheader("Train / test split")
default_split = 2013 if sport == "NFL" and 2013 in line_seasons else line_seasons[len(line_seasons) * 2 // 3]
split = st.sidebar.select_slider("First holdout season", options=line_seasons[2:], value=default_split)
train = tuple(s for s in line_seasons if s < split and s >= line_seasons[min(3, len(line_seasons) - 3)])
test = tuple(s for s in line_seasons if s >= split)
st.sidebar.caption(f"Tune on {train[0]}–{train[-1]} · Test on {test[0]}–{test[-1]} (never seen while tuning)")

st.sidebar.subheader("Staking")
bankroll = st.sidebar.number_input("Bankroll ($)", 100.0, 1e7, 1000.0, step=100.0)
kelly = st.sidebar.slider("Kelly multiplier", 0.05, 1.0, 0.25, 0.05)
max_pct = st.sidebar.slider("Max stake (% bankroll)", 0.5, 10.0, 3.0, 0.5) / 100
auto_edge = st.sidebar.toggle("Choose min EV on training data", True)
manual_edge = st.sidebar.slider("Min EV per $1 (manual)", 0.0, 0.10, 0.03, 0.005, disabled=auto_edge)
tune = st.sidebar.toggle("Tune rating parameters", True)

rules_d = dict(min_edge=manual_edge, kelly_mult=kelly, max_stake_pct=max_pct,
               default_odds=SPORTS[sport].default_odds, start_bankroll=bankroll)
R = pipeline(df, sport, train, test, rules_d, auto_edge, tune)
rules: BetRules = R["rules"]
mv, bets, priced = R["mv"], R["bets"], R["priced"]
S = summarize(bets, bankroll) if len(bets) else None

tabs = st.tabs(["Verdict", "Backtest", "Early lines & CLV", "This week", "Ledger", "How it works"])

# --------------------------------------------------------------- verdict ----
with tabs[0]:
    st.header(f"{sport}: does the model beat the market?")
    st.caption(f"Out-of-sample seasons {test[0]}–{test[-1]} · parameters fit on {train[0]}–{train[-1]} only")
    c = st.columns(4)
    c[0].metric("Info beyond the line (b1)", f"{mv.get('b1', np.nan):+.3f}",
                f"t = {mv.get('t1', np.nan):.2f}, p = {mv.get('p1', np.nan):.3f}", delta_color="off")
    if S:
        c[1].metric("ATS record", f"{S['wins']}-{S['losses']}-{S['pushes']}",
                    f"{S['win_rate']:.1%} (need {S['breakeven']:.1%})", delta_color="off")
        c[2].metric("ROI (flat stakes)", f"{S['roi']:+.2%}",
                    f"95% CI {S['roi_ci_lo']:+.1%} to {S['roi_ci_hi']:+.1%}", delta_color="off")
        c[3].metric("p-value vs. break-even", f"{S['p_value']:.3f}", "< 0.05 = significant", delta_color="off")
    ps = probability_scores(priced)

    t1, p1 = mv.get("t1", 0), mv.get("p1", 1)
    sig_bets = S and S["p_value"] < 0.05 and S["roi_ci_lo"] > 0
    if t1 > 2 and sig_bets:
        st.success("**Edge supported.** The model carries information the closing line doesn't, and the "
                   "out-of-sample betting record beats the vig with statistical significance. "
                   "Still size small and keep tracking CLV.")
    elif t1 > 2:
        st.warning("**Information, but not enough to beat the vig.** The model predicts results beyond "
                   "the line, but the betting record isn't significant yet.")
    else:
        st.error(f"**No proven edge against closing lines.** Holding the market's price fixed, the model's "
                 f"opinion explains ~0 of the remaining outcome (b1 = {mv.get('b1', 0):+.3f}, p = {p1:.2f}). "
                 "Betting this at closing lines is paying the vig. See *Early lines & CLV* for where "
                 "the model does show a measurable signal.")

    st.subheader("What the numbers mean")
    be = S["breakeven"] if S else 0.524
    need54, need55 = bets_needed(0.54, be), bets_needed(0.55, be)
    st.markdown(
        f"- **b1** is the share of the model-vs-line disagreement that shows up in real results. "
        f"0 = the market already knew it; 1 = the model is right and the line is wrong.\n"
        f"- **Probability quality** vs. the market's 50/50: Brier skill "
        f"{ps.get('brier_skill', np.nan):+.4f} (positive = better than the market) over {ps.get('n', 0):,} games.\n"
        f"- **Sample size reality:** proving a true 54% bettor needs ~**{need54:,} bets**; a 55% bettor ~"
        f"**{need55:,}** (one-sided, 5% significance, 80% power). A hot 100-bet stretch proves nothing.\n"
        f"- The model *claimed* average EV of {S['avg_ev_claimed']:+.2%} on its bets; the realised ROI was "
        f"{S['roi']:+.2%}. A large gap between those means the model is overconfident." if S else "")

# -------------------------------------------------------------- backtest ----
with tabs[1]:
    st.header("Walk-forward backtest")
    st.caption(f"Bets at closing lines/odds · min EV {rules.min_edge:.1%} · {kelly:.2f}× Kelly, cap {max_pct:.1%}")
    if not len(bets):
        st.info("No bets met the threshold in the holdout.")
    else:
        b = bets.sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b["date"], y=b["bankroll"], mode="lines", name="Bankroll"))
        fig.add_hline(y=bankroll, line_dash="dot", annotation_text="start")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Kelly bankroll")
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("By season")
            st.dataframe(by_season(bets).style.format({"units": "{:+.1f}", "win_rate": "{:.1%}", "roi": "{:+.1%}"}),
                         hide_index=True, width="stretch")
        with c2:
            st.subheader("Calibration (side bet)")
            cal = calibration_table(bets)
            if len(cal):
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="markers+text",
                                        text=cal["bets"], textposition="top center", name="buckets"))
                f2.add_trace(go.Scatter(x=[0.5, 0.62], y=[0.5, 0.62], mode="lines",
                                        line=dict(dash="dot"), name="perfect"))
                f2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                 xaxis_title="Predicted win %", yaxis_title="Actual win %")
                st.plotly_chart(f2, width="stretch")
        st.subheader("Trust coefficient by season (walk-forward fits)")
        st.dataframe(R["coefs"].round(3), hide_index=True, width="stretch")
        if len(R["thr_table"]):
            st.subheader("Min-EV search (training seasons only)")
            st.dataframe(R["thr_table"], hide_index=True)
        st.caption(f"Rating parameters used: {R['params']}")
        cols = ["date", "season", "away_team", "home_team", "spread_line", "model_margin", "mu", "pick",
                "pick_line", "odds", "p_win", "ev", "stake", "outcome", "units", "bankroll"]
        st.download_button("Download bet log (CSV)", bets[cols].to_csv(index=False), "bet_log.csv")

# ------------------------------------------------------- early lines ----
fits_by = {f.season: f for f in R["fits"]}
with tabs[2]:
    st.header("Early lines & closing-line value")
    lm = E.line_movement_test(R["model_df"])
    if not lm:
        st.info("No early-line data in this dataset. Add an `early_line` column (the line when you'd "
                "actually bet) to run these tests.")
    else:
        c = st.columns(3)
        c[0].metric("Line moves toward model (b)", f"{lm['b']:+.3f}",
                    f"t = {lm['t']:.2f}, p = {lm['p']:.4f}", delta_color="off")
        c[1].metric("Games", f"{lm['n']:,}", lm["seasons"], delta_color="off")
        st.markdown(
            f"For every point the model disagrees with the mid-week line, the closing line moves "
            f"**{lm['b']:.2f} pts** toward the model on average. "
            + ("That's statistically significant: the model spots things before the market prices them. "
               "This is closing-line value, and it is the best leading indicator of a real edge."
               if lm["p"] < 0.05 else "That is not statistically significant."))
        ep, co = E.early_walk_forward(R["model_df"], fits_by, rules)
        if len(ep):
            eb = ep[ep["bet"]]
            es = summarize(eb) if len(eb) else None
            st.subheader("Walk-forward bets at early lines (-110)")
            if es:
                c = st.columns(4)
                c[0].metric("Record", f"{es['wins']}-{es['losses']}-{es['pushes']}", f"{es['win_rate']:.1%}",
                            delta_color="off")
                c[1].metric("ROI", f"{es['roi']:+.2%}", f"CI {es['roi_ci_lo']:+.1%} to {es['roi_ci_hi']:+.1%}",
                            delta_color="off")
                c[2].metric("Avg CLV", f"{eb['clv_pts'].mean():+.2f} pts",
                            f"beat close {(eb['clv_pts'] > 0).mean():.0%} / lost {(eb['clv_pts'] < 0).mean():.0%}",
                            delta_color="off")
                c[3].metric("p-value", f"{es['p_value']:.3f}")
                st.caption("Positive CLV with a losing record over a small sample is normal variance; "
                           "CLV is the more stable signal, but ~0.2 pts is worth well under the ~2.4% "
                           "win-rate the -110 vig demands.")
            st.dataframe(co.round(3), hide_index=True)

# ------------------------------------------------------------ this week ----
with tabs[3]:
    st.header("This week")
    upcoming = R["model_df"][~R["model_df"]["played"] & R["model_df"]["model_margin"].notna()]
    upcoming = upcoming[upcoming["date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=1)]
    if not len(upcoming) or not R["fits"]:
        st.info("No upcoming games in the data. Refresh the data or add rows with blank scores.")
    else:
        latest = R["fits"][-1]
        b_close = max(0.0, float(latest.beta[1]))
        b_early = E.trust_coefficient(R["model_df"], latest) if lm else 0.0
        mode = st.radio("I'm betting at…", [
            f"Current/closing-type lines (trust b = {b_close:.2f})",
            f"Early lines, before they move (trust b = {b_early:.2f})"], horizontal=True)
        b = b_close if mode.startswith("Current") else b_early
        st.caption("Edit the line and odds to match your book before reading the picks. "
                   "Line = home perspective, + means home is favored.")
        ed = upcoming[["game_id", "date", "away_team", "home_team", "spread_line", "home_odds",
                       "away_odds", "model_margin"]].copy()
        ed["home_odds"] = ed["home_odds"].fillna(-110)
        ed["away_odds"] = ed["away_odds"].fillna(-110)
        ed = st.data_editor(ed, hide_index=True, width="stretch",
                            disabled=["game_id", "date", "away_team", "home_team", "model_margin"],
                            column_config={"model_margin": st.column_config.NumberColumn("model home margin",
                                                                                         format="%.1f")})
        ed = ed[ed["spread_line"].notna()]
        if len(ed):
            full = ed.merge(upcoming[["game_id", "season", "neutral", "rest_diff"]], on="game_id")
            pr = E.price_with_trust(full, "spread_line", b, latest, rules)
            bank_now = st.number_input("Current bankroll ($)", 1.0, 1e7, float(bankroll), key="bnow")
            pr["stake"] = (bank_now * np.minimum(rules.kelly_mult * pr["kelly_full"], rules.max_stake_pct)
                           ).where(pr["bet"], 0.0).round(2)
            show = pr[["date", "away_team", "home_team", "spread_line", "model_margin", "mu", "pick",
                       "pick_line", "odds", "p_win", "ev", "stake", "bet"]].sort_values("ev", ascending=False)
            st.dataframe(show.style.format({"model_margin": "{:.1f}", "mu": "{:.2f}", "p_win": "{:.1%}",
                                            "ev": "{:+.1%}", "pick_line": "{:+.1f}", "odds": "{:+.0f}",
                                            "stake": "${:,.2f}"}),
                         hide_index=True, width="stretch")
            n_bets = int(pr["bet"].sum())
            st.write(f"**{n_bets}** game(s) clear the {rules.min_edge:.1%} EV bar.")
            if b == 0:
                st.warning("Trust coefficient is 0: the backtest found no information beyond this kind of "
                           "line, so every game prices at the market and nothing qualifies. That's the "
                           "model working as intended.")
            if n_bets and st.button("Log these picks to the paper-trade ledger"):
                k = L.append(pr[pr["bet"]], sport, "close" if mode.startswith("Current") else "early")
                st.success(f"Logged {k} pick(s). Grade them in the Ledger tab after the games.")

# ---------------------------------------------------------------- ledger ----
with tabs[4]:
    st.header("Paper-trade ledger")
    led = L.grade(L.load(), R["model_df"])
    if not len(led):
        st.info("No picks logged yet. Log picks from *This week*; refresh data after games to grade them.")
    else:
        done = led[led["outcome"].isin(["W", "L", "P"])]
        c = st.columns(4)
        c[0].metric("Logged", len(led))
        c[1].metric("Record", f"{(done.outcome == 'W').sum()}-{(done.outcome == 'L').sum()}-{(done.outcome == 'P').sum()}")
        c[2].metric("Units", f"{done['units'].sum():+.2f}")
        c[3].metric("Avg CLV", f"{led['clv_pts'].mean():+.2f} pts" if led["clv_pts"].notna().any() else "—")
        st.dataframe(led, hide_index=True, width="stretch")
        st.download_button("Download ledger", led.to_csv(index=False), "ledger.csv")

# ---------------------------------------------------------- how it works ----
with tabs[5]:
    st.header("How it works")
    st.markdown("""
1. **Ratings (Margin-Elo).** Every team has a rating in points vs. league average. Predicted margin =
   home rating − away rating + home advantage. After each game, ratings move by *k* × (actual − predicted),
   capped to dampen blowouts, and shrink toward average each off-season. Predictions are always made
   *before* the game's result is seen.
2. **Market baseline.** The line plus no-vig odds define what the market expects. Because lines are
   medians, not means, the app solves for the mean margin that reproduces the market's 50/50 (with NFL
   key numbers 3, 7, 6, 10… learned from data). The market alone produces **zero** edge, so any edge
   must come from the model.
3. **Trust coefficient (b1).** Each season is fit only on prior seasons:
   `result − market mean = b0 + b1·(model − line) + b2·rest diff`. b1 is how much to trust the model over
   the line, estimated, not assumed.
4. **Probabilities → bets.** Final mean margin → full margin distribution → P(cover / push / lose) →
   EV at your odds → push-aware fractional Kelly with a hard cap.
5. **Honest testing.** Parameters are tuned on training seasons (on forecast error, not profit); the
   holdout is never touched. Results are reported with Wilson CIs, a binomial test against break-even,
   bootstrap ROI CIs, calibration, and the sample size needed to prove an edge.
6. **CLV.** The ledger records the line you took vs. the closing line. Consistently beating the close is
   the strongest early evidence of an edge.
""")
