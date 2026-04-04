"""
ml/analytics.py
================
Advanced IPL analytics:
  - Season trends
  - Toss impact modelling
  - Win probability over overs
  - Orange/Purple Cap trackers
  - Team win heatmap data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _resolve_season(season_val) -> int:
    """Normalise season strings like '2007/08' → 2007, '2009' → 2009."""
    s = str(season_val).strip()
    return int(s.split("/")[0])


# ── Season-wise summary ───────────────────────────────────────────────────────

def season_summary(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches[matches["winner"].notna()].copy()
    df["_season"] = df["season"].apply(_resolve_season)

    rows = []
    for season, grp in df.groupby("_season"):
        total = len(grp)

        # Support both old (win_by_runs / win_by_wickets) and new (result / result_margin) schemas
        if "win_by_runs" in grp.columns and "win_by_wickets" in grp.columns:
            bat_wins  = int((grp["win_by_runs"] > 0).sum())
            chase_wins = int((grp["win_by_wickets"] > 0).sum())
            avg_bat   = grp.loc[grp["win_by_runs"] > 0, "win_by_runs"].mean()
            avg_chase = grp.loc[grp["win_by_wickets"] > 0, "win_by_wickets"].mean()
        elif "result" in grp.columns and "result_margin" in grp.columns:
            bat_wins  = int((grp["result"] == "runs").sum())
            chase_wins = int((grp["result"] == "wickets").sum())
            avg_bat   = grp.loc[grp["result"] == "runs", "result_margin"].mean()
            avg_chase = grp.loc[grp["result"] == "wickets", "result_margin"].mean()
        else:
            bat_wins, chase_wins = 0, 0
            avg_bat = avg_chase = float("nan")

        toss_match = (grp["toss_winner"] == grp["winner"]).sum()

        rows.append({
            "season":             int(season),
            "total_matches":      total,
            "bat_first_wins":     bat_wins,
            "chase_wins":         chase_wins,
            "bat_win_pct":        round(bat_wins / total * 100, 1) if total else 0,
            "toss_win_match_pct": round(toss_match / total * 100, 1) if total else 0,
            "avg_win_runs":       round(float(avg_bat), 1) if not (isinstance(avg_bat, float) and np.isnan(avg_bat)) else 0,
            "avg_win_wkts":       round(float(avg_chase), 1) if not (isinstance(avg_chase, float) and np.isnan(avg_chase)) else 0,
            "champion":           _get_champion(grp),
        })

    return pd.DataFrame(rows)


def _get_champion(season_df: pd.DataFrame) -> str:
    """Last match winner in a season = champion."""
    last = season_df.sort_values("id").tail(1)
    return last["winner"].values[0] if len(last) else "Unknown"


# ── Toss impact analysis ──────────────────────────────────────────────────────

def toss_impact_analysis(matches: pd.DataFrame) -> Dict:
    df = matches[matches["winner"].notna()].copy()
    df["_season"] = df["season"].apply(_resolve_season)
    total = len(df)
    toss_won_match = (df["toss_winner"] == df["winner"]).sum()

    bat_decide   = df[df["toss_decision"] == "bat"]
    field_decide = df[df["toss_decision"] == "field"]

    bat_win_pct   = (bat_decide["toss_winner"] == bat_decide["winner"]).mean() * 100
    field_win_pct = (field_decide["toss_winner"] == field_decide["winner"]).mean() * 100

    season_toss = df.groupby("_season").apply(
        lambda g: round((g["toss_winner"] == g["winner"]).mean() * 100, 1)
    ).reset_index(name="toss_win_pct")
    season_toss.rename(columns={"_season": "season"}, inplace=True)

    return {
        "overall_toss_win_pct":   round(toss_won_match / total * 100, 2),
        "bat_decision_win_pct":   round(float(bat_win_pct), 2),
        "field_decision_win_pct": round(float(field_win_pct), 2),
        "bat_decisions_count":    len(bat_decide),
        "field_decisions_count":  len(field_decide),
        "season_trend":           season_toss.to_dict("records"),
    }


# ── Orange / Purple cap trackers ──────────────────────────────────────────────

def orange_cap_tracker(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Top run scorers per season."""
    d = deliveries.copy()

    # Resolve batter column name (older datasets use 'batsman')
    batter_col = "batter" if "batter" in d.columns else "batsman"
    if batter_col not in d.columns:
        return pd.DataFrame()

    # Exclude wide deliveries
    if "wide_runs" in d.columns:
        d = d[d["wide_runs"] != 1]

    m = matches[["id", "season"]].copy()
    m["season"] = m["season"].apply(_resolve_season)
    d = d.merge(m, left_on="match_id", right_on="id", how="left")

    season_bat = (
        d.groupby(["season", batter_col])["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "runs", batter_col: "orange_cap_holder"})
    )

    top = (
        season_bat
        .sort_values(["season", "runs"], ascending=[True, False])
        .groupby("season")
        .head(1)
        .reset_index(drop=True)
    )
    return top


def purple_cap_tracker(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Top wicket takers per season."""
    d = deliveries.copy()
    if "player_dismissed" not in d.columns:
        return pd.DataFrame()

    m = matches[["id", "season"]].copy()
    m["season"] = m["season"].apply(_resolve_season)
    d = d.merge(m, left_on="match_id", right_on="id", how="left")

    season_bowl = (
        d[d["player_dismissed"].notna()]
        .groupby(["season", "bowler"])["player_dismissed"]
        .count()
        .reset_index(name="wickets")
    )

    top = (
        season_bowl
        .sort_values(["season", "wickets"], ascending=[True, False])
        .groupby("season")
        .head(1)
        .reset_index(drop=True)
        .rename(columns={"bowler": "purple_cap_holder"})
    )
    return top


# ── Win probability over overs ────────────────────────────────────────────────

def win_probability_over_overs(
    deliveries: pd.DataFrame,
    match_id: int,
    target: Optional[int] = None,
) -> pd.DataFrame:
    from ml.features import live_win_probability

    match_del = deliveries[deliveries["match_id"] == match_id].copy()
    if match_del.empty:
        return pd.DataFrame()

    rows = []
    for inning in [1, 2]:
        inn_del  = match_del[match_del["inning"] == inning].sort_values(["over", "ball"])
        cum_runs = 0
        cum_wkts = 0
        for over_num, over_grp in inn_del.groupby("over"):
            cum_runs += over_grp["total_runs"].sum()
            if "player_dismissed" in over_grp.columns:
                cum_wkts += over_grp["player_dismissed"].notna().sum()
            balls = (over_num + 1) * 6

            prob = live_win_probability(
                runs_scored=cum_runs,
                wickets_fallen=cum_wkts,
                balls_bowled=balls,
                target=target if inning == 2 else None,
                inning=inning,
            )
            rows.append({
                "inning":       inning,
                "over":         over_num + 1,
                "cum_runs":     cum_runs,
                "wickets":      cum_wkts,
                "bat_win_prob": prob["batting_team"],
                "bowl_win_prob": prob["bowling_team"],
            })

    return pd.DataFrame(rows)


# ── Powerplay / Middle / Death phase analysis ─────────────────────────────────

def phase_analysis(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown of scoring rates and wickets by match phase:
    Powerplay (1-6), Middle (7-15), Death (16-20).
    """
    df = deliveries.copy()
    df["phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 19],
        labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"],
    )

    has_dismissed = "player_dismissed" in df.columns

    summary = df.groupby("phase", observed=True).agg(
        total_runs = ("total_runs", "sum"),
        balls      = ("total_runs", "count"),
        wickets    = ("player_dismissed",
                      lambda x: x.notna().sum()) if has_dismissed else ("total_runs", lambda x: 0),
        boundaries = ("batsman_runs", lambda x: ((x == 4) | (x == 6)).sum()),
    ).reset_index()

    summary["run_rate"] = (summary["total_runs"] / summary["balls"] * 6).round(2)
    summary["balls"]    = summary["balls"].astype(int)
    return summary


# ── Top partnerships ──────────────────────────────────────────────────────────

def top_partnerships(deliveries: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top batting partnerships by runs."""
    if "non_striker" not in deliveries.columns:
        return pd.DataFrame()

    batter_col = "batter" if "batter" in deliveries.columns else "batsman"
    if batter_col not in deliveries.columns:
        return pd.DataFrame()

    df = (
        deliveries[deliveries["wide_runs"] != 1].copy()
        if "wide_runs" in deliveries.columns
        else deliveries.copy()
    )

    df["pair"] = df.apply(
        lambda r: " & ".join(sorted([str(r[batter_col]), str(r["non_striker"])])), axis=1
    )
    pairs = df.groupby("pair")["batsman_runs"].sum().reset_index()
    pairs.rename(columns={"batsman_runs": "partnership_runs"}, inplace=True)
    return pairs.sort_values("partnership_runs", ascending=False).head(top_n)
