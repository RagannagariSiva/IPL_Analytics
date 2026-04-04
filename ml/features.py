"""
ml/features.py
===============
IPL Feature Engineering Pipeline.
Transforms raw matches.csv + deliveries.csv into ML-ready features.
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Tuple, Optional


#  Team name normalisation map 
TEAM_MAP = {
    "Delhi Daredevils":          "Delhi Capitals",
    "Deccan Chargers":           "Sunrisers Hyderabad",
    "Rising Pune Supergiant":    "Rising Pune Supergiants",
    "Kings XI Punjab":           "Punjab Kings",
    "Pune Warriors":             "Pune Warriors India",
}


def normalize_teams(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(TEAM_MAP)
    return df


#  Load raw CSVs 

def load_raw_data(
    matches_path: str = "data/matches.csv",
    deliveries_path: str = "data/deliveries.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and basic-clean IPL CSVs."""
    if not os.path.exists(matches_path):
        raise FileNotFoundError(
            f"matches.csv not found at '{matches_path}'.\n"
            "Download from Kaggle  see README.md for instructions."
        )
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(
            f"deliveries.csv not found at '{deliveries_path}'.\n"
            "Download from Kaggle  see README.md for instructions."
        )

    matches = pd.read_csv(matches_path, low_memory=False)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    matches.columns    = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    deliveries.columns = deliveries.columns.str.strip().str.lower().str.replace(" ", "_")

    matches    = normalize_teams(matches,    ["team1","team2","toss_winner","winner"])
    deliveries = normalize_teams(deliveries, ["batting_team","bowling_team"])

    # NEW dataset uses result_margin + result instead of win_by_runs / win_by_wickets
    # Derive win_by_runs and win_by_wickets for backward compatibility
    if "result_margin" in matches.columns and "result" in matches.columns:
        matches["result_margin"] = pd.to_numeric(matches["result_margin"], errors="coerce").fillna(0)
        matches["win_by_runs"]     = matches.apply(
            lambda r: int(r["result_margin"]) if str(r["result"]).lower() == "runs" else 0, axis=1
        )
        matches["win_by_wickets"]  = matches.apply(
            lambda r: int(r["result_margin"]) if str(r["result"]).lower() == "wickets" else 0, axis=1
        )
    else:
        # Legacy dataset already has these columns – just coerce them
        for col in ["win_by_runs", "win_by_wickets"]:
            if col in matches.columns:
                matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0).astype(int)

    # Coerce dl_applied if present
    if "dl_applied" in matches.columns:
        matches["dl_applied"] = pd.to_numeric(matches["dl_applied"], errors="coerce").fillna(0).astype(int)

    logger.info(f"Loaded {len(matches)} matches, {len(deliveries)} deliveries")
    return matches, deliveries


#  Batting stats 

def compute_batting_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player batting statistics.
    Returns DataFrame indexed by player name.
    """
    df = deliveries[deliveries.get("wide_runs", pd.Series(dtype=int)) != 1].copy() \
        if "wide_runs" in deliveries.columns else deliveries.copy()

    bat = df.groupby("batter").agg(
        total_runs   = ("batsman_runs", "sum"),
        balls_faced  = ("batsman_runs", "count"),
        fours        = ("batsman_runs", lambda x: (x == 4).sum()),
        sixes        = ("batsman_runs", lambda x: (x == 6).sum()),
        dot_balls    = ("batsman_runs", lambda x: (x == 0).sum()),
        innings      = ("match_id", "nunique"),
    ).reset_index()

    # dismissals
    if "player_dismissed" in deliveries.columns:
        dis = (
            deliveries[deliveries["player_dismissed"].notna()]
            .groupby("player_dismissed").size().reset_index(name="dismissals")
        )
        dis.rename(columns={"player_dismissed": "batter"}, inplace=True)
        bat = bat.merge(dis, on="batter", how="left")
    bat["dismissals"] = bat.get("dismissals", 0).fillna(0).astype(int)

    bat["batting_avg"]  = np.where(bat["dismissals"] > 0,
                                   bat["total_runs"] / bat["dismissals"],
                                   bat["total_runs"])
    bat["strike_rate"]  = (bat["total_runs"] / bat["balls_faced"].replace(0, 1)) * 100
    bat["boundary_pct"] = ((bat["fours"]*4 + bat["sixes"]*6) /
                           bat["total_runs"].replace(0, 1)) * 100

    # Batting Impact Score (0-100)
    for c in ["batting_avg","strike_rate","boundary_pct"]:
        rng = bat[c].max() - bat[c].min()
        bat[f"{c}_norm"] = (bat[c] - bat[c].min()) / rng if rng else 0
    bat["batting_impact"] = (
        bat["batting_avg_norm"]  * 35 +
        bat["strike_rate_norm"]  * 35 +
        bat["boundary_pct_norm"] * 20 +
        (bat["innings"] / bat["innings"].max()) * 10
    ).round(2)

    return bat.sort_values("total_runs", ascending=False)


#  Bowling stats 

def compute_bowling_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per-player bowling statistics."""
    bowl = deliveries.groupby("bowler").agg(
        total_balls       = ("total_runs", "count"),
        runs_conceded     = ("total_runs", "sum"),
        wickets           = ("player_dismissed", lambda x: x.notna().sum()
                              if "player_dismissed" in deliveries.columns else 0),
        dot_balls         = ("total_runs", lambda x: (x == 0).sum()),
        matches           = ("match_id", "nunique"),
    ).reset_index()

    bowl["overs"]         = bowl["total_balls"] / 6
    bowl["economy_rate"]  = bowl["runs_conceded"] / bowl["overs"].replace(0, 1)
    bowl["bowling_avg"]   = np.where(bowl["wickets"] > 0,
                                     bowl["runs_conceded"] / bowl["wickets"], 999)
    bowl["bowling_sr"]    = np.where(bowl["wickets"] > 0,
                                     bowl["total_balls"] / bowl["wickets"], 999)
    bowl["dot_ball_pct"]  = bowl["dot_balls"] / bowl["total_balls"].replace(0, 1) * 100

    # Bowling Impact Score (0-100)  lower economy & more wickets = better
    bowl["economy_inv_norm"] = 1 - (bowl["economy_rate"] - bowl["economy_rate"].min()) / \
                               (bowl["economy_rate"].max() - bowl["economy_rate"].min() + 1e-9)
    bowl["wickets_norm"]     = bowl["wickets"] / bowl["wickets"].max()
    bowl["dot_norm"]         = bowl["dot_ball_pct"] / 100
    bowl["bowling_impact"]   = (
        bowl["economy_inv_norm"] * 35 +
        bowl["wickets_norm"]     * 40 +
        bowl["dot_norm"]         * 25
    ).round(2)

    return bowl.sort_values("wickets", ascending=False)


#  Team stats 

def compute_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-team aggregate win/loss/form/toss stats."""
    df = matches[matches["winner"].notna()].copy()
    teams = sorted(set(df["team1"].tolist() + df["team2"].tolist()))
    rows = []

    for team in teams:
        played = df[(df["team1"] == team) | (df["team2"] == team)]
        won    = played[played["winner"] == team]
        total  = len(played)
        wins   = len(won)

        toss_won = played[played["toss_winner"] == team]
        toss_win_rate = (len(toss_won[toss_won["winner"] == team]) / len(toss_won)
                         if len(toss_won) > 0 else 0.5)

        recent = played.sort_values("id").tail(10)
        recent_form = len(recent[recent["winner"] == team]) / len(recent) if len(recent) else 0

        # Use derived win_by_runs / win_by_wickets columns (created in load_raw_data)
        avg_win_runs = won["win_by_runs"].mean()
        avg_win_wkts = won["win_by_wickets"].mean()

        strength = (
            (wins / total if total else 0) * 40 +
            recent_form * 30 +
            toss_win_rate * 10 +
            min(wins / 20, 1.0) * 20
        )
        rows.append({
            "team":                team,
            "total_matches":       total,
            "total_wins":          wins,
            "win_rate":            round(wins / total, 4) if total else 0,
            "recent_form":         round(recent_form, 4),
            "toss_win_rate":       round(toss_win_rate, 4),
            "avg_win_runs":        round(float(avg_win_runs), 2) if not np.isnan(avg_win_runs) else 0,
            "avg_win_wickets":     round(float(avg_win_wkts), 2) if not np.isnan(avg_win_wkts) else 0,
            "team_strength_index": round(strength, 4),
        })

    return pd.DataFrame(rows).sort_values("team_strength_index", ascending=False)


#  Head-to-Head stats 

def compute_h2h_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Pairwise head-to-head win rates."""
    df = matches[matches["winner"].notna()].copy()
    teams = sorted(set(df["team1"].tolist() + df["team2"].tolist()))
    rows = []

    for i, t1 in enumerate(teams):
        for t2 in teams[i+1:]:
            m = df[
                ((df["team1"] == t1) & (df["team2"] == t2)) |
                ((df["team1"] == t2) & (df["team2"] == t1))
            ]
            if len(m) == 0:
                continue
            t1w = len(m[m["winner"] == t1])
            t2w = len(m[m["winner"] == t2])
            rows.append({
                "team1": t1, "team2": t2,
                "total_meetings": len(m),
                "team1_wins": t1w, "team2_wins": t2w,
                "team1_win_rate": round(t1w / len(m), 4),
            })
    return pd.DataFrame(rows)


#  Venue stats 

def compute_venue_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Venue-level match outcome statistics."""
    df = matches[matches["winner"].notna()].copy()
    rows = []
    for venue, grp in df.groupby("venue"):
        # win_by_runs / win_by_wickets derived in load_raw_data
        bat_wins   = (grp["win_by_runs"] > 0).sum()
        total      = len(grp)
        rows.append({
            "venue":               venue,
            "city":                grp["city"].iloc[0] if "city" in grp.columns else "",
            "total_matches":       total,
            "bat_first_wins":      int(bat_wins),
            "bat_first_win_rate":  round(bat_wins / total, 4) if total else 0.5,
            "avg_first_innings":   round(grp["win_by_runs"].mean(), 1),
        })
    return pd.DataFrame(rows).sort_values("total_matches", ascending=False)


#  ML Training Dataset 

def build_ml_dataset(
    matches: pd.DataFrame,
    team_stats: pd.DataFrame,
    h2h: pd.DataFrame,
    venue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final ML-ready dataset for match winner prediction.
    Target = 1 if team1 wins, 0 if team2 wins.
    """
    df = matches[matches["winner"].notna()].copy()
    df["target"] = (df["winner"] == df["team1"]).astype(int)

    ts  = team_stats.set_index("team")
    vs  = venue.set_index("venue")

    def tstat(team, col, default=0.5):
        try:    return float(ts.loc[team, col])
        except: return default

    def vstat(v, col, default=0.5):
        try:    return float(vs.loc[v, col])
        except: return default

    def h2h_wr(t1, t2):
        r = h2h[((h2h["team1"]==t1)&(h2h["team2"]==t2))|
                ((h2h["team1"]==t2)&(h2h["team2"]==t1))]
        if len(r)==0: return 0.5
        row=r.iloc[0]
        return float(row["team1_win_rate"]) if row["team1"]==t1 else 1-float(row["team1_win_rate"])

    records = []
    for _, m in df.iterrows():
        t1, t2, v = m["team1"], m["team2"], m.get("venue","")
        records.append({
            "match_id":               m["id"],
            "season":                 int(str(m["season"]).split("/")[0]),
            "toss_winner_is_team1":   int(m["toss_winner"] == t1),
            "toss_decision_bat":      int(m.get("toss_decision","field") == "bat"),
            "t1_win_rate":            tstat(t1,"win_rate"),
            "t2_win_rate":            tstat(t2,"win_rate"),
            "t1_strength":            tstat(t1,"team_strength_index"),
            "t2_strength":            tstat(t2,"team_strength_index"),
            "t1_recent_form":         tstat(t1,"recent_form"),
            "t2_recent_form":         tstat(t2,"recent_form"),
            "t1_toss_wr":             tstat(t1,"toss_win_rate"),
            "t2_toss_wr":             tstat(t2,"toss_win_rate"),
            "h2h_win_rate_t1":        h2h_wr(t1,t2),
            "venue_bat_win_rate":     vstat(v,"bat_first_win_rate"),
            "win_rate_diff":          tstat(t1,"win_rate") - tstat(t2,"win_rate"),
            "strength_diff":          tstat(t1,"team_strength_index") - tstat(t2,"team_strength_index"),
            "form_diff":              tstat(t1,"recent_form") - tstat(t2,"recent_form"),
            "target":                 int(m["target"]),
        })
    return pd.DataFrame(records)


#  Player Impact Score 

def compute_player_impact(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine batting + bowling impact into a single Player Impact Score (PIS).
    Categorises players by dominant role.
    """
    bat = batting[["batter","batting_impact","total_runs",
                   "batting_avg","strike_rate","innings"]].copy()
    bat.rename(columns={"batter":"player"}, inplace=True)

    bowl = bowling[["bowler","bowling_impact","wickets",
                    "economy_rate","bowling_avg"]].copy()
    bowl.rename(columns={"bowler":"player"}, inplace=True)

    merged = bat.merge(bowl, on="player", how="outer").fillna(0)
    merged["player_impact_score"] = (
        merged["batting_impact"]  * 0.55 +
        merged["bowling_impact"]  * 0.45
    ).round(2)
    merged["role"] = merged.apply(_infer_role, axis=1)
    return merged.sort_values("player_impact_score", ascending=False)


def _infer_role(row) -> str:
    b_imp = row.get("batting_impact", 0)
    w_imp = row.get("bowling_impact", 0)
    if b_imp > 50 and w_imp > 50:
        return "All-Rounder"
    if b_imp > w_imp:
        return "Batsman"
    if w_imp > b_imp:
        return "Bowler"
    return "Unknown"


# ── Win Probability (Live) ───────────────────────────────────────────────────

# Historical IPL baseline constants (derived from 2008-2024 dataset)
_IPL_AVG_FIRST_INNINGS   = 162   # average first-innings total
_IPL_STD_FIRST_INNINGS   = 24    # std of first-innings total

def live_win_probability(
    runs_scored: int,
    wickets_fallen: int,
    balls_bowled: int,
    target: Optional[int] = None,
    inning: int = 1,
) -> Dict[str, float]:
    """
    Statistically grounded live win probability estimator.

    Inning 1  — projects final score from CRR and wickets in hand,
                then applies a logistic model relative to the IPL average.
    Inning 2  — combines run-rate gap and wicket pressure using a
                calibrated sigmoid, with hard boundary conditions.

    Returns dict: {batting_team: float, bowling_team: float}
    """
    # --- hard clamp inputs ---
    wickets_fallen  = int(np.clip(wickets_fallen, 0, 10))
    balls_bowled    = int(np.clip(balls_bowled,   0, 120))
    runs_scored     = max(0, int(runs_scored))

    balls_remaining   = 120 - balls_bowled
    wickets_remaining = 10  - wickets_fallen
    all_out           = wickets_remaining == 0
    innings_complete  = all_out or balls_remaining == 0

    # ── INNING 1 ────────────────────────────────────────────────────────────
    if inning == 1:
        if innings_complete:
            # Innings is over – batting team has posted their score.
            # Bowling team now needs to chase: estimate their success odds.
            # Logistic centred on _IPL_AVG_FIRST_INNINGS with ~24-run spread.
            z = (runs_scored - _IPL_AVG_FIRST_INNINGS) / _IPL_STD_FIRST_INNINGS
            # Positive z → higher score → batting team posted good total → better position
            win_prob = float(1 / (1 + np.exp(-z * 0.9)))
            win_prob = float(np.clip(win_prob, 0.08, 0.92))
        else:
            if balls_bowled == 0:
                return {"batting_team": 0.5, "bowling_team": 0.5}
            crr = runs_scored / balls_bowled * 6

            # Wicket-resource weight: more wickets in hand → can accelerate
            wkt_weight = 0.5 + (wickets_remaining / 10) * 0.5  # 0.5 – 1.0
            projected  = runs_scored + (crr * wkt_weight) * (balls_remaining / 6)

            z = (projected - _IPL_AVG_FIRST_INNINGS) / _IPL_STD_FIRST_INNINGS
            win_prob = float(1 / (1 + np.exp(-z * 0.9)))
            win_prob = float(np.clip(win_prob, 0.10, 0.90))

    # ── INNING 2 ────────────────────────────────────────────────────────────
    else:
        if target is None:
            target = _IPL_AVG_FIRST_INNINGS + 1

        runs_needed = target - runs_scored

        # Hard boundaries first
        if runs_needed <= 0:
            return {"batting_team": 1.0, "bowling_team": 0.0}

        if all_out or (balls_remaining <= 0 and runs_needed > 0):
            return {"batting_team": 0.0, "bowling_team": 1.0}

        # Ongoing chase
        rrr = runs_needed / (balls_remaining / 6)
        crr = (runs_scored / (balls_bowled / 6)) if balls_bowled > 0 else 0.0

        # Run-rate gap: positive → bowling team favoured
        rate_gap = rrr - crr

        # Wicket pressure: heavier when most wickets gone, especially late
        overs_done     = balls_bowled / 6
        wicket_penalty = (wickets_fallen / 10) ** 1.5 * 0.35  # 0 → 0.35

        # Urgency: approaching end of innings amplifies both factors
        urgency = 1.0 + (balls_bowled / 120) * 0.4  # 1.0 → 1.4

        z = -(rate_gap * 0.10 + wicket_penalty) * urgency
        win_prob = float(1 / (1 + np.exp(-z)))
        win_prob = float(np.clip(win_prob, 0.02, 0.98))

    return {
        "batting_team": round(win_prob, 4),
        "bowling_team": round(1 - win_prob, 4),
    }
