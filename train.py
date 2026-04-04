"""
train.py
=========
One-command training script.
Run:  python train.py

Steps:
  1. Load matches.csv + deliveries.csv from data/
  2. Engineer all features
  3. Train match-winner model (XGBoost + LightGBM + RF stacking)
  4. Save model to data/models/
  5. Persist features to SQLite database
"""

import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger

import pandas as pd
from sqlalchemy.orm import Session

#  Setup paths 
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from database import (
    init_db, SessionLocal,
    Team, Player, Match, TeamStats, H2HStats, VenueStats, PlayerBattingStats,
)
from ml.features import (
    load_raw_data, compute_batting_stats, compute_bowling_stats,
    compute_team_stats, compute_h2h_stats, compute_venue_stats,
    build_ml_dataset, compute_player_impact,
)
from ml.predictor import train_model


def seed_database(
    db: Session,
    matches: pd.DataFrame,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    team_stats: pd.DataFrame,
    h2h: pd.DataFrame,
    venue: pd.DataFrame,
    player_impact: pd.DataFrame,
):
    """Persist all computed features into SQLite."""
    logger.info("Seeding database...")

    #  Teams 
    db.query(Team).delete()
    teams_list = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
    team_titles = {
        "Mumbai Indians": 5, "Chennai Super Kings": 5,
        "Kolkata Knight Riders": 3, "Sunrisers Hyderabad": 1,
        "Rajasthan Royals": 2, "Deccan Chargers": 1,
        "Gujarat Titans": 1,
    }
    for t in teams_list:
        db.add(Team(name=t, short_name=t[:3].upper(), titles=team_titles.get(t, 0)))
    db.commit()
    logger.info(f"   {len(teams_list)} teams")

    #  Matches 
    db.query(Match).delete()
    for _, row in matches.iterrows():
        db.add(Match(
            id=int(row["id"]),
            season=int(str(row["season"]).split("/")[0]),
            date=str(row.get("date","")),
            venue=str(row.get("venue","")),
            city=str(row.get("city","")),
            team1=str(row.get("team1","")),
            team2=str(row.get("team2","")),
            toss_winner=str(row.get("toss_winner","")),
            toss_decision=str(row.get("toss_decision","field")),
            winner=str(row.get("winner","")) if pd.notna(row.get("winner")) else None,
            result=str(row.get("result","normal")),
            win_by_runs=int(row.get("win_by_runs",0)),
            win_by_wickets=int(row.get("win_by_wickets",0)),
            player_of_match=str(row.get("player_of_match","")) if pd.notna(row.get("player_of_match")) else None,
            dl_applied=bool(row.get("dl_applied",0)),
        ))
    db.commit()
    logger.info(f"   {len(matches)} matches")

    #  Team stats 
    db.query(TeamStats).delete()
    for _, row in team_stats.iterrows():
        db.add(TeamStats(
            team=str(row["team"]),
            total_matches=int(row["total_matches"]),
            total_wins=int(row["total_wins"]),
            win_rate=float(row["win_rate"]),
            recent_form=float(row["recent_form"]),
            toss_win_rate=float(row["toss_win_rate"]),
            avg_win_runs=float(row["avg_win_runs"]),
            avg_win_wickets=float(row["avg_win_wickets"]),
            team_strength_index=float(row["team_strength_index"]),
        ))
    db.commit()
    logger.info(f"   {len(team_stats)} team stats")

    #  H2H 
    db.query(H2HStats).delete()
    for _, row in h2h.iterrows():
        db.add(H2HStats(
            team1=str(row["team1"]),
            team2=str(row["team2"]),
            total_meetings=int(row["total_meetings"]),
            team1_wins=int(row["team1_wins"]),
            team2_wins=int(row["team2_wins"]),
            team1_win_rate=float(row["team1_win_rate"]),
        ))
    db.commit()
    logger.info(f"   {len(h2h)} H2H records")

    #  Venue stats 
    db.query(VenueStats).delete()
    for _, row in venue.iterrows():
        db.add(VenueStats(
            venue=str(row["venue"]),
            city=str(row.get("city","")),
            total_matches=int(row["total_matches"]),
            bat_first_wins=int(row["bat_first_wins"]),
            bat_first_win_rate=float(row["bat_first_win_rate"]),
            avg_first_innings=float(row.get("avg_first_innings",0)),
        ))
    db.commit()
    logger.info(f"   {len(venue)} venue stats")

    #  Players 
    db.query(Player).delete()
    for _, row in player_impact.iterrows():
        pname = str(row["player"])
        bat_avg   = float(batting.loc[batting["batter"]==pname,"batting_avg"].values[0]) \
                    if pname in batting["batter"].values else 0.0
        sr        = float(batting.loc[batting["batter"]==pname,"strike_rate"].values[0]) \
                    if pname in batting["batter"].values else 0.0
        runs      = int(batting.loc[batting["batter"]==pname,"total_runs"].values[0]) \
                    if pname in batting["batter"].values else 0
        innings   = int(batting.loc[batting["batter"]==pname,"innings"].values[0]) \
                    if pname in batting["batter"].values else 0
        wkts      = int(bowling.loc[bowling["bowler"]==pname,"wickets"].values[0]) \
                    if pname in bowling["bowler"].values else 0
        econ      = float(bowling.loc[bowling["bowler"]==pname,"economy_rate"].values[0]) \
                    if pname in bowling["bowler"].values else 0.0

        db.add(Player(
            name=pname,
            role=str(row.get("role","Unknown")),
            total_runs=runs,
            innings=innings,
            batting_avg=bat_avg,
            strike_rate=sr,
            wickets=wkts,
            economy_rate=econ,
            batting_impact=float(row.get("batting_impact",0)),
            bowling_impact=float(row.get("bowling_impact",0)),
            player_impact=float(row.get("player_impact_score",0)),
        ))
    db.commit()
    logger.info(f"   {len(player_impact)} players")


def main(args):
    logger.info("=" * 55)
    logger.info("  IPL Analytics Platform  Training Pipeline")
    logger.info("=" * 55)

    #  1. Init DB 
    init_db()

    #  2. Load data 
    matches_path    = args.matches    or "data/matches.csv"
    deliveries_path = args.deliveries or "data/deliveries.csv"

    matches, deliveries = load_raw_data(matches_path, deliveries_path)

    #  3. Feature engineering 
    logger.info("Engineering features...")
    batting      = compute_batting_stats(deliveries)
    bowling      = compute_bowling_stats(deliveries)
    team_stats   = compute_team_stats(matches)
    h2h          = compute_h2h_stats(matches)
    venue        = compute_venue_stats(matches)
    player_impact= compute_player_impact(batting, bowling)
    ml_dataset   = build_ml_dataset(matches, team_stats, h2h, venue)

    logger.info(f"ML dataset shape: {ml_dataset.shape}")

    #  4. Save features as CSVs (for API/Streamlit to load fast) 
    os.makedirs("data/processed", exist_ok=True)
    batting.to_csv("data/processed/batting_stats.csv",     index=False)
    bowling.to_csv("data/processed/bowling_stats.csv",     index=False)
    team_stats.to_csv("data/processed/team_stats.csv",     index=False)
    h2h.to_csv("data/processed/h2h_stats.csv",             index=False)
    venue.to_csv("data/processed/venue_stats.csv",         index=False)
    player_impact.to_csv("data/processed/player_impact.csv", index=False)
    ml_dataset.to_csv("data/processed/ml_dataset.csv",     index=False)
    logger.info("   Feature CSVs saved to data/processed/")

    #  5. Train ML model 
    logger.info("Training ML model...")
    metrics = train_model(ml_dataset)

    logger.info("\n" + "" * 45)
    logger.info("  MODEL EVALUATION RESULTS")
    logger.info("" * 45)
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score  : {metrics['f1_score']:.4f}")
    logger.info(f"  Log Loss  : {metrics['log_loss']:.4f}")
    logger.info(f"  CV AUC    : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    logger.info(f"  Train time: {metrics['training_time_s']}s")
    logger.info("" * 45)

    #  6. Seed database 
    logger.info("Seeding database...")
    db = SessionLocal()
    try:
        seed_database(db, matches, batting, bowling, team_stats, h2h, venue, player_impact)
    finally:
        db.close()

    logger.success("\n  Training complete!")
    logger.success("   Run:  streamlit run app.py   to launch the dashboard")
    logger.success("   Run:  uvicorn api:app --reload   to start the REST API")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL Analytics Training Pipeline")
    parser.add_argument("--matches",    default=None, help="Path to matches.csv")
    parser.add_argument("--deliveries", default=None, help="Path to deliveries.csv")
    args = parser.parse_args()
    main(args)
