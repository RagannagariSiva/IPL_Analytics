"""
database.py
============
SQLite database setup using SQLAlchemy.
Stores matches, players, teams, predictions, and analytics.
No external database server required - runs fully local & on cloud.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Boolean, Text, ForeignKey, JSON
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, Session
from loguru import logger

#  Database path 
DB_PATH = os.getenv("DATABASE_PATH", "./data/ipl_analytics.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


#  ORM Models 

class Team(Base):
    __tablename__ = "teams"
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(120), unique=True, nullable=False, index=True)
    short_name  = Column(String(10))
    city        = Column(String(100))
    home_ground = Column(String(200))
    titles      = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)


class Player(Base):
    __tablename__ = "players"
    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(200), nullable=False, index=True)
    role            = Column(String(50))   # batsman | bowler | all-rounder | wk
    nationality     = Column(String(100))
    is_overseas     = Column(Boolean, default=False)
    total_runs      = Column(Integer, default=0)
    innings         = Column(Integer, default=0)
    dismissals      = Column(Integer, default=0)
    balls_faced     = Column(Integer, default=0)
    fours           = Column(Integer, default=0)
    sixes           = Column(Integer, default=0)
    batting_avg     = Column(Float, default=0.0)
    strike_rate     = Column(Float, default=0.0)
    wickets         = Column(Integer, default=0)
    economy_rate    = Column(Float, default=0.0)
    bowling_avg     = Column(Float, default=0.0)
    batting_impact  = Column(Float, default=0.0)
    bowling_impact  = Column(Float, default=0.0)
    player_impact   = Column(Float, default=0.0)
    updated_at      = Column(DateTime, default=datetime.utcnow)


class Match(Base):
    __tablename__ = "matches"
    id              = Column(Integer, primary_key=True, index=True)
    season          = Column(Integer, nullable=False, index=True)
    date            = Column(String(20))
    venue           = Column(String(300))
    city            = Column(String(100))
    team1           = Column(String(120))
    team2           = Column(String(120))
    toss_winner     = Column(String(120))
    toss_decision   = Column(String(10))
    winner          = Column(String(120))
    result          = Column(String(50))
    win_by_runs     = Column(Integer, default=0)
    win_by_wickets  = Column(Integer, default=0)
    player_of_match = Column(String(200))
    dl_applied      = Column(Boolean, default=False)


class TeamStats(Base):
    __tablename__ = "team_stats"
    id                   = Column(Integer, primary_key=True, index=True)
    team                 = Column(String(120), unique=True, nullable=False, index=True)
    total_matches        = Column(Integer, default=0)
    total_wins           = Column(Integer, default=0)
    win_rate             = Column(Float, default=0.0)
    recent_form          = Column(Float, default=0.0)
    toss_win_rate        = Column(Float, default=0.0)
    avg_win_runs         = Column(Float, default=0.0)
    avg_win_wickets      = Column(Float, default=0.0)
    team_strength_index  = Column(Float, default=0.0)
    updated_at           = Column(DateTime, default=datetime.utcnow)


class H2HStats(Base):
    __tablename__ = "h2h_stats"
    id             = Column(Integer, primary_key=True, index=True)
    team1          = Column(String(120), nullable=False)
    team2          = Column(String(120), nullable=False)
    total_meetings = Column(Integer, default=0)
    team1_wins     = Column(Integer, default=0)
    team2_wins     = Column(Integer, default=0)
    team1_win_rate = Column(Float, default=0.0)


class VenueStats(Base):
    __tablename__ = "venue_stats"
    id                  = Column(Integer, primary_key=True, index=True)
    venue               = Column(String(300), unique=True, nullable=False)
    city                = Column(String(100))
    total_matches       = Column(Integer, default=0)
    bat_first_wins      = Column(Integer, default=0)
    bat_first_win_rate  = Column(Float, default=0.0)
    avg_first_innings   = Column(Float, default=0.0)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id              = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50))   # match_winner | player_perf
    team1           = Column(String(120))
    team2           = Column(String(120))
    predicted_winner= Column(String(120))
    team1_win_prob  = Column(Float)
    team2_win_prob  = Column(Float)
    confidence      = Column(Float)
    input_data      = Column(JSON)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class PlayerBattingStats(Base):
    __tablename__ = "player_batting_stats"
    id              = Column(Integer, primary_key=True, index=True)
    player          = Column(String(200), nullable=False)
    team            = Column(String(120))
    matches         = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    balls_faced     = Column(Integer, default=0)
    avg             = Column(Float, default=0.0)
    strike_rate     = Column(Float, default=0.0)
    fifties         = Column(Integer, default=0)
    hundreds        = Column(Integer, default=0)

#  Helpers 

def get_db() -> Session:
    """Yield a DB session (use as FastAPI dependency or context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at {DB_PATH}")
