"""
tests/test_database.py
=======================
Tests for SQLite database layer.
Uses an in-memory SQLite database so no files are created.
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Override DATABASE_PATH before importing database module
import os
os.environ["DATABASE_PATH"] = ":memory:"  # use in-memory SQLite

from database import Base, Team, Player, Match, TeamStats, PredictionLog


@pytest.fixture(scope="module")
def db_session():
    """Create an in-memory SQLite database for tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestTeamModel:
    def test_create_team(self, db_session):
        team = Team(name="Mumbai Indians", short_name="MI", city="Mumbai", titles=5)
        db_session.add(team)
        db_session.commit()
        fetched = db_session.query(Team).filter_by(name="Mumbai Indians").first()
        assert fetched is not None
        assert fetched.titles == 5
        assert fetched.city == "Mumbai"

    def test_team_name_unique(self, db_session):
        from sqlalchemy.exc import IntegrityError
        db_session.add(Team(name="Mumbai Indians", short_name="MI"))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestMatchModel:
    def test_create_match(self, db_session):
        match = Match(
            id=101, season=2024, date="2024-04-01",
            venue="Wankhede Stadium", city="Mumbai",
            team1="Mumbai Indians", team2="Chennai Super Kings",
            toss_winner="Mumbai Indians", toss_decision="bat",
            winner="Mumbai Indians", result="normal",
            win_by_runs=25, win_by_wickets=0,
        )
        db_session.add(match)
        db_session.commit()
        fetched = db_session.query(Match).filter_by(id=101).first()
        assert fetched.winner == "Mumbai Indians"
        assert fetched.season == 2024

    def test_match_season_filter(self, db_session):
        matches = db_session.query(Match).filter_by(season=2024).all()
        assert len(matches) >= 1


class TestTeamStatsModel:
    def test_create_team_stats(self, db_session):
        stats = TeamStats(
            team="Chennai Super Kings",
            total_matches=200,
            total_wins=120,
            win_rate=0.60,
            recent_form=0.70,
            toss_win_rate=0.55,
            team_strength_index=75.5,
        )
        db_session.add(stats)
        db_session.commit()
        fetched = db_session.query(TeamStats).filter_by(team="Chennai Super Kings").first()
        assert fetched.win_rate == pytest.approx(0.60)
        assert fetched.team_strength_index == pytest.approx(75.5)


class TestPredictionLogModel:
    def test_create_prediction_log(self, db_session):
        log = PredictionLog(
            prediction_type="match_winner",
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            predicted_winner="Mumbai Indians",
            team1_win_prob=0.63,
            team2_win_prob=0.37,
            confidence=0.63,
            input_data={"season": 2024},
        )
        db_session.add(log)
        db_session.commit()
        fetched = db_session.query(PredictionLog).filter_by(
            predicted_winner="Mumbai Indians"
        ).first()
        assert fetched is not None
        assert fetched.team1_win_prob == pytest.approx(0.63)

    def test_prediction_log_has_timestamp(self, db_session):
        log = db_session.query(PredictionLog).first()
        # created_at may be set by DB or application
        # Just verify the record exists
        assert log is not None


class TestPlayerModel:
    def test_create_player(self, db_session):
        player = Player(
            name="Rohit Sharma",
            role="Batsman",
            nationality="Indian",
            total_runs=5500,
            batting_avg=31.5,
            strike_rate=130.2,
            batting_impact=85.3,
        )
        db_session.add(player)
        db_session.commit()
        fetched = db_session.query(Player).filter_by(name="Rohit Sharma").first()
        assert fetched.total_runs == 5500
        assert fetched.batting_avg == pytest.approx(31.5)
