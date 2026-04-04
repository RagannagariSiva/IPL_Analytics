"""
tests/test_features.py
=======================
Unit tests for feature engineering pipeline.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import (
    normalize_teams, compute_batting_stats, compute_bowling_stats,
    compute_team_stats, compute_h2h_stats, compute_venue_stats,
    build_ml_dataset, live_win_probability, TEAM_MAP,
)


#  Fixtures 

@pytest.fixture
def sample_matches():
    return pd.DataFrame([
        {"id":1,"season":2022,"city":"Mumbai","date":"2022-04-02","venue":"Wankhede Stadium",
         "team1":"Mumbai Indians","team2":"Chennai Super Kings",
         "toss_winner":"Mumbai Indians","toss_decision":"bat",
         "winner":"Mumbai Indians","result":"normal",
         "win_by_runs":20,"win_by_wickets":0,"player_of_match":"P1","dl_applied":0},
        {"id":2,"season":2022,"city":"Chennai","date":"2022-04-03","venue":"MA Chidambaram Stadium",
         "team1":"Chennai Super Kings","team2":"Mumbai Indians",
         "toss_winner":"Chennai Super Kings","toss_decision":"field",
         "winner":"Chennai Super Kings","result":"normal",
         "win_by_runs":0,"win_by_wickets":5,"player_of_match":"P2","dl_applied":0},
        {"id":3,"season":2022,"city":"Kolkata","date":"2022-04-05","venue":"Eden Gardens",
         "team1":"Kolkata Knight Riders","team2":"Mumbai Indians",
         "toss_winner":"Mumbai Indians","toss_decision":"bat",
         "winner":"Mumbai Indians","result":"normal",
         "win_by_runs":15,"win_by_wickets":0,"player_of_match":"P1","dl_applied":0},
        {"id":4,"season":2023,"city":"Mumbai","date":"2023-04-01","venue":"Wankhede Stadium",
         "team1":"Mumbai Indians","team2":"Kolkata Knight Riders",
         "toss_winner":"Kolkata Knight Riders","toss_decision":"field",
         "winner":"Kolkata Knight Riders","result":"normal",
         "win_by_runs":0,"win_by_wickets":3,"player_of_match":"P3","dl_applied":0},
    ])


@pytest.fixture
def sample_deliveries():
    rows = []
    for match_id in [1,2,3,4]:
        for inning in [1,2]:
            for over in range(10):
                for ball in range(1,7):
                    rows.append({
                        "match_id": match_id, "inning": inning,
                        "batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings",
                        "over": over, "ball": ball,
                        "batsman": "Batsman_A", "non_striker": "Batsman_B",
                        "bowler": "Bowler_X",
                        "wide_runs": 0, "batsman_runs": np.random.choice([0,1,2,4,6]),
                        "extra_runs": 0, "total_runs": np.random.choice([0,1,2,4,6]),
                        "player_dismissed": None if np.random.random() > 0.1 else "Batsman_A",
                        "dismissal_kind": None,
                    })
    return pd.DataFrame(rows)


#  Tests 

class TestNormalizeTeams:
    def test_maps_old_names(self):
        df = pd.DataFrame({"team1": ["Delhi Daredevils","Deccan Chargers"]})
        result = normalize_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Delhi Capitals"
        assert result["team1"].iloc[1] == "Sunrisers Hyderabad"

    def test_keeps_current_names(self):
        df = pd.DataFrame({"team1": ["Mumbai Indians","Chennai Super Kings"]})
        result = normalize_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Mumbai Indians"


class TestBattingStats:
    def test_returns_dataframe(self, sample_deliveries):
        result = compute_batting_stats(sample_deliveries)
        assert isinstance(result, pd.DataFrame)
        assert "batsman" in result.columns

    def test_total_runs_positive(self, sample_deliveries):
        result = compute_batting_stats(sample_deliveries)
        assert (result["total_runs"] >= 0).all()

    def test_strike_rate_positive(self, sample_deliveries):
        result = compute_batting_stats(sample_deliveries)
        assert (result["strike_rate"] >= 0).all()

    def test_batting_impact_range(self, sample_deliveries):
        result = compute_batting_stats(sample_deliveries)
        assert (result["batting_impact"] >= 0).all()
        assert (result["batting_impact"] <= 100).all()


class TestBowlingStats:
    def test_returns_dataframe(self, sample_deliveries):
        result = compute_bowling_stats(sample_deliveries)
        assert isinstance(result, pd.DataFrame)
        assert "bowler" in result.columns

    def test_economy_positive(self, sample_deliveries):
        result = compute_bowling_stats(sample_deliveries)
        assert (result["economy_rate"] >= 0).all()


class TestTeamStats:
    def test_all_teams_present(self, sample_matches):
        result = compute_team_stats(sample_matches)
        teams_in = set(result["team"].tolist())
        assert "Mumbai Indians" in teams_in
        assert "Chennai Super Kings" in teams_in

    def test_win_rate_between_0_and_1(self, sample_matches):
        result = compute_team_stats(sample_matches)
        assert (result["win_rate"] >= 0).all()
        assert (result["win_rate"] <= 1).all()

    def test_strength_index_positive(self, sample_matches):
        result = compute_team_stats(sample_matches)
        assert (result["team_strength_index"] >= 0).all()


class TestH2HStats:
    def test_returns_dataframe(self, sample_matches):
        result = compute_h2h_stats(sample_matches)
        assert isinstance(result, pd.DataFrame)

    def test_win_rate_sums_correctly(self, sample_matches):
        result = compute_h2h_stats(sample_matches)
        for _, row in result.iterrows():
            total = row["team1_wins"] + row["team2_wins"]
            assert total <= row["total_meetings"]


class TestLiveWinProbability:
    def test_inning1_returns_dict(self):
        p = live_win_probability(80, 3, 60, inning=1)
        assert "batting_team" in p
        assert "bowling_team" in p

    def test_probabilities_sum_to_1(self):
        p = live_win_probability(80, 3, 60, target=160, inning=2)
        assert abs(p["batting_team"] + p["bowling_team"] - 1.0) < 0.001

    def test_target_met_returns_1(self):
        p = live_win_probability(171, 5, 100, target=170, inning=2)
        assert p["batting_team"] == 1.0

    def test_all_balls_bowled_batting_loses(self):
        p = live_win_probability(100, 5, 120, target=170, inning=2)
        assert p["batting_team"] == 0.0

    def test_probability_in_range(self):
        for runs in [50, 100, 150]:
            for wickets in [0, 5, 9]:
                p = live_win_probability(runs, wickets, 72, target=160, inning=2)
                assert 0 <= p["batting_team"] <= 1
                assert 0 <= p["bowling_team"] <= 1


class TestMLDataset:
    def test_returns_dataframe_with_target(self, sample_matches):
        ts = compute_team_stats(sample_matches)
        h2h = compute_h2h_stats(sample_matches)
        vs = compute_venue_stats(sample_matches)
        ml = build_ml_dataset(sample_matches, ts, h2h, vs)
        assert isinstance(ml, pd.DataFrame)
        assert "target" in ml.columns
        assert ml["target"].isin([0,1]).all()

    def test_no_missing_target(self, sample_matches):
        ts = compute_team_stats(sample_matches)
        h2h = compute_h2h_stats(sample_matches)
        vs = compute_venue_stats(sample_matches)
        ml = build_ml_dataset(sample_matches, ts, h2h, vs)
        assert ml["target"].isna().sum() == 0
