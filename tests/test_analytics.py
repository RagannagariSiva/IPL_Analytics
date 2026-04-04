"""
tests/test_analytics.py
========================
Tests for advanced analytics functions.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.analytics import (
    season_summary, toss_impact_analysis,
    phase_analysis, _get_champion,
)


@pytest.fixture
def sample_matches():
    return pd.DataFrame([
        {"id":1,"season":2022,"city":"Mumbai","date":"2022-04-02","venue":"Wankhede Stadium",
         "team1":"Mumbai Indians","team2":"Chennai Super Kings",
         "toss_winner":"Mumbai Indians","toss_decision":"bat",
         "winner":"Mumbai Indians","result":"normal",
         "win_by_runs":20,"win_by_wickets":0,"player_of_match":"P1","dl_applied":0},
        {"id":2,"season":2022,"city":"Chennai","date":"2022-04-05","venue":"MA Chidambaram Stadium",
         "team1":"Chennai Super Kings","team2":"Mumbai Indians",
         "toss_winner":"Chennai Super Kings","toss_decision":"field",
         "winner":"Chennai Super Kings","result":"normal",
         "win_by_runs":0,"win_by_wickets":5,"player_of_match":"P2","dl_applied":0},
        {"id":3,"season":2023,"city":"Kolkata","date":"2023-04-01","venue":"Eden Gardens",
         "team1":"Kolkata Knight Riders","team2":"Mumbai Indians",
         "toss_winner":"Mumbai Indians","toss_decision":"bat",
         "winner":"Mumbai Indians","result":"normal",
         "win_by_runs":15,"win_by_wickets":0,"player_of_match":"P1","dl_applied":0},
    ])


@pytest.fixture
def sample_deliveries():
    rows = []
    for match_id in range(1, 5):
        for inning in [1, 2]:
            for ov in range(20):
                for ball in range(1, 7):
                    rows.append({
                        "match_id": match_id,
                        "inning": inning,
                        "batting_team": "Mumbai Indians",
                        "bowling_team": "Chennai Super Kings",
                        "over": ov, "ball": ball,
                        "batsman": "Player_A",
                        "non_striker": "Player_B",
                        "bowler": "Bowler_X",
                        "batsman_runs": int(np.random.choice([0,1,2,4,6])),
                        "total_runs": int(np.random.choice([0,1,2,4,6])),
                        "wide_runs": 0,
                        "player_dismissed": None,
                    })
    return pd.DataFrame(rows)


class TestSeasonSummary:
    def test_returns_dataframe(self, sample_matches):
        result = season_summary(sample_matches)
        assert isinstance(result, pd.DataFrame)

    def test_seasons_present(self, sample_matches):
        result = season_summary(sample_matches)
        assert set(result["season"].tolist()) == {2022, 2023}

    def test_bat_win_pct_in_range(self, sample_matches):
        result = season_summary(sample_matches)
        assert (result["bat_win_pct"] >= 0).all()
        assert (result["bat_win_pct"] <= 100).all()

    def test_toss_win_pct_in_range(self, sample_matches):
        result = season_summary(sample_matches)
        assert (result["toss_win_match_pct"] >= 0).all()
        assert (result["toss_win_match_pct"] <= 100).all()


class TestTossImpact:
    def test_returns_dict(self, sample_matches):
        result = toss_impact_analysis(sample_matches)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_matches):
        result = toss_impact_analysis(sample_matches)
        assert "overall_toss_win_pct" in result
        assert "bat_decision_win_pct" in result
        assert "field_decision_win_pct" in result
        assert "season_trend" in result

    def test_pct_in_range(self, sample_matches):
        result = toss_impact_analysis(sample_matches)
        assert 0 <= result["overall_toss_win_pct"] <= 100
        if result["bat_decision_win_pct"] > 0:
            assert 0 <= result["bat_decision_win_pct"] <= 100


class TestPhaseAnalysis:
    def test_returns_dataframe(self, sample_deliveries):
        result = phase_analysis(sample_deliveries)
        assert isinstance(result, pd.DataFrame)

    def test_three_phases(self, sample_deliveries):
        result = phase_analysis(sample_deliveries)
        assert len(result) == 3

    def test_run_rate_positive(self, sample_deliveries):
        result = phase_analysis(sample_deliveries)
        assert (result["run_rate"] >= 0).all()

    def test_balls_positive(self, sample_deliveries):
        result = phase_analysis(sample_deliveries)
        assert (result["balls"] > 0).all()


class TestGetChampion:
    def test_returns_last_match_winner(self, sample_matches):
        season_df = sample_matches[sample_matches["season"] == 2022]
        champ = _get_champion(season_df)
        # Last match id=2 won by Chennai Super Kings
        assert champ == "Chennai Super Kings"

    def test_empty_df_returns_unknown(self):
        champ = _get_champion(pd.DataFrame())
        assert champ == "Unknown"
