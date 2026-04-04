"""
tests/test_api.py
==================
Integration tests for the FastAPI backend.
Uses TestClient  no running server required.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only import if fastapi + httpx available
try:
    from fastapi.testclient import TestClient
    from api import app
    HAS_API = True
except ImportError:
    HAS_API = False


@pytest.mark.skipif(not HAS_API, reason="FastAPI not available")
class TestHealthEndpoints:
    def setup_method(self):
        self.client = TestClient(app)

    def test_root_returns_200(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert "IPL" in r.json()["message"]

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_ready" in data

    def test_docs_accessible(self):
        r = self.client.get("/docs")
        assert r.status_code == 200


@pytest.mark.skipif(not HAS_API, reason="FastAPI not available")
class TestLiveWinProbability:
    def setup_method(self):
        self.client = TestClient(app)

    def test_inning1_no_target(self):
        r = self.client.post("/api/v1/predict/live-win-probability", json={
            "runs_scored": 80,
            "wickets_fallen": 3,
            "balls_bowled": 60,
            "inning": 1
        })
        assert r.status_code == 200
        data = r.json()
        assert "batting_team" in data
        assert "bowling_team" in data
        assert abs(data["batting_team"] + data["bowling_team"] - 1.0) < 0.01

    def test_inning2_with_target(self):
        r = self.client.post("/api/v1/predict/live-win-probability", json={
            "runs_scored": 100,
            "wickets_fallen": 4,
            "balls_bowled": 72,
            "target": 160,
            "inning": 2
        })
        assert r.status_code == 200
        data = r.json()
        assert 0 <= data["batting_team"] <= 1

    def test_invalid_wickets_rejected(self):
        r = self.client.post("/api/v1/predict/live-win-probability", json={
            "runs_scored": 80,
            "wickets_fallen": 15,  # invalid > 10
            "balls_bowled": 60,
            "inning": 1
        })
        assert r.status_code == 422  # validation error

    def test_negative_runs_rejected(self):
        r = self.client.post("/api/v1/predict/live-win-probability", json={
            "runs_scored": -5,  # invalid
            "wickets_fallen": 3,
            "balls_bowled": 60,
            "inning": 1
        })
        assert r.status_code == 422


@pytest.mark.skipif(not HAS_API, reason="FastAPI not available")
class TestPredictionEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_predict_returns_expected_keys(self):
        """If model is ready, verify response structure."""
        r = self.client.post("/api/v1/predict/match", json={
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "toss_winner": "Mumbai Indians",
            "toss_decision": "bat",
            "venue": "Wankhede Stadium",
            "season": 2024
        })
        # Either model works (200) or not trained (503)
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "predicted_winner" in data
            assert "team1_win_probability" in data
            assert "team2_win_probability" in data
            assert "confidence" in data
            total = data["team1_win_probability"] + data["team2_win_probability"]
            assert abs(total - 1.0) < 0.01

    def test_prediction_history_accessible(self):
        r = self.client.get("/api/v1/predictions/history")
        assert r.status_code in (200, 404)
