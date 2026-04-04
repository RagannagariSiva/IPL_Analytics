"""
api.py
=======
FastAPI REST API backend for IPL Analytics Platform.
Provides endpoints for:
  - Match predictions
  - Player analytics
  - Team stats
  - Head-to-head data
  - Live win probability
  - Prediction history (CRUD)

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import time
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Dict, Any
from uuid import UUID

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from loguru import logger

from database import get_db, init_db, PredictionLog, TeamStats, Player, Match, VenueStats
from ml.predictor import load_model, model_exists, get_model_meta, predict_match
from ml.features import live_win_probability

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IPL Analytics Platform API",
    description="IPL match prediction and analytics REST API",
    docs_url=None,   # Serve custom docs
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Strip version from OpenAPI schema ────────────────────────────────────────

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="IPL Analytics Platform API",
        version="",
        description="IPL match prediction and analytics REST API",
        routes=app.routes,
    )
    schema["info"].pop("version", None)
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi


# ── Custom Swagger UI (hides version / OAS badge) ────────────────────────────

@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    html = """<!DOCTYPE html>
<html>
<head>
    <title>IPL Analytics Platform API</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        .swagger-ui .info .title small,
        .swagger-ui .info .title small.version-stamp,
        .swagger-ui .info hgroup.main .version { display: none !important; }
        .swagger-ui .info { margin: 20px 0; }
    </style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
window.onload = function () {
    SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        layout: "BaseLayout",
        deepLinking: true,
        defaultModelsExpandDepth: -1,
    });
};
</script>
</body>
</html>"""
    return HTMLResponse(html)


# ── Cached data loaders ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_team_stats() -> pd.DataFrame:
    path = "data/processed/team_stats.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_h2h_stats() -> pd.DataFrame:
    path = "data/processed/h2h_stats.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_venue_stats() -> pd.DataFrame:
    path = "data/processed/venue_stats.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_player_impact() -> pd.DataFrame:
    path = "data/processed/player_impact.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_batting_stats() -> pd.DataFrame:
    path = "data/processed/batting_stats.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_bowling_stats() -> pd.DataFrame:
    path = "data/processed/bowling_stats.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@lru_cache(maxsize=1)
def _get_model():
    if not model_exists():
        return None
    return load_model()


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("IPL Analytics API started")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class MatchPredictRequest(BaseModel):
    team1:         str = Field(..., example="Mumbai Indians")
    team2:         str = Field(..., example="Chennai Super Kings")
    toss_winner:   str = Field(..., example="Mumbai Indians")
    toss_decision: str = Field(..., example="bat", description="bat or field")
    venue:         str = Field(..., example="Wankhede Stadium")
    season:        int = Field(2024, example=2024)


class LiveWinProbRequest(BaseModel):
    runs_scored:    int           = Field(..., ge=0)
    wickets_fallen: int           = Field(..., ge=0, le=10)
    balls_bowled:   int           = Field(..., ge=0, le=120)
    target:         Optional[int] = Field(None, ge=1)
    inning:         int           = Field(1, ge=1, le=2)


class PredictionResponse(BaseModel):
    predicted_winner:      str
    team1:                 str
    team2:                 str
    team1_win_probability: float
    team2_win_probability: float
    confidence:            float
    toss_winner:           str
    toss_decision:         str
    venue:                 str
    season:                int


class PredictionUpdateRequest(BaseModel):
    notes: Optional[str] = Field(None, description="Analyst notes for this prediction")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"message": "IPL Analytics API is running", "status": "healthy"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":      "healthy",
        "model_ready": model_exists(),
        "timestamp":   datetime.utcnow().isoformat(),
    }


# ── Model ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/model/info", tags=["Model"])
def model_info():
    """Get trained model metadata and evaluation metrics."""
    meta = get_model_meta()
    if not meta:
        raise HTTPException(404, "Model not trained yet. Run: python train.py")
    return meta


# ── Prediction ────────────────────────────────────────────────────────────────

@app.post("/api/v1/predict/match", response_model=PredictionResponse, tags=["Prediction"])
def predict_match_winner(req: MatchPredictRequest, db: Session = Depends(get_db)):
    """Predict IPL match winner. Returns predicted winner with win probabilities."""
    model = _get_model()
    if model is None:
        raise HTTPException(503, "Model not available. Run: python train.py")

    t0     = time.perf_counter()
    result = predict_match(
        model=model,
        team1=req.team1, team2=req.team2,
        toss_winner=req.toss_winner, toss_decision=req.toss_decision,
        venue=req.venue,
        team_stats=_get_team_stats(), h2h_stats=_get_h2h_stats(),
        venue_stats=_get_venue_stats(), season=req.season,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    log = PredictionLog(
        prediction_type="match_winner",
        team1=req.team1, team2=req.team2,
        predicted_winner=result["predicted_winner"],
        team1_win_prob=result["team1_win_probability"],
        team2_win_prob=result["team2_win_probability"],
        confidence=result["confidence"],
        input_data=req.dict(),
    )
    db.add(log)
    db.commit()

    result["response_time_ms"] = elapsed_ms
    return result


@app.post("/api/v1/predict/live-win-probability", tags=["Prediction"])
def live_probability(req: LiveWinProbRequest):
    """Real-time win probability calculator for in-progress matches."""
    result = live_win_probability(
        runs_scored=req.runs_scored,
        wickets_fallen=req.wickets_fallen,
        balls_bowled=req.balls_bowled,
        target=req.target,
        inning=req.inning,
    )
    return {
        **result,
        "inning":         req.inning,
        "runs_scored":    req.runs_scored,
        "wickets_fallen": req.wickets_fallen,
        "balls_bowled":   req.balls_bowled,
        "overs_bowled":   round(req.balls_bowled / 6, 1),
    }


# ── Teams ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/teams", tags=["Teams"])
def get_teams():
    """List all IPL teams with strength indices."""
    df = _get_team_stats()
    if df.empty:
        raise HTTPException(404, "Team stats not available. Run train.py first.")
    return df.to_dict("records")


@app.get("/api/v1/teams/{team_name}", tags=["Teams"])
def get_team(team_name: str):
    """Get detailed stats for a specific team."""
    df  = _get_team_stats()
    row = df[df["team"].str.lower() == team_name.lower()]
    if row.empty:
        raise HTTPException(404, f"Team '{team_name}' not found.")
    return row.iloc[0].to_dict()


@app.put("/api/v1/teams/{team_name}", tags=["Teams"])
def update_team(team_name: str, payload: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Update editable fields for a specific team record in the database
    (e.g. custom strength override, notes).
    """
    team = db.query(TeamStats).filter(
        TeamStats.team.ilike(team_name)
    ).first()
    if team is None:
        raise HTTPException(404, f"Team '{team_name}' not found in database.")

    allowed = {"custom_strength_override", "notes"}
    updated = {}
    for key, val in payload.items():
        if key in allowed and hasattr(team, key):
            setattr(team, key, val)
            updated[key] = val

    db.commit()
    db.refresh(team)
    return {"team": team_name, "updated_fields": updated}


@app.delete("/api/v1/teams/{team_name}/cache", tags=["Teams"], status_code=status.HTTP_204_NO_CONTENT)
def invalidate_team_cache(team_name: str):
    """
    Invalidate the in-memory cache for team stats so the next request
    reloads from disk after a retraining run.
    """
    _get_team_stats.cache_clear()
    _get_h2h_stats.cache_clear()
    return


@app.get("/api/v1/teams/h2h/{team1}/{team2}", tags=["Teams"])
def head_to_head(team1: str, team2: str):
    """Head-to-head record between two teams."""
    h2h = _get_h2h_stats()
    row = h2h[
        ((h2h["team1"].str.lower() == team1.lower()) & (h2h["team2"].str.lower() == team2.lower())) |
        ((h2h["team1"].str.lower() == team2.lower()) & (h2h["team2"].str.lower() == team1.lower()))
    ]
    if row.empty:
        raise HTTPException(404, "No H2H record found.")
    return row.iloc[0].to_dict()


# ── Players ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/players/top-batsmen", tags=["Players"])
def top_batsmen(limit: int = Query(20, ge=1, le=100)):
    """Top IPL batsmen by total runs."""
    df = _get_batting_stats()
    if df.empty:
        raise HTTPException(404, "Player data not available. Run train.py first.")
    cols      = ["batsman", "total_runs", "innings", "batting_avg", "strike_rate",
                 "fours", "sixes", "batting_impact"]
    available = [c for c in cols if c in df.columns]
    return df[available].head(limit).to_dict("records")


@app.get("/api/v1/players/top-bowlers", tags=["Players"])
def top_bowlers(limit: int = Query(20, ge=1, le=100)):
    """Top IPL bowlers by wickets."""
    df = _get_bowling_stats()
    if df.empty:
        raise HTTPException(404, "Player data not available. Run train.py first.")
    cols      = ["bowler", "wickets", "matches", "economy_rate", "bowling_avg",
                 "bowling_sr", "dot_ball_pct", "bowling_impact"]
    available = [c for c in cols if c in df.columns]
    return df[available].head(limit).to_dict("records")


@app.get("/api/v1/players/impact-scores", tags=["Players"])
def player_impact_scores(limit: int = Query(30, ge=1, le=200)):
    """Player Impact Score leaderboard."""
    df = _get_player_impact()
    if df.empty:
        raise HTTPException(404, "Player impact data not available.")
    cols      = ["player", "player_impact_score", "role", "batting_impact",
                 "bowling_impact", "total_runs", "wickets"]
    available = [c for c in cols if c in df.columns]
    return df[available].head(limit).to_dict("records")


# ── Venues ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/venues", tags=["Venues"])
def get_venues(limit: int = Query(20, ge=1, le=100)):
    """Venue statistics."""
    df = _get_venue_stats()
    if df.empty:
        raise HTTPException(404, "Venue data not available.")
    return df.head(limit).to_dict("records")


# ── Prediction history (full CRUD) ────────────────────────────────────────────

@app.get("/api/v1/predictions/history", tags=["Prediction"])
def prediction_history(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """List recent prediction logs."""
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_log_to_dict(log) for log in logs]


@app.get("/api/v1/predictions/history/{prediction_id}", tags=["Prediction"])
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    """Get a single prediction record by ID."""
    log = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if log is None:
        raise HTTPException(404, f"Prediction '{prediction_id}' not found.")
    return _log_to_dict(log)


@app.put("/api/v1/predictions/history/{prediction_id}", tags=["Prediction"])
def update_prediction(
    prediction_id: str,
    req: PredictionUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update analyst notes on an existing prediction record."""
    log = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if log is None:
        raise HTTPException(404, f"Prediction '{prediction_id}' not found.")
    if req.notes is not None:
        log.notes = req.notes
    db.commit()
    db.refresh(log)
    return _log_to_dict(log)


@app.delete("/api/v1/predictions/history/{prediction_id}", tags=["Prediction"],
            status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(prediction_id: str, db: Session = Depends(get_db)):
    """Delete a prediction record from history."""
    log = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if log is None:
        raise HTTPException(404, f"Prediction '{prediction_id}' not found.")
    db.delete(log)
    db.commit()
    return


# ── Matches ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/matches/seasons", tags=["Matches"])
def get_seasons(db: Session = Depends(get_db)):
    """Available seasons in the database."""
    from sqlalchemy import distinct
    seasons = db.query(distinct(Match.season)).order_by(Match.season).all()
    return [s[0] for s in seasons]


@app.get("/api/v1/matches/by-season/{season}", tags=["Matches"])
def matches_by_season(season: int, db: Session = Depends(get_db)):
    """All matches in a given season."""
    matches = db.query(Match).filter(Match.season == season).all()
    if not matches:
        raise HTTPException(404, f"No matches found for season {season}")
    return [
        {
            "id": m.id, "date": m.date, "venue": m.venue,
            "team1": m.team1, "team2": m.team2,
            "toss_winner": m.toss_winner, "toss_decision": m.toss_decision,
            "winner": m.winner,
            "player_of_match": m.player_of_match,
        }
        for m in matches
    ]


@app.put("/api/v1/matches/{match_id}", tags=["Matches"])
def update_match(match_id: int, payload: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Update editable fields on a match record (e.g. correct a venue or winner
    after a data quality review).
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(404, f"Match {match_id} not found.")

    allowed = {"venue", "winner", "player_of_match", "toss_winner", "toss_decision"}
    updated = {}
    for key, val in payload.items():
        if key in allowed and hasattr(match, key):
            setattr(match, key, val)
            updated[key] = val

    db.commit()
    db.refresh(match)
    return {"match_id": match_id, "updated_fields": updated}


@app.delete("/api/v1/matches/{match_id}", tags=["Matches"],
            status_code=status.HTTP_204_NO_CONTENT)
def delete_match(match_id: int, db: Session = Depends(get_db)):
    """
    Remove a match record from the database
    (use when a match is marked 'no result' or data is erroneous).
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(404, f"Match {match_id} not found.")
    db.delete(match)
    db.commit()
    return


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_to_dict(log: PredictionLog) -> Dict:
    return {
        "id":              str(log.id),
        "team1":           log.team1,
        "team2":           log.team2,
        "predicted_winner": log.predicted_winner,
        "team1_win_prob":  log.team1_win_prob,
        "team2_win_prob":  log.team2_win_prob,
        "confidence":      log.confidence,
        "notes":           getattr(log, "notes", None),
        "created_at":      log.created_at.isoformat() if log.created_at else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
