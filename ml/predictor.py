"""
ml/predictor.py
================
IPL Match Winner Prediction  Ensemble ML Model.
Uses XGBoost + LightGBM + RandomForest stacking with Logistic Regression meta-learner.
Saves/loads model artifacts using joblib.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, log_loss,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

#  Feature columns 
FEATURE_COLS = [
    "toss_winner_is_team1","toss_decision_bat",
    "t1_win_rate","t2_win_rate",
    "t1_strength","t2_strength",
    "t1_recent_form","t2_recent_form",
    "t1_toss_wr","t2_toss_wr",
    "h2h_win_rate_t1","venue_bat_win_rate",
    "win_rate_diff","strength_diff","form_diff",
    "season",
]

MODEL_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "./data/models")
MODEL_PATH = os.path.join(MODEL_DIR, "winner_model.joblib")
META_PATH  = os.path.join(MODEL_DIR, "winner_meta.json")


#  Build ensemble 

def _build_estimators():
    estimators = []
    if _HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            verbosity=0,
        )))
    if _HAS_LGB:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )))
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )))
    return estimators


def build_model():
    estimators = _build_estimators()
    if len(estimators) >= 2:
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
            cv=5, stack_method="predict_proba", n_jobs=-1,
        )
    return estimators[0][1]   # fallback: plain RF


#  Train 

def train_model(ml_df: pd.DataFrame) -> Dict:
    """
    Train the match winner model from the processed ML dataset.
    Saves artifacts to MODEL_DIR.
    Returns evaluation metrics dict.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    available = [c for c in FEATURE_COLS if c in ml_df.columns]
    missing   = [c for c in FEATURE_COLS if c not in ml_df.columns]
    if missing:
        logger.warning(f"Missing features (using 0): {missing}")
        for c in missing:
            ml_df[c] = 0.0

    X = ml_df[FEATURE_COLS].copy()
    # normalise season: "2009/10" -> 2009
    if "season" in X.columns:
        X["season"] = X["season"].apply(
            lambda s: int(str(s).split("/")[0]) if pd.notna(s) else 2008)
    X = X.fillna(0.0).astype(float)
    y = ml_df["target"].astype(int)

    if len(X) < 50:
        raise ValueError(f"Too few samples to train ({len(X)}). Need at least 50 matches.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info(f"Training on {len(X_train)} samples | test {len(X_test)} samples")
    t0 = time.perf_counter()

    model = build_model()

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    logger.info(f"CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    model.fit(X_train, y_train)
    elapsed = round(time.perf_counter() - t0, 2)

    # Evaluate
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, preds)
    auc    = roc_auc_score(y_test, probas)
    ll     = log_loss(y_test, probas)
    cm     = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall    = tp / (tp+fn) if (tp+fn) > 0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    elif hasattr(model, "final_estimator_") and hasattr(model.final_estimator_, "coef_"):
        fi = {}
    else:
        fi = {}

    metrics = {
        "accuracy":        round(acc,  4),
        "roc_auc":         round(auc,  4),
        "log_loss":        round(ll,   4),
        "precision":       round(precision, 4),
        "recall":          round(recall,    4),
        "f1_score":        round(f1,        4),
        "cv_auc_mean":     round(float(cv_auc.mean()), 4),
        "cv_auc_std":      round(float(cv_auc.std()),  4),
        "train_samples":   len(X_train),
        "test_samples":    len(X_test),
        "training_time_s": elapsed,
        "feature_importance": fi,
    }

    # Save artifacts
    joblib.dump(model, MODEL_PATH, compress=3)
    with open(META_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Model saved  {MODEL_PATH}  |  AUC={auc:.4f}  acc={acc:.4f}")
    return metrics


#  Load 

def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Run: python train.py   first."
        )
    return joblib.load(MODEL_PATH)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def get_model_meta() -> Dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


#  Predict 

def predict_match(
    model,
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
    team_stats: pd.DataFrame,
    h2h_stats: pd.DataFrame,
    venue_stats: pd.DataFrame,
    season: int = 2024,
) -> Dict:
    """
    Predict match winner given match setup.
    Returns full prediction dict with probabilities.
    """
    row = _build_input_row(
        team1, team2, toss_winner, toss_decision, venue,
        team_stats, h2h_stats, venue_stats, season,
    )
    X = pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)

    proba = model.predict_proba(X)[0]   # [prob_team2_wins, prob_team1_wins]
    t1_prob = float(proba[1])
    t2_prob = float(proba[0])

    winner = team1 if t1_prob >= 0.5 else team2
    confidence = max(t1_prob, t2_prob)

    return {
        "predicted_winner":    winner,
        "team1":               team1,
        "team2":               team2,
        "team1_win_probability": round(t1_prob, 4),
        "team2_win_probability": round(t2_prob, 4),
        "confidence":           round(confidence, 4),
        "toss_winner":          toss_winner,
        "toss_decision":        toss_decision,
        "venue":                venue,
        "season":               season,
    }


def _build_input_row(
    team1, team2, toss_winner, toss_decision, venue,
    team_stats, h2h_stats, venue_stats, season,
) -> Dict:
    ts = team_stats.set_index("team") if "team" in team_stats.columns else team_stats

    def tstat(t, c, dv=0.5):
        try:    return float(ts.loc[t, c])
        except: return dv

    vs = venue_stats.set_index("venue") if "venue" in venue_stats.columns else venue_stats
    def vstat(v, c, dv=0.5):
        try:    return float(vs.loc[v, c])
        except: return dv

    def h2h(t1, t2):
        r = h2h_stats[
            ((h2h_stats["team1"]==t1)&(h2h_stats["team2"]==t2)) |
            ((h2h_stats["team1"]==t2)&(h2h_stats["team2"]==t1))
        ]
        if len(r)==0: return 0.5
        row=r.iloc[0]
        return float(row["team1_win_rate"]) if row["team1"]==t1 else 1-float(row["team1_win_rate"])

    t1wr  = tstat(team1,"win_rate")
    t2wr  = tstat(team2,"win_rate")
    t1str = tstat(team1,"team_strength_index")
    t2str = tstat(team2,"team_strength_index")
    t1rf  = tstat(team1,"recent_form")
    t2rf  = tstat(team2,"recent_form")

    return {
        "season":               season,
        "toss_winner_is_team1": int(toss_winner == team1),
        "toss_decision_bat":    int(toss_decision == "bat"),
        "t1_win_rate":          t1wr,
        "t2_win_rate":          t2wr,
        "t1_strength":          t1str,
        "t2_strength":          t2str,
        "t1_recent_form":       t1rf,
        "t2_recent_form":       t2rf,
        "t1_toss_wr":           tstat(team1,"toss_win_rate"),
        "t2_toss_wr":           tstat(team2,"toss_win_rate"),
        "h2h_win_rate_t1":      h2h(team1,team2),
        "venue_bat_win_rate":   vstat(venue,"bat_first_win_rate"),
        "win_rate_diff":        t1wr  - t2wr,
        "strength_diff":        t1str - t2str,
        "form_diff":            t1rf  - t2rf,
    }
