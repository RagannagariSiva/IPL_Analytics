"""
tests/test_predictor.py
========================
Tests for the ML prediction model.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.predictor import build_model, FEATURE_COLS


class TestModelBuilding:
    def test_build_model_returns_estimator(self):
        model = build_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_fits_simple_data(self):
        model = build_model()
        X = pd.DataFrame(np.random.rand(100, len(FEATURE_COLS)), columns=FEATURE_COLS)
        y = pd.Series(np.random.randint(0, 2, 100))
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_1(self):
        model = build_model()
        X = pd.DataFrame(np.random.rand(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
        y = pd.Series(np.random.randint(0, 2, 50))
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (50, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_range(self):
        model = build_model()
        X = pd.DataFrame(np.random.rand(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
        y = pd.Series(np.random.randint(0, 2, 50))
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert (probas >= 0).all()
        assert (probas <= 1).all()
