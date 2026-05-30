import os
import sys

import pytest

# Repo root and backend on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

# Allow API startup without models for unit tests that mock predictions
os.environ.setdefault("REQUIRE_MODELS", "false")


@pytest.fixture
def model_dir():
    return os.path.join(ROOT, "backend", "models")


@pytest.fixture
def models_available(model_dir):
    required = ["stacking_model.pkl", "lgbm_model.pkl", "xgb_model.pkl"]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required)
