import json
import logging
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

from app.config import get_settings

# Backward compatibility for pickles saved as rasonix_ml.*
import ml_project.transformers as _transformers_module
import ml_project.preprocessing as _preprocessing_module
sys.modules.setdefault("rasonix_ml", _transformers_module)
sys.modules.setdefault("rasonix_ml.transformers", _transformers_module)
sys.modules.setdefault("rasonix_ml.preprocessing", _preprocessing_module)

logger = logging.getLogger(__name__)

REQUIRED_MODELS = ("ensemble", "lgbm", "xgb")


class ModelManager:
    def __init__(self):
        settings = get_settings()
        self.model_dir = settings["model_dir"]
        self.model_version = settings["model_version"]
        self.models = {}
        self.locations = []
        self.insights = {}
        self.metrics = {}
        self.location_counts = {}
        self.model_files = {}
        self.load_all_models()

    def reload(self):
        logger.info("Reloading all models and metadata...")
        self.models = {}
        self.locations = []
        self.insights = {}
        self.metrics = {}
        self.location_counts = {}
        self.model_files = {}
        self.load_all_models()

    def load_all_models(self):
        model_files = {
            "ensemble": "stacking_model.pkl",
            "lgbm": "lgbm_model.pkl",
            "xgb": "xgb_model.pkl",
            "default": "bangalore_house_price_model.pkl",
        }

        for key, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            self.model_files[key] = {"path": path, "exists": os.path.exists(path)}
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                    mtime = os.path.getmtime(path)
                    self.model_files[key]["modified_at"] = datetime.fromtimestamp(
                        mtime, tz=timezone.utc
                    ).isoformat()
                    logger.info("Loaded %s model from %s", key, filename)
                except Exception as e:
                    logger.error("Error loading %s: %s", filename, e)
            else:
                logger.warning("Model file not found: %s", path)

        loc_path = os.path.join(self.model_dir, "locations.json")
        if os.path.exists(loc_path):
            with open(loc_path, encoding="utf-8") as f:
                self.locations = json.load(f)

        ins_path = os.path.join(self.model_dir, "insights.json")
        if os.path.exists(ins_path):
            with open(ins_path, encoding="utf-8") as f:
                self.insights = json.load(f)

        metrics_path = os.path.join(self.model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                self.metrics = json.load(f)

        counts_path = os.path.join(self.model_dir, "location_counts.json")
        if os.path.exists(counts_path):
            with open(counts_path, encoding="utf-8") as f:
                self.location_counts = json.load(f)

    def validate_startup(self):
        settings = get_settings()
        if not settings["require_models"]:
            return
        missing = [k for k in REQUIRED_MODELS if k not in self.models]
        if missing:
            raise RuntimeError(
                f"Required models not loaded: {missing}. "
                f"Run 'python ML/train.py' from the repo root."
            )

    def get_locations(self):
        return self.locations

    def get_insights(self):
        return self.insights

    def get_metrics(self):
        return self.metrics

    def get_model_info(self):
        loaded = [k for k in REQUIRED_MODELS if k in self.models]
        return {
            "model_version": self.model_version,
            "loaded_models": loaded,
            "all_models": list(self.models.keys()),
            "locations_count": len(self.locations),
            "metrics": self.metrics,
            "model_files": {
                k: {
                    "exists": v.get("exists", False),
                    "modified_at": v.get("modified_at"),
                }
                for k, v in self.model_files.items()
                if k != "default"
            },
        }

    def predict_all(self, df):
        results = {}
        for name, model in self.models.items():
            if name == "default" and "lgbm" in self.models:
                continue
            pred = model.predict(df)[0]
            results[name] = round(float(np.expm1(pred)), 2)
        return results

    def health_status(self):
        if all(k in self.models for k in REQUIRED_MODELS):
            return "ok"
        if self.models:
            return "degraded"
        return "model_missing"


manager = ModelManager()
