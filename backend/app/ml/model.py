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
            "locations": "locations.json",
            "insights": "insights.json",
            "metrics": "metrics.json",
            "location_counts": "location_counts.json",
            "deep_model": "deep/embedding_mlp.pt",
            "deep_encoders": "deep/encoders.json",
        }

        for key, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            self.model_files[key] = {"path": path, "exists": os.path.exists(path)}
            if os.path.exists(path):
                try:
                    if filename.endswith(".pkl"):
                        self.models[key] = joblib.load(path)
                    mtime = os.path.getmtime(path)
                    self.model_files[key]["modified_at"] = datetime.fromtimestamp(
                        
                        mtime, tz=timezone.utc
                    ).isoformat()
                    logger.info("Loaded %s from %s", key, filename)
                except Exception as e:
                    logger.error("Error loading %s: %s", filename, e)
            else:
                logger.warning("File not found: %s", path)

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

        # Check and dynamically load the PyTorch deep learning model if available
        deep_model_path = os.path.join(self.model_dir, "deep", "embedding_mlp.pt")
        deep_encoders_path = os.path.join(self.model_dir, "deep", "encoders.json")
        if os.path.exists(deep_model_path) and os.path.exists(deep_encoders_path):
            try:
                import torch
                from ml_project.deep.model import EmbeddingMLP
                
                with open(deep_encoders_path, "r", encoding="utf-8") as f:
                    encoders = json.load(f)
                
                n_locations = len(encoders["locations"])
                deep_model = EmbeddingMLP(n_locations=n_locations)
                deep_model.load_state_dict(torch.load(deep_model_path, map_location="cpu"))
                deep_model.eval()
                
                self.models["deep_learning_mlp"] = {
                    "model": deep_model,
                    "encoders": encoders
                }
                logger.info("Loaded PyTorch EmbeddingMLP deep learning model")
            except Exception as e:
                logger.error("Error loading PyTorch deep learning model: %s", e)

    def check_for_updates(self):
        """Check if any model files or metadata on disk have been modified since they were loaded, and reload if so."""
        needs_reload = False
        for key, info in list(self.model_files.items()):
            path = info["path"]
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                if mtime_iso != info.get("modified_at"):
                    needs_reload = True
                    break
            elif info.get("exists"):
                # File was deleted
                needs_reload = True
                break
        if needs_reload:
            try:
                self.reload()
            except Exception as e:
                logger.error("Auto-reload of models failed: %s", e)

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
        self.check_for_updates()
        return self.locations

    def get_insights(self):
        self.check_for_updates()
        return self.insights

    def get_metrics(self):
        self.check_for_updates()
        return self.metrics

    def get_model_info(self):
        self.check_for_updates()
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
                if k not in ("default", "locations", "insights", "metrics", "location_counts", "deep_model", "deep_encoders")
            },
        }

    def predict_all(self, df):
        self.check_for_updates()
        results = {}
        for name, model in self.models.items():
            if name == "default" and "lgbm" in self.models:
                continue
            if name == "deep_learning_mlp":
                try:
                    import torch
                    deep_model = model["model"]
                    encoders = model["encoders"]
                    
                    loc = df["location"].iloc[0]
                    loc_list = encoders["locations"]
                    if loc in loc_list:
                        loc_idx = loc_list.index(loc)
                    elif "other" in loc_list:
                        loc_idx = loc_list.index("other")
                    else:
                        loc_idx = 0
                        
                    area_idx = int(df["area_type_enc"].iloc[0])
                    area_idx = max(0, min(3, area_idx))
                    
                    ready = float(df["is_ready_to_move"].iloc[0])
                    
                    numeric_cols = encoders["numeric_cols"]
                    numeric_vals = [float(df[col].iloc[0]) for col in numeric_cols]
                    mean = np.array(encoders["scaler_mean"])
                    scale = np.array(encoders["scaler_scale"])
                    numeric_scaled = (np.array(numeric_vals) - mean) / scale
                    
                    with torch.no_grad():
                        pred = deep_model(
                            torch.tensor(numeric_scaled, dtype=torch.float32).unsqueeze(0),
                            torch.tensor([loc_idx], dtype=torch.long),
                            torch.tensor([area_idx], dtype=torch.long),
                            torch.tensor([ready], dtype=torch.float32)
                        ).item()
                    results[name] = round(float(np.expm1(pred)), 2)
                except Exception as e:
                    logger.error("Failed running PyTorch inference: %s", e)
                continue
            pred = model.predict(df)[0]
            results[name] = round(float(np.expm1(pred)), 2)
        return results

    def health_status(self):
        self.check_for_updates()
        if all(k in self.models for k in REQUIRED_MODELS):
            return "ok"
        if self.models:
            return "degraded"
        return "model_missing"


manager = ModelManager()
