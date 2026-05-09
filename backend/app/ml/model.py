import os
import joblib
import numpy as np
import sys
from .encoder import LocationTargetEncoder

# Aliasing to fix 'No module named src' during unpickling
# This maps the training script's module path to the backend's module path
import app.ml.encoder as encoder_module
sys.modules['src'] = encoder_module
sys.modules['src.transformers'] = encoder_module

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        self.load_all_models()

    def load_all_models(self):
        # We try to load the three distinct models
        model_files = {
            "ensemble": "stacking_model.pkl",
            "lgbm": "lgbm_model.pkl",
            "xgb": "xgb_model.pkl",
            "default": "bangalore_house_price_model.pkl"
        }

        for key, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                    print(f"Successfully loaded {key} model from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Notice: {filename} not found at {path}")

        # Load metadata
        self.locations = []
        self.insights = {}
        
        loc_path = os.path.join(self.model_dir, "locations.json")
        if os.path.exists(loc_path):
            import json
            with open(loc_path, 'r') as f:
                self.locations = json.load(f)

        ins_path = os.path.join(self.model_dir, "insights.json")
        if os.path.exists(ins_path):
            import json
            with open(ins_path, 'r') as f:
                self.insights = json.load(f)

    def get_locations(self):
        return self.locations

    def get_insights(self):
        return self.insights

    def predict_all(self, df):
        results = {}
        # Use whatever models we successfully loaded
        for name, model in self.models.items():
            if name == "default" and "lgbm" in self.models:
                continue # Skip default if we already have lgbm
            
            pred = model.predict(df)[0]
            results[name] = round(float(np.expm1(pred)), 2)
        
        return results

# Singleton instance
manager = ModelManager()
