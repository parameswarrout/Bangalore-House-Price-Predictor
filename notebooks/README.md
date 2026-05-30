# Notebooks

## Canonical modeling path

Production training and serving use **V2** (`ML/train.py` + `ml_project` package):

- Target encoding for locations
- Interaction features inside sklearn pipelines
- Stacking ensemble (XGBoost + LightGBM + CatBoost when installed)
- Unified preprocessing: rare-location bucketing, p99 price cap, extra features

## Notebook versions

| Notebook | Status | Notes |
|----------|--------|-------|
| `Bangalore_House_Price_Prediction_V1.ipynb` | Deprecated | One-hot encoding (~243 features); research only |
| `Bangalore_House_Price_Prediction_V2.ipynb` | Reference | Aligns with V2 pipeline; includes SHAP/Optuna experiments |

Run training for deployable artifacts:

```powershell
python ML/train.py
# Optional: hyperparameter tuning, SHAP, PyTorch deep learning (research)
python ML/train.py --tune --explain
python ML/train.py --deep

# Reproducible EDA
python scripts/eda.py
```

V2 notebook imports from `ml_project` and includes a **Phase 3: Deep Learning** section comparing Embedding MLP / TabNet to the ensemble. See `metrics.json` → `promotion_criteria` for API promotion rules.
