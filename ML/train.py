import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml_project.metrics import regression_metrics
from ml_project.preprocessing import (
    MODEL_FEATURES,
    build_location_counts,
    load_and_prepare_training_frame,
)
from ml_project.transformers import InteractionFeatureTransformer, LocationTargetEncoder

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def _make_xgb_pipeline(**xgb_kwargs):
    defaults = dict(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    defaults.update(xgb_kwargs)
    pipe = Pipeline([
        ("interactions", InteractionFeatureTransformer()),
        ("encoder", LocationTargetEncoder()),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(**defaults)),
    ])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def _make_lgbm_pipeline(**lgbm_kwargs):
    defaults = dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    defaults.update(lgbm_kwargs)
    pipe = Pipeline([
        ("interactions", InteractionFeatureTransformer()),
        ("encoder", LocationTargetEncoder()),
        ("scaler", StandardScaler()),
        ("model", LGBMRegressor(**defaults)),
    ])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def _make_catboost_pipeline(**catboost_kwargs):
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed")
    defaults = dict(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )
    defaults.update(catboost_kwargs)
    pipe = Pipeline([
        ("interactions", InteractionFeatureTransformer()),
        ("encoder", LocationTargetEncoder()),
        ("scaler", StandardScaler()),
        ("model", CatBoostRegressor(**defaults)),
    ])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def _run_optuna_tuning(X_train, y_train, model_dir: str) -> dict:
    import optuna

    tuning = {"xgb": {}, "lgbm": {}, "catboost": {}}

    def xgb_objective(trial):
        pipe = _make_xgb_pipeline(
            n_estimators=trial.suggest_int("n_estimators", 50, 250),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
        )
        return cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()

    def lgbm_objective(trial):
        pipe = _make_lgbm_pipeline(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 64),
        )
        return cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()

    print("Optuna: tuning XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=15, show_progress_bar=False)
    tuning["xgb"] = study_xgb.best_params
    print(f"  XGB best: {tuning['xgb']}")

    print("Optuna: tuning LightGBM...")
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(lgbm_objective, n_trials=15, show_progress_bar=False)
    tuning["lgbm"] = study_lgbm.best_params
    print(f"  LGBM best: {tuning['lgbm']}")

    if HAS_CATBOOST:

        def catboost_objective(trial):
            pipe = _make_catboost_pipeline(
                iterations=trial.suggest_int("iterations", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 10),
            )
            return cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()

        print("Optuna: tuning CatBoost...")
        study_cb = optuna.create_study(direction="maximize")
        study_cb.optimize(catboost_objective, n_trials=15, show_progress_bar=False)
        tuning["catboost"] = study_cb.best_params
        print(f"  CatBoost best: {tuning['catboost']}")

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "tuning.json"), "w", encoding="utf-8") as f:
        json.dump(tuning, f, indent=2)
    return tuning


import sys
import logging

class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.log_level, line.rstrip())
            
    def flush(self):
        pass


def setup_logging(dev_mode: bool = True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    orig_stdout = sys.stdout
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if dev_mode else logging.INFO)
    root_logger.handlers = []
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    is_subprocess = os.environ.get("SUBPROCESS_RUN", "false").lower() == "true"
    
    if not is_subprocess:
        log_file = os.path.join(logs_dir, "ml_train.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    console_handler = logging.StreamHandler(orig_stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.ERROR)


def _score_model(model, X_test, y_test):
    return regression_metrics(y_test, model.predict(X_test))


def run_training(tune: bool = False, explain: bool = False, deep: bool = False, custom_data_path: str = None, dev: bool = True):
    setup_logging(dev)
    print("Starting Property Price Model Training (Modular Structure)...")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "bengaluru_house_prices.csv")
    model_dir = os.path.join(base_dir, "backend", "models")

    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    df = load_and_prepare_training_frame(data_path, custom_csv_path=custom_data_path)
    print(f"Prepared {len(df)} samples after unified preprocessing.")

    X = df[MODEL_FEATURES]
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} samples, holdout {len(X_test)}...")

    tuning = {}
    if tune:
        try:
            tuning = _run_optuna_tuning(X_train, y_train, model_dir)
        except ImportError:
            print("optuna not installed; skipping hyperparameter tuning")

    xgb_params = tuning.get("xgb", {})
    lgbm_params = tuning.get("lgbm", {})
    catboost_params = tuning.get("catboost", {})

    xgb_pipe = _make_xgb_pipeline(**xgb_params)
    lgbm_pipe = _make_lgbm_pipeline(**lgbm_params)

    estimators = [("xgb", xgb_pipe), ("lgbm", lgbm_pipe)]
    if HAS_CATBOOST:
        estimators.append(("catboost", _make_catboost_pipeline(**catboost_params)))
    else:
        print("CatBoost not installed; stacking uses XGB + LGBM only.")

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )

    print("Fitting models...")
    xgb_pipe.fit(X_train, y_train)
    lgbm_pipe.fit(X_train, y_train)
    stacking_model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(xgb_pipe, os.path.join(model_dir, "xgb_model.pkl"))
    joblib.dump(lgbm_pipe, os.path.join(model_dir, "lgbm_model.pkl"))
    joblib.dump(stacking_model, os.path.join(model_dir, "stacking_model.pkl"))
    joblib.dump(lgbm_pipe, os.path.join(model_dir, "bangalore_house_price_model.pkl"))

    if HAS_CATBOOST:
        catboost_pipe = estimators[-1][1]
        catboost_pipe.fit(X_train, y_train)
        joblib.dump(catboost_pipe, os.path.join(model_dir, "catboost_model.pkl"))

    location_counts = build_location_counts(df)
    with open(os.path.join(model_dir, "location_counts.json"), "w", encoding="utf-8") as f:
        json.dump(location_counts, f, indent=2)

    locations = sorted(df["location"].unique().tolist())

    price_hist, bins = np.histogram(df["price"], bins=10, range=(0, 1000))
    price_dist = [
        {"range": f"{int(bins[i])}-{int(bins[i+1])}L", "count": int(price_hist[i])}
        for i in range(len(price_hist))
    ]

    loc_stats = df.groupby("location")["price"].agg(["mean", "count"]).reset_index()
    top_locs = loc_stats[loc_stats["count"] >= 5].sort_values(by="mean", ascending=False).head(8)
    loc_data = [
        {"location": row["location"], "avg_price": round(row["mean"], 2)}
        for _, row in top_locs.iterrows()
    ]

    xgb_metrics = _score_model(xgb_pipe, X_test, y_test)
    lgbm_metrics = _score_model(lgbm_pipe, X_test, y_test)
    stack_metrics = _score_model(stacking_model, X_test, y_test)

    cv_summary = {}
    for name, pipe in [("xgb", xgb_pipe), ("lgbm", lgbm_pipe), ("ensemble", stacking_model)]:
        oof = cross_val_predict(pipe, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42))
        cv_summary[name] = regression_metrics(y_train, oof)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "2.1.0",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": {
            "xgb": xgb_metrics,
            "lgbm": lgbm_metrics,
            "ensemble": stack_metrics,
        },
        "cv_5fold_train": cv_summary,
        "promotion_criteria": {
            "ensemble_mae_lakhs_baseline": 18.86,
            "deep_learning_api_promotion": (
                "Promote DL to API only if holdout MAE < ensemble MAE "
                "and 0-100L bin MAPE does not regress."
            ),
        },
    }

    insights = {
        "price_distribution": price_dist,
        "location_insights": loc_data,
        "model_performance": [
            {"name": "XGBoost", "r2": xgb_metrics["r2"]},
            {"name": "LightGBM", "r2": lgbm_metrics["r2"]},
            {"name": "Stacking", "r2": stack_metrics["r2"]},
        ],
    }

    if explain:
        try:
            import shap

            sample = X_test.head(50)
            # Use the pipeline's pandas-native output for SHAP to avoid warnings.
            X_sample_df = lgbm_pipe[:-1].transform(sample)
            explainer = shap.Explainer(lgbm_pipe[-1], X_sample_df)
            shap_values = explainer(X_sample_df)
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            feature_names = X_sample_df.columns.tolist()
            top_features = sorted(
                zip(feature_names[: len(mean_abs)], mean_abs),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
            metrics["shap_top_features"] = [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in top_features
            ]
            insights["shap_top_features"] = metrics["shap_top_features"]
        except Exception as e:
            print(f"Explainability step skipped or failed: {e}")

    if deep:
        try:
            import ml_project.deep.train as deep_train

            train_deep_models = deep_train.train_deep_models

            deep_metrics = train_deep_models(
                X_train,
                y_train,
                X_test,
                y_test,
                os.path.join(model_dir, "deep"),
                ensemble_mae=stack_metrics["mae_lakhs"],
            )
            metrics["deep_learning"] = deep_metrics
            metrics["promotion_criteria"]["deep_vs_ensemble"] = {
                "ensemble_mae_lakhs": stack_metrics["mae_lakhs"],
                "embedding_mlp_mae_lakhs": deep_metrics.get("embedding_mlp", {}).get(
                    "mae_lakhs"
                ),
                "tabnet_mae_lakhs": deep_metrics.get("tabnet", {}).get("mae_lakhs"),
                "beats_ensemble": deep_metrics.get("beats_ensemble", False),
                "api_promotion_recommended": deep_metrics.get("beats_ensemble", False),
            }
            print(
                "Deep learning vs ensemble:",
                metrics["promotion_criteria"]["deep_vs_ensemble"],
            )
        except (ImportError, OSError) as exc:
            print(f"Deep learning skipped: {exc}")
            metrics["deep_learning"] = {"error": str(exc)}

    with open(os.path.join(model_dir, "locations.json"), "w", encoding="utf-8") as f:
        json.dump(locations, f)

    with open(os.path.join(model_dir, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f)

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Models and metadata saved to {model_dir}")
    print(
        f"Ensemble — R²: {stack_metrics['r2']:.4f}, "
        f"MAE: {stack_metrics['mae_lakhs']:.2f}L, "
        f"MAPE: {stack_metrics['mape_pct']:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bangalore house price models")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--explain", action="store_true", help="Compute SHAP feature importance")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Train PyTorch embedding MLP and TabNet (research artifacts)",
    )
    parser.add_argument("--custom-data", type=str, default=None, help="Path to custom CSV data to merge")
    parser.add_argument("--no-dev", action="store_true", help="Disable dev mode console logging")
    args = parser.parse_args()
    run_training(tune=args.tune, explain=args.explain, deep=args.deep, custom_data_path=args.custom_data, dev=not args.no_dev)
