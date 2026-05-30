"""Train PyTorch embedding MLP and optional TabNet baseline."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml_project.metrics import regression_metrics
from ml_project.preprocessing import MODEL_FEATURES

NUMERIC_COLS = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "has_society",
    "location_count",
    "possession_months",
]


def _prepare_arrays(X: pd.DataFrame):
    loc_enc = LabelEncoder()
    loc_enc.fit(X["location"].astype(str))
    area_vals = X["area_type_enc"].astype(int).clip(0, 3).values

    numeric = X[NUMERIC_COLS].astype(float).values
    return numeric, loc_enc.transform(X["location"].astype(str)), area_vals, loc_enc


def _torch():
    import torch

    return torch


def _train_embedding_mlp(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    out_dir: str,
    epochs: int = 120,
    batch_size: int = 256,
    patience: int = 15,
) -> dict:
    torch = _torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from ml_project.deep.model import EmbeddingMLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tr, loc_tr, area_tr, loc_enc = _prepare_arrays(
        pd.concat([X_train, X_val], ignore_index=True)
    )
    n_train = len(X_train)
    num_train, num_val = num_tr[:n_train], num_tr[n_train:]
    loc_train, loc_val = loc_tr[:n_train], loc_tr[n_train:]
    area_train, area_val = area_tr[:n_train], area_tr[n_train:]
    ready_train = X_train["is_ready_to_move"].values
    ready_val = X_val["is_ready_to_move"].values

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_train)
    num_val = scaler.transform(num_val)

    model = EmbeddingMLP(n_locations=len(loc_enc.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    train_ds = TensorDataset(
        torch.tensor(num_train, dtype=torch.float32),
        torch.tensor(loc_train, dtype=torch.long),
        torch.tensor(area_train, dtype=torch.long),
        torch.tensor(ready_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_mae = float("inf")
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))
            loss = criterion(pred, batch[4].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(
                torch.tensor(num_val, dtype=torch.float32).to(device),
                torch.tensor(loc_val, dtype=torch.long).to(device),
                torch.tensor(area_val, dtype=torch.long).to(device),
                torch.tensor(ready_val, dtype=torch.float32).to(device),
            ).cpu().numpy()
        mae = regression_metrics(y_val, val_pred)["mae_lakhs"]
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(
            torch.tensor(num_val, dtype=torch.float32).to(device),
            torch.tensor(loc_val, dtype=torch.long).to(device),
            torch.tensor(area_val, dtype=torch.long).to(device),
            torch.tensor(ready_val, dtype=torch.float32).to(device),
        ).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "embedding_mlp.pt"))
    with open(os.path.join(out_dir, "encoders.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "locations": loc_enc.classes_.tolist(),
                "numeric_cols": NUMERIC_COLS,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            f,
        )

    return regression_metrics(y_val, test_pred)


def _train_tabnet(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict | None:
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
    except (ImportError, OSError):
        return None

    loc_enc = LabelEncoder()
    loc_enc.fit(pd.concat([X_train["location"], X_val["location"]]).astype(str))

    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_tr["location_idx"] = loc_enc.transform(X_tr["location"].astype(str))
    X_va["location_idx"] = loc_enc.transform(X_va["location"].astype(str))

    feature_cols = NUMERIC_COLS + ["area_type_enc", "is_ready_to_move", "location_idx"]
    cat_idxs = [len(feature_cols) - 1]
    cat_dims = [len(loc_enc.classes_)]

    reg = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
    )
    reg.fit(
        X_tr[feature_cols].values.astype(np.float32),
        y_train.reshape(-1, 1),
        eval_set=[(X_va[feature_cols].values.astype(np.float32), y_val.reshape(-1, 1))],
        max_epochs=80,
        patience=15,
        batch_size=256,
    )
    pred = reg.predict(X_va[feature_cols].values.astype(np.float32)).ravel()
    return regression_metrics(y_val, pred)


def train_deep_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    out_dir: str,
    ensemble_mae: float | None = None,
) -> dict:
    """Train research DL models; return holdout metrics."""
    _torch()

    # Ensure target values are numpy arrays to avoid shape errors with pandas Series
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print("Training Embedding MLP...")
    mlp_metrics = _train_embedding_mlp(X_train, y_train, X_test, y_test, out_dir)

    print("Training TabNet (if pytorch-tabnet installed)...")
    tabnet_metrics = _train_tabnet(X_train, y_train, X_test, y_test)

    result = {"embedding_mlp": mlp_metrics}
    if tabnet_metrics:
        result["tabnet"] = tabnet_metrics

    best_mae = mlp_metrics["mae_lakhs"]
    if tabnet_metrics and tabnet_metrics["mae_lakhs"] < best_mae:
        best_mae = tabnet_metrics["mae_lakhs"]

    result["best_dl_mae_lakhs"] = best_mae
    result["beats_ensemble"] = (
        ensemble_mae is not None and best_mae < ensemble_mae
    )
    return result
