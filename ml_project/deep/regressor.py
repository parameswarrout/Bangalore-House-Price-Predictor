import os
import math
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml_project.preprocessing import MODEL_FEATURES
from ml_project.deep.model import EmbeddingMLP

NUMERIC_COLS = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "has_society",
    "location_count",
    "possession_months",
]


def _to_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=MODEL_FEATURES)


class EmbeddingMLPRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn compatible wrapper for PyTorch EmbeddingMLP."""

    _estimator_type = "regressor"

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 256,
        patience: int = 15,
        device: str = "cpu",
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device_ = self.device
        
        # Enforce CPU if CUDA is not available
        if self.device_ == "cuda" and not torch.cuda.is_available():
            self.device_ = "cpu"

        X_df = _to_dataframe(X)
        y_arr = np.asarray(y, dtype=np.float32)

        # Build Label Encoder
        self.loc_enc_ = LabelEncoder()
        self.loc_enc_.fit(X_df["location"].astype(str))
        n_locations = len(self.loc_enc_.classes_)

        # Prepare categorical area types
        area_vals = X_df["area_type_enc"].astype(int).clip(0, 3).values
        location_idx = self.loc_enc_.transform(X_df["location"].astype(str))
        ready_vals = X_df["is_ready_to_move"].astype(float).values

        # Prepare Scaler for Numeric features
        numeric_vals = X_df[NUMERIC_COLS].astype(float).values
        self.scaler_ = StandardScaler()
        numeric_scaled = self.scaler_.fit_transform(numeric_vals)

        # Setup PyTorch components
        train_ds = TensorDataset(
            torch.tensor(numeric_scaled, dtype=torch.float32),
            torch.tensor(location_idx, dtype=torch.long),
            torch.tensor(area_vals, dtype=torch.long),
            torch.tensor(ready_vals, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
        )
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.model_ = EmbeddingMLP(
            n_locations=n_locations,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.HuberLoss()

        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                pred = self.model_(
                    batch[0].to(self.device_),
                    batch[1].to(self.device_),
                    batch[2].to(self.device_),
                    batch[3].to(self.device_),
                )
                loss = criterion(pred, batch[4].to(self.device_))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1:
                avg_loss = epoch_loss / len(loader)
                print(f"  [EmbeddingMLP] Epoch {epoch + 1:02d}/{self.epochs} - Loss: {avg_loss:.5f}")

        return self

    def predict(self, X):
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("This EmbeddingMLPRegressor instance is not fitted yet.")

        self.model_.eval()
        X_df = _to_dataframe(X)

        # Handle unseen categories gracefully at prediction/inference time
        loc_classes = self.loc_enc_.classes_.tolist()
        default_idx = loc_classes.index("other") if "other" in loc_classes else 0

        locations_str = X_df["location"].astype(str).values
        location_idx = []
        for loc in locations_str:
            if loc in self.loc_enc_.classes_:
                location_idx.append(self.loc_enc_.transform([loc])[0])
            else:
                location_idx.append(default_idx)
        location_idx = np.array(location_idx, dtype=np.int64)

        area_vals = X_df["area_type_enc"].astype(int).clip(0, 3).values
        ready_vals = X_df["is_ready_to_move"].astype(float).values

        numeric_vals = X_df[NUMERIC_COLS].astype(float).values
        numeric_scaled = self.scaler_.transform(numeric_vals)

        with torch.no_grad():
            preds = self.model_(
                torch.tensor(numeric_scaled, dtype=torch.float32).to(self.device_),
                torch.tensor(location_idx, dtype=torch.long).to(self.device_),
                torch.tensor(area_vals, dtype=torch.long).to(self.device_),
                torch.tensor(ready_vals, dtype=torch.float32).to(self.device_),
            )
        return preds.cpu().numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Move state dict / module to CPU before pickling
        if "model_" in state:
            import torch
            state["model_"] = state["model_"].cpu()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore device context
        import torch
        if hasattr(self, "model_"):
            device = "cuda" if (torch.cuda.is_available() and self.device == "cuda") else "cpu"
            self.device_ = device
            self.model_ = self.model_.to(device)
