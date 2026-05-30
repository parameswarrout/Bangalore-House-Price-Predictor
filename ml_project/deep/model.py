import math

import torch
import torch.nn as nn  # noqa: F401 — required at class definition time when training runs


class EmbeddingMLP(nn.Module):
    """Tabular regressor with embeddings for location and area type."""

    def __init__(
        self,
        n_locations: int,
        n_area_types: int = 4,
        location_emb_dim: int | None = None,
        area_emb_dim: int = 2,
        numeric_dim: int = 7,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if location_emb_dim is None:
            location_emb_dim = min(50, max(8, math.ceil(n_locations**0.25)))

        self.location_emb = nn.Embedding(n_locations, location_emb_dim)
        self.area_emb = nn.Embedding(n_area_types, area_emb_dim)

        in_dim = numeric_dim + location_emb_dim + area_emb_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        location_idx: torch.Tensor,
        area_idx: torch.Tensor,
        ready: torch.Tensor,
    ) -> torch.Tensor:
        loc_e = self.location_emb(location_idx)
        area_e = self.area_emb(area_idx)
        ready_e = ready.unsqueeze(1).float()
        x = torch.cat([numeric, loc_e, area_e, ready_e], dim=1)
        return self.mlp(x).squeeze(1)
