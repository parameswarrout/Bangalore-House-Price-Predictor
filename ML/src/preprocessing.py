"""Backward-compatible re-exports. Prefer `from ml_project.preprocessing import ...`."""

from ml_project.preprocessing import (
    AREA_TYPE_LABELS,
    AREA_TYPE_MAP,
    apply_feature_engineering,
    clean_total_sqft,
)

__all__ = [
    "AREA_TYPE_LABELS",
    "AREA_TYPE_MAP",
    "apply_feature_engineering",
    "clean_total_sqft",
]
