import asyncio
import math
from difflib import get_close_matches

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ml_project.preprocessing import (
    DEFAULT_POSSESSION_MONTHS,
    normalize_location_for_inference,
)

from app.config import get_settings
from app.ml.model import manager
from app.schemas.house import ExplainResponse, FeatureContribution, HouseInput, PriceResponse


router = APIRouter()


def _validate_location(location: str) -> None:
    known = manager.get_locations()
    if not known:
        return
    if location in known:
        return
    suggestions = get_close_matches(location, known, n=3, cutoff=0.6)
    detail = f"Unknown location: '{location}'"
    if suggestions:
        detail += f". Did you mean: {', '.join(suggestions)}?"
    raise HTTPException(status_code=422, detail=detail)


def _compute_consensus(all_preds: dict[str, float]) -> tuple[float, str, float | None]:
    if not all_preds:
        raise HTTPException(status_code=500, detail="No predictions returned")

    if "ensemble" in all_preds:
        primary = all_preds["ensemble"]
        method = "ensemble_primary"
    elif "lgbm" in all_preds:
        primary = all_preds["lgbm"]
        method = "lgbm_fallback"
    else:
        primary = list(all_preds.values())[0]
        method = "first_available"

    prices = list(all_preds.values())
    spread_pct = None
    if len(prices) >= 2:
        mean = sum(prices) / len(prices)
        spread_pct = round((max(prices) - min(prices)) / mean * 100, 2) if mean else None

    return primary, method, spread_pct


def _predict_sync(data: HouseInput) -> PriceResponse:
    if not manager.models:
        raise HTTPException(
            status_code=500,
            detail="No models loaded. Run 'python ML/train.py' from the repo root.",
        )

    _validate_location(data.location)

    loc = normalize_location_for_inference(
        data.location, manager.location_counts or None
    )
    loc_count = (manager.location_counts or {}).get(loc, 1)

    df = pd.DataFrame([{
        "location": loc,
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk,
        "area_type_enc": data.area_type_enc,
        "is_ready_to_move": data.is_ready_to_move,
        "has_society": data.has_society,
        "location_count": loc_count,
        "possession_months": data.possession_months
        if data.possession_months is not None
        else (0 if data.is_ready_to_move else DEFAULT_POSSESSION_MONTHS),
    }])

    all_preds = manager.predict_all(df)
    primary_price, method, spread_pct = _compute_consensus(all_preds)
    settings = get_settings()

    return PriceResponse(
        predicted_price_lakhs=primary_price,
        predicted_price_crores=round(primary_price / 100, 4),
        model_consensus=all_preds,
        model_version=settings["model_version"],
        consensus_method=method,
        spread_pct=spread_pct,
    )


@router.post("/predict", response_model=PriceResponse)
async def predict_price(data: HouseInput):
    return await asyncio.to_thread(_predict_sync, data)


@router.get("/locations")
def get_locations():
    locations = manager.get_locations()
    if not locations:
        return ["Indira Nagar", "Whitefield", "Electronic City", "Other"]
    return locations


@router.get("/insights")
def get_insights():
    return manager.get_insights()


@router.get("/model-info")
def get_model_info():
    return manager.get_model_info()


# ──────────────────────────────────────────────
# SHAP Explanation Endpoint
# ──────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "total_sqft": "Total Area",
    "bath": "Bathrooms",
    "bhk": "Bedrooms (BHK)",
    "balcony": "Balcony Count",
    "area_type_enc": "Area Type",
    "is_ready_to_move": "Ready to Move",
    "has_society": "Gated Society",
    "location_count": "Location Popularity",
    "possession_months": "Possession Timeline",
    "location": "Location (Locality)",
}

AREA_TYPE_LABELS = {0: "Super Built-up", 1: "Built-up", 2: "Plot Area", 3: "Carpet Area"}


def _build_raw_value_label(feature: str, value, formData: HouseInput) -> str:
    if feature == "location":
        return formData.location
    if feature == "area_type_enc":
        return AREA_TYPE_LABELS.get(int(value), str(value))
    if feature == "is_ready_to_move":
        return "Ready to Move" if value == 1 else "Under Construction"
    if feature == "has_society":
        return "Yes" if value == 1 else "No"
    if feature == "possession_months":
        return f"{int(value)} months"
    if feature == "total_sqft":
        return f"{int(value)} sqft"
    return str(int(value)) if isinstance(value, float) and value == int(value) else str(round(value, 2))


def _explain_sync(data: HouseInput) -> ExplainResponse:
    try:
        import shap
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP not installed. Run: pip install shap")

    if "ensemble" not in manager.models:
        raise HTTPException(status_code=500, detail="Stacking ensemble model not loaded.")

    _validate_location(data.location)

    loc = normalize_location_for_inference(data.location, manager.location_counts or None)
    loc_count = (manager.location_counts or {}).get(loc, 1)

    row = {
        "location": loc,
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk,
        "area_type_enc": data.area_type_enc,
        "is_ready_to_move": data.is_ready_to_move,
        "has_society": data.has_society,
        "location_count": loc_count,
        "possession_months": data.possession_months
        if data.possession_months is not None
        else (0 if data.is_ready_to_move else DEFAULT_POSSESSION_MONTHS),
    }
    df = pd.DataFrame([row])

    ensemble = manager.models["ensemble"]
    settings = get_settings()

    try:
        explainer = shap.TreeExplainer(ensemble)
        shap_values = explainer.shap_values(df)  # shape: (1, n_features)
        base_log = float(explainer.expected_value)
        sv = shap_values[0]  # array of shape (n_features,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explainer failed: {e}")

    # Convert log-space SHAP → Lakhs contributions using delta method
    # predicted_log = base_log + sum(sv)
    predicted_log = base_log + float(np.sum(sv))
    base_lakhs = round(float(np.expm1(base_log)), 2)
    predicted_lakhs = round(float(np.expm1(predicted_log)), 2)

    feature_names = list(df.columns)
    contributions = []
    for i, feat in enumerate(feature_names):
        sv_i = float(sv[i])
        # Convert marginal log contribution to Lakhs delta
        # contribution_lakhs = expm1(base + sum_so_far + sv_i) - expm1(base + sum_so_far)
        # Approximate: contribution ≈ sv_i * exp(base + cumsum_except_i)  (first-order)
        partial_log = base_log + float(np.sum(sv)) - sv_i
        contribution_lakhs = round(float(np.expm1(partial_log + sv_i) - np.expm1(partial_log)), 2)

        raw_val = row[feat]
        contributions.append(FeatureContribution(
            feature=feat,
            display_name=FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
            raw_value=_build_raw_value_label(feat, raw_val, data),
            shap_value=round(sv_i, 5),
            contribution_lakhs=contribution_lakhs,
        ))

    # Sort by absolute contribution descending
    contributions.sort(key=lambda c: abs(c.contribution_lakhs), reverse=True)

    return ExplainResponse(
        base_value_lakhs=base_lakhs,
        predicted_price_lakhs=predicted_lakhs,
        contributions=contributions,
        model_version=settings["model_version"],
    )


@router.post("/predict/explain", response_model=ExplainResponse)
async def explain_prediction(data: HouseInput):
    return await asyncio.to_thread(_explain_sync, data)

