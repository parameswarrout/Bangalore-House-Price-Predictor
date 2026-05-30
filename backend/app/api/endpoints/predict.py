import asyncio
from difflib import get_close_matches

import pandas as pd
from fastapi import APIRouter, HTTPException

from ml_project.preprocessing import (
    DEFAULT_POSSESSION_MONTHS,
    normalize_location_for_inference,
)

from app.config import get_settings
from app.ml.model import manager
from app.schemas.house import HouseInput, PriceResponse

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
