from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from typing import Dict, Optional

from app.schemas.house import HouseInput, PriceResponse
from app.ml.model import manager

router = APIRouter()

@router.post("/predict", response_model=PriceResponse)
def predict_price(data: HouseInput):
    if not manager.models:
        raise HTTPException(
            status_code=500, 
            detail="No models loaded. Please ensure model files exist in the models/ folder."
        )
    
    # Just prepare the raw input DataFrame
    # The InteractionFeatureTransformer inside the Pipeline will handle the rest
    df = pd.DataFrame([{
        "location"        : data.location,
        "total_sqft"      : data.total_sqft,
        "bath"            : data.bath,
        "balcony"         : data.balcony,
        "bhk"             : data.bhk,
        "area_type_enc"   : data.area_type_enc,
        "is_ready_to_move": data.is_ready_to_move,
    }])

    # Get predictions from all available models
    all_preds = manager.predict_all(df)
    
    # Primary price (weighted average or specific model preference)
    if "ensemble" in all_preds:
        primary_price = all_preds["ensemble"]
    elif "lgbm" in all_preds:
        primary_price = all_preds["lgbm"]
    else:
        primary_price = list(all_preds.values())[0]

    return PriceResponse(
        predicted_price_lakhs=primary_price,
        predicted_price_crores=round(primary_price / 100, 4),
        model_consensus=all_preds
    )

@router.get("/locations")
def get_locations():
    locations = manager.get_locations()
    if not locations:
        # Fallback if locations.json is missing
        return ["Indira Nagar", "Whitefield", "Electronic City", "Other"]
    return locations

@router.get("/insights")
def get_insights():
    return manager.get_insights()
