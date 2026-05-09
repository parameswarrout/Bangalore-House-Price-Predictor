from pydantic import BaseModel, Field

class HouseInput(BaseModel):
    location: str = Field(..., example="Indira Nagar")
    total_sqft: float = Field(..., gt=0, example=1200)
    bath: int = Field(..., ge=1, example=2)
    bhk: int = Field(..., ge=1, example=2)
    balcony: int = Field(default=1, ge=0)
    area_type_enc: int = Field(default=0, ge=0, le=3)
    is_ready_to_move: int = Field(default=1, ge=0, le=1)

class PriceResponse(BaseModel):
    predicted_price_lakhs: float
    predicted_price_crores: float
    model_consensus: dict[str, float]
