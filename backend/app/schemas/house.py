from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml_project.preprocessing import AREA_TYPE_MAP


class HouseInput(BaseModel):
    location: str = Field(..., examples=["Indira Nagar"])
    total_sqft: float = Field(..., gt=0, examples=[1200])
    bath: int = Field(..., ge=1, examples=[2])
    bhk: int = Field(..., ge=1, examples=[2])
    balcony: int = Field(default=1, ge=0)
    area_type_enc: int = Field(default=0, ge=0, le=3)
    is_ready_to_move: int = Field(default=1, ge=0, le=1)
    has_society: int = Field(default=0, ge=0, le=1)
    possession_months: int | None = Field(
        default=None,
        ge=0,
        description="Months until possession; 0 if ready. Inferred from is_ready_to_move when omitted.",
    )
    area_type: str | None = Field(
        default=None,
        description="Human-readable area type; overrides area_type_enc if set",
    )

    @field_validator("area_type")
    @classmethod
    def normalize_area_type(cls, v):
        if v is None:
            return v
        if v not in AREA_TYPE_MAP:
            valid = ", ".join(AREA_TYPE_MAP.keys())
            raise ValueError(f"area_type must be one of: {valid}")
        return v

    @model_validator(mode="after")
    def apply_area_type_and_business_rules(self):
        if self.area_type is not None:
            self.area_type_enc = AREA_TYPE_MAP[self.area_type]
        if self.possession_months is None:
            self.possession_months = 0 if self.is_ready_to_move == 1 else 12
        if self.total_sqft / self.bhk < 300:
            raise ValueError(
                f"total_sqft per BHK must be at least 300 (got {self.total_sqft / self.bhk:.0f})"
            )
        return self


class PriceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_price_lakhs: float
    predicted_price_crores: float
    model_consensus: dict[str, float]
    model_version: str
    consensus_method: str
    spread_pct: float | None = None


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    raw_value: str
    shap_value: float
    contribution_lakhs: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_value_lakhs: float
    predicted_price_lakhs: float
    contributions: list[FeatureContribution]
    model_version: str

