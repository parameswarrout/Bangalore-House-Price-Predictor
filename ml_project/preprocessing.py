import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AREA_TYPE_MAP = {
    "Super built-up  Area": 0,
    "Built-up  Area": 1,
    "Plot  Area": 2,
    "Carpet  Area": 3,
}

AREA_TYPE_LABELS = {v: k for k, v in AREA_TYPE_MAP.items()}

RARE_LOCATION_THRESHOLD = 10
RARE_LOCATION_LABEL = "other"
POSSESSION_REFERENCE = pd.Timestamp("2020-01-01")
DEFAULT_POSSESSION_MONTHS = 12


def clean_total_sqft(x):
    """Handles range values like '1200-1400' and converts to float."""
    tokens = str(x).split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except (ValueError, TypeError):
        logger.debug("Could not parse total_sqft value: %s", x)
        return None


def parse_possession_months(availability: str) -> int:
    """Months from reference date until possession; 0 if ready to move."""
    if pd.isna(availability) or str(availability).strip() == "Ready To Move":
        return 0
    dt = pd.to_datetime(availability, errors="coerce")
    if pd.isna(dt):
        return DEFAULT_POSSESSION_MONTHS
    months = (dt.year - POSSESSION_REFERENCE.year) * 12 + (
        dt.month - POSSESSION_REFERENCE.month
    )
    return int(max(0, months))


def bucket_rare_locations(df: pd.DataFrame, threshold: int = RARE_LOCATION_THRESHOLD) -> pd.DataFrame:
    """Collapse locations with <= threshold listings into 'other'."""
    df = df.copy()
    df["location"] = df["location"].astype(str).str.strip()
    counts = df.groupby("location")["location"].transform("count")
    df.loc[counts <= threshold, "location"] = RARE_LOCATION_LABEL
    return df


def remove_pps_outliers(df: pd.DataFrame, std_multiplier: float = 1.0) -> pd.DataFrame:
    """Removes outliers based on price per sqft per location."""
    df_out = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        if st == 0 or np.isnan(st):
            df_out = pd.concat([df_out, subdf], ignore_index=True)
            continue
        reduced_df = subdf[
            (subdf.price_per_sqft > (m - std_multiplier * st)) & (subdf.price_per_sqft <= (m + std_multiplier * st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


def apply_price_cap(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Drop listings above the given price quantile (training filter)."""
    cap = df["price"].quantile(quantile)
    return df[df["price"] <= cap].copy()


def build_location_counts(df: pd.DataFrame) -> dict[str, int]:
    """Location listing counts after bucketing (for inference lookups)."""
    return df.groupby("location").size().astype(int).to_dict()


def normalize_location_for_inference(
    location: str,
    location_counts: dict[str, int] | None = None,
    threshold: int = RARE_LOCATION_THRESHOLD,
) -> str:
    """Map unknown or rare locations to 'other' at inference time."""
    loc = str(location).strip()
    if location_counts is not None:
        count = location_counts.get(loc, 0)
        if count <= threshold or loc not in location_counts:
            return RARE_LOCATION_LABEL
    return loc


def remove_outliers_and_cap_prices(df: pd.DataFrame, quantile: float = 0.99, std_multiplier: float = 1.0) -> pd.DataFrame:
    """Applies outlier removal and global price capping (training-only filters)."""
    df = df.copy()
    if "price_per_sqft" not in df.columns and "price" in df.columns:
        df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
    df = remove_pps_outliers(df, std_multiplier=std_multiplier)
    df = apply_price_cap(df, quantile=quantile)
    return df


def apply_feature_engineering(
    df: pd.DataFrame,
    *,
    apply_training_filters: bool = True,
    rare_location_threshold: int = RARE_LOCATION_THRESHOLD,
    location_counts: dict[str, int] | None = None,
    std_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Applies V2 feature engineering aligned with the research notebook.

    Training-only steps (when apply_training_filters=True):
      - per-location price/sqft outlier removal
      - global price cap at p99
    """
    df = df.copy()
    if "society" in df.columns:
        df["has_society"] = df["society"].notna().astype(int)
    else:
        df["has_society"] = 0

    df["bhk"] = df["size"].apply(
        lambda x: int(str(x).split(" ")[0]) if pd.notnull(x) else 2
    )
    df["balcony"] = df["balcony"].fillna(df["balcony"].median())
    df["area_type_enc"] = df["area_type"].map(AREA_TYPE_MAP).fillna(0).astype(int)

    if "availability" in df.columns:
        df["is_ready_to_move"] = (df["availability"] == "Ready To Move").astype(int)
        df["possession_months"] = df["availability"].apply(parse_possession_months)
    else:
        df["is_ready_to_move"] = df.get("is_ready_to_move", 1)
        df["possession_months"] = df.apply(
            lambda row: 0
            if row.get("is_ready_to_move", 1) == 1
            else DEFAULT_POSSESSION_MONTHS,
            axis=1,
        )

    df["location"] = df["location"].astype(str).str.strip()
    
    if location_counts is not None:
        # Use precomputed location counts (inference/test set)
        df["location"] = df["location"].apply(
            lambda loc: normalize_location_for_inference(
                loc, location_counts=location_counts, threshold=rare_location_threshold
            )
        )
        df["location_count"] = df["location"].apply(
            lambda loc: location_counts.get(loc, 1)
        )
    else:
        # On-the-fly bucketing (training set base)
        df = bucket_rare_locations(df, threshold=rare_location_threshold)
        df["location_count"] = df.groupby("location")["location"].transform("count")

    df = df[df.total_sqft / df.bhk >= 300]
    
    if "price" in df.columns:
        df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

    if apply_training_filters:
        df = remove_outliers_and_cap_prices(df, quantile=0.99, std_multiplier=std_multiplier)

    df["sqft_per_room"] = df["total_sqft"] / (df["bhk"] + df["bath"])
    df["room_density"] = df["bhk"] / (df["total_sqft"] / 1000)
    df["bath_to_bhk"] = df["bath"] / df["bhk"].apply(lambda x: max(x, 1))
    df["total_rooms"] = df["bhk"] + df["bath"] + df["balcony"]
    return df


import os


def load_and_prepare_training_frame(
    csv_path: str,
    custom_csv_path: str = None,
    apply_training_filters: bool = True,
    rare_location_threshold: int = RARE_LOCATION_THRESHOLD,
    location_counts: dict[str, int] | None = None,
    std_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Load raw CSV and return the cleaned training dataframe."""
    df = pd.read_csv(csv_path)
    if custom_csv_path and os.path.exists(custom_csv_path):
        try:
            df_custom = pd.read_csv(custom_csv_path)
            logger.info("Loaded %d custom rows from %s", len(df_custom), custom_csv_path)
            df = pd.concat([df, df_custom], ignore_index=True)
        except Exception as e:
            logger.error("Failed to load custom CSV: %s", e)
            
    df["total_sqft"] = df["total_sqft"].apply(clean_total_sqft)
    df = df.dropna(subset=["total_sqft", "bath", "location", "price"])
    return apply_feature_engineering(
        df,
        apply_training_filters=apply_training_filters,
        rare_location_threshold=rare_location_threshold,
        location_counts=location_counts,
        std_multiplier=std_multiplier,
    )


EXPECTED_COLUMNS = [
    "area_type",
    "availability",
    "location",
    "size",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "price",
]

MODEL_FEATURES = [
    "location",
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "area_type_enc",
    "is_ready_to_move",
    "has_society",
    "location_count",
    "possession_months",
]
