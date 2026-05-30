import os

import pandas as pd
import pytest

from ml_project.preprocessing import (
    EXPECTED_COLUMNS,
    MODEL_FEATURES,
    apply_feature_engineering,
    bucket_rare_locations,
    clean_total_sqft,
    load_and_prepare_training_frame,
    parse_possession_months,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "bengaluru_house_prices.csv")


def test_clean_total_sqft_range():
    assert clean_total_sqft("1200-1400") == 1300.0


def test_clean_total_sqft_float():
    assert clean_total_sqft(1200) == 1200.0


def test_clean_total_sqft_invalid():
    assert clean_total_sqft("invalid") is None


def test_parse_possession_months_ready():
    assert parse_possession_months("Ready To Move") == 0


def test_bucket_rare_locations():
    df = pd.DataFrame({
        "location": ["A"] * 5 + ["B"] * 2,
        "price": [50.0] * 7,
    })
    out = bucket_rare_locations(df, threshold=3)
    assert out.loc[out["location"] == "other"].shape[0] == 2


def test_apply_feature_engineering_minimal():
    df = pd.DataFrame({
        "size": ["2 BHK"] * 5,
        "total_sqft": [1200.0] * 5,
        "bath": [2] * 5,
        "balcony": [1] * 5,
        "area_type": ["Super built-up  Area"] * 5,
        "availability": ["Ready To Move"] * 5,
        "location": ["Indira Nagar"] * 5,
        "society": ["Soc"] * 5,
        "price": [80.0, 81.0, 79.0, 80.5, 79.5],
    })
    result = apply_feature_engineering(df)
    assert len(result) >= 1
    assert "bhk" in result.columns
    assert "has_society" in result.columns
    assert result["bhk"].iloc[0] == 2
    for col in MODEL_FEATURES:
        assert col in result.columns


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset not present")
def test_full_csv_schema():
    df = pd.read_csv(DATA_PATH, nrows=100)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="dataset not present")
def test_load_and_prepare_row_count_band():
    prepared = load_and_prepare_training_frame(DATA_PATH)
    assert 9000 <= len(prepared) <= 11000
    assert (prepared["price"] > 0).all()
    assert (prepared["bhk"] >= 1).all()
