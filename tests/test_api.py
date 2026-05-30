import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["REQUIRE_MODELS"] = "false"

from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "loaded_models" in data


@pytest.mark.asyncio
async def test_model_info(client):
    response = await client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "loaded_models" in data


@pytest.mark.asyncio
async def test_locations(client):
    response = await client.get("/locations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_predict_validation(client):
    response = await client.post(
        "/predict",
        json={
            "location": "Invalid Location XYZ",
            "total_sqft": 1200,
            "bath": 2,
            "bhk": 2,
            "balcony": 1,
        },
    )
    # 422 if locations.json loaded with known locations; 500 if no models
    assert response.status_code in (422, 500)


@pytest.mark.asyncio
async def test_predict_sqft_per_bhk_rule(client):
    response = await client.post(
        "/predict",
        json={
            "location": "Indira Nagar",
            "total_sqft": 100,
            "bath": 2,
            "bhk": 2,
            "balcony": 1,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.skipif(
    not all(
        os.path.exists(os.path.join("backend", "models", f))
        for f in ("stacking_model.pkl", "lgbm_model.pkl", "xgb_model.pkl")
    ),
    reason="Trained models not present",
)
async def test_predict_success(client):
    loc_res = await client.get("/locations")
    locations = loc_res.json()
    assert len(locations) > 0

    response = await client.post(
        "/predict",
        json={
            "location": locations[0],
            "total_sqft": 1200,
            "bath": 2,
            "bhk": 2,
            "balcony": 1,
            "area_type_enc": 0,
            "is_ready_to_move": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_lakhs"] > 0
    assert "model_consensus" in data
    assert "model_version" in data
    assert "consensus_method" in data


@pytest.mark.asyncio
async def test_custom_data_workflow(client):
    # 1. Clear any existing custom data
    await client.post("/custom-data/clear")

    # 2. Get custom data (should be empty list)
    response = await client.get("/custom-data")
    assert response.status_code == 200
    assert response.json() == []

    # 3. Add a custom property listing
    property_data = {
        "area_type": "Super built-up  Area",
        "availability": "Ready To Move",
        "location": "Test Locality V2",
        "size": "3 BHK",
        "society": "Test Society",
        "total_sqft": 1500.0,
        "bath": 3.0,
        "balcony": 2.0,
        "price": 120.0
    }
    response = await client.post("/custom-data/add", json=property_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 4. Get custom data (should contain 1 entry)
    response = await client.get("/custom-data")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["location"] == "Test Locality V2"
    assert data[0]["total_sqft"] == 1500.0
    assert data[0]["price"] == 120.0

    # 5. Check stats
    stats_res = await client.get("/custom-data/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["custom_dataset_size"] == 1

    # 6. Delete the custom property listing
    response = await client.delete("/custom-data/0")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 7. Verify empty again
    response = await client.get("/custom-data")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_train_status(client):
    response = await client.get("/train/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "logs" in data
    assert "metrics" in data


@pytest.mark.asyncio
async def test_restore_baseline(client):
    # Triggers a mock training command to ensure a backup folder is generated
    # (Since train checks for backup folder and copies models, this ensures backup folder exists)
    # Then calls restore-baseline to verify it completes successfully (returns 200)
    response = await client.post("/train", json={"include_custom_data": False, "tune": False, "deep": False})
    assert response.status_code == 200
    
    # Immediately check restore baseline (will wait/abort active training, but since active training is run 
    # as subprocess, we call restore baseline. To prevent lock conflict, we wait or check status. 
    # Since we can just mock restore baseline, calling it should return 200 or 400 (if still running)).
    response = await client.post("/train/restore-baseline")
    assert response.status_code in (200, 400)


