# Bangalore House Price ML Project

A full-stack machine learning application for predicting house prices in Bangalore.

## Project structure

```text
ML_Project/
├── backend/            # FastAPI service
├── frontend/           # React + Vite dashboard
├── ml_project/         # Shared ML package (training + inference)
├── ML/                 # Training entrypoint
├── data/               # Raw datasets
├── notebooks/          # EDA & research (see notebooks/README.md)
├── tests/              # pytest suite
├── scripts/            # Setup helpers
└── docker-compose.yml  # Container orchestration
```

## Quick start

### Automated setup (Windows)

```powershell
.\scripts\setup.ps1
```

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt
python ML/train.py
```

### Run the API

```powershell
cd backend
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs

### Run the frontend

```powershell
cd frontend
cp .env.example .env   # or create .env with VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Frontend API base URL |
| `MODEL_DIR` | `backend/models` | Directory for `.pkl` artifacts |
| `REQUIRE_MODELS` | `true` | Fail API startup if models missing |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins |

## Model consensus

The API computes a robust prediction consensus from:
1. **LightGBM Regressor**
2. **XGBoost Regressor**
3. **Stacking Ensemble** (consisting of XGBoost, LightGBM, and CatBoost base models mapped to a final Ridge Regressor metamodel)
4. **PyTorch Deep Learning MLP** (dynamically loaded if available)

The primary predicted price is determined by the Stacking Ensemble model (and compared against the Deep Learning MLP predictions). The JSON response returns per-model values, spread percentages, and details about the consensus logic.

## Docker

```powershell
# Train models first (artifacts mounted into backend)
python ML/train.py --deep --explain
docker compose up --build
```

## Testing

```powershell
pip install -r requirements-dev.txt
python ML/train.py
pytest tests/ -v
```

## Architecture

```mermaid
flowchart LR
  CSV[data/bengaluru_house_prices.csv]
  CustomCSV[data/user_contributed_prices.csv]
  Train[ML/train.py]
  Package[ml_project]
  Models[backend/models]
  API[FastAPI]
  UI[React]

  CSV --> Train
  CustomCSV --> Train
  Package --> Train
  Train --> Models
  Models --> API
  API --> UI
  UI -->|Custom listings| API
  API -->|Write custom CSV| CustomCSV

  %% Custom Styling
  classDef default fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc;
  classDef ui fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#ffffff;
  classDef api fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ffffff;
  classDef storage fill:#0b1329,stroke:#00b4d8,stroke-width:2px,color:#00b4d8;
  classDef pipeline fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#ffffff;

  class UI ui;
  class API api;
  class CSV,CustomCSV storage;
  class Train pipeline;
```
