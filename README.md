# Bangalore Property Analytics

A production-grade machine learning application for predicting house prices in Bangalore.

## 📁 Project Structure

```text
Rasonix_ML_Project/
├── backend/            # FastAPI service for predictions
├── frontend/           # React + Vite dashboard
├── ML/                 # Automated model training scripts
├── data/               # Raw datasets (CSV)
├── notebooks/          # Exploratory Data Analysis & research
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### 1. Training the Models
To train the Top 3 models (Stacking, LGBM, XGBoost) and save them for the API:
```powershell
python ML/train.py
```

### 2. Running the Backend
```powershell
cd backend
uvicorn main:app --reload
```

### 3. Running the Frontend
```powershell
cd frontend
npm run dev
```

## 🧠 Model Consensus Feature
This POC features a **Consensus Engine** that compares predictions from multiple high-performance algorithms to ensure valuation accuracy.
