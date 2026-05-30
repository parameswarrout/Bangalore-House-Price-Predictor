# Property Analytics - Machine Learning Pipeline

This folder contains the training engine and feature engineering modules for predicting property prices in Bangalore.

---

## 1. Conceptual System Integration Overview

The flowchart below visualizes how the manual form UI, the CSV dataset uploader, the backend API endpoints, the separate data storage layer, and the asynchronous model hot-reloader connect together:

```mermaid
graph TD
    A[Frontend UI: Custom Data Form] -->|POST /api/custom-data| B[Backend API]
    B -->|Save Separately| C[(user_contributed_prices.csv)]
    
    D[(bengaluru_house_prices.csv)] --> E[Training Pipeline]
    C -->|Optional Merge| E
    
    F[Frontend UI: Retrain Button] -->|POST /api/train| B
    B -->|Background Task| E
    E -->|Train XGB, LGBM, Stacking| G[(models/*.pkl)]
    G -->|Hot Reload| H[Model Manager]
    H -->|Serve Predictions| A
```

---

## 2. Machine Learning Pipeline Flowchart

The flowchart below outlines the data ingestion, preprocessing filters, model stacking, hyperparameter optimization, and output artifact exportation of the ML pipeline.

```mermaid
graph TD
    A[(Base Dataset: bengaluru_house_prices.csv)] -->|Data Ingestion| C[pandas.concat]
    B[(Custom Dataset: user_contributed_prices.csv)] -->|Optional Merge| C
    
    C -->|Preprocessing & Filters| D{Preprocessing Pipeline}
    D -->|Clean Total Sqft| D1[Parse ranges '1200-1400' to averages]
    D -->|BHK Validation| D2[Drop total_sqft / bhk < 300]
    D -->|Location Bucketing| D3[Rare localities mapped to 'other']
    D -->|Outliers Filters| D4[Filter price per sqft outliers per locality]
    D -->|Price Capping| D5[Clip top 1% prices to filter extremes]
    
    D5 -->|Feature Engineering| E[Feature Transformers]
    E -->|Interact Features| E1[sqft_per_room, room_density, bath_to_bhk, total_rooms]
    E -->|Location Encoders| E2[Target Encoding for localities]
    E -->|Scalers| E3[StandardScaler normalization]
    
    E3 -->|Dataset Split| F[Train / Holdout Split 80/20]
    
    subgraph Model Stack Training
        F -->|Option: --tune| G[Optuna Hyperparameter Tuning]
        G -->|Optimize Parameters| H1[XGBoost Pipeline]
        G -->|Optimize Parameters| H2[LightGBM Pipeline]
        G -->|Optimize Parameters| H3[CatBoost Pipeline]
        
        F -->|Default Config| H1
        F -->|Default Config| H2
        F -->|Default Config| H3
        
        H1 -->|Fit Base Learners| I[Stacking Regressor]
        H2 -->|Fit Base Learners| I
        H3 -->|Fit Base Learners| I
        I -->|Final Estimator| J[Ridge Regression Metamodel]
    end
    
    subgraph Advanced Capabilities
        F -->|Option: --explain| K[SHAP Value Explainability]
        F -->|Option: --deep| L[PyTorch Deep Learning & TabNet]
    end
    
    J -->|Serialize Models| M[(backend/models/*.pkl)]
    K -->|Extract Feature Importances| N[(insights.json)]
    I -->|Evaluate metrics| O[(metrics.json)]
    D3 -->|Generate lookup| P[(location_counts.json & locations.json)]
```

---

## 3. Preprocessing & Feature Engineering Details

The preprocessing steps defined in `ml_project/preprocessing.py` ensure high data quality and representational capacity for regression fitting:

1. **Total Sqft Parsing**: Resolves range strings (e.g. `"1200 - 1400"`) into floating averages (`1300.0`) and drops non-numeric artifacts.
2. **Locality Binning**: Locations with 10 or fewer listings are collapsed into a generic `"other"` category to prevent target-encoder overfitting on thin data.
3. **Outlier Filtering**: Listings exceeding standard deviations of price-per-square-foot in their respective localities are dropped to remove data-entry anomalies.
4. **Interaction Variables**: Captures critical property design ratios:
   * **Room Density**: $\text{BHK} / (\text{Total Sqft} / 1000)$
   * **Bathroom-to-BHK Ratio**: $\text{Bathrooms} / \text{BHK}$
   * **Square Feet per Room**: $\text{Total Sqft} / (\text{BHK} + \text{Bathrooms})$

---

## 4. Stacking Ensemble Architecture

The core model is a **Stacking Regressor** (`sklearn.ensemble.StackingRegressor`) consisting of:
* **Base Regressors**:
  1. **XGBoost Regressor** (`xgboost.XGBRegressor`) inside a processing Pipeline.
  2. **LightGBM Regressor** (`lightgbm.LGBMRegressor`) inside a processing Pipeline.
  3. **CatBoost Regressor** (`catboost.CatBoostRegressor`) (if installed).
* **Final Metamodel**:
  * **Ridge Regressor** (`sklearn.linear_model.Ridge`) trained using Out-Of-Fold (OOF) cross-validation predictions from base learners.

---

## 5. In-Training Flags and Utilities

Run the pipeline from the project root using `sys.executable ML/train.py` with the following flags:

* `--tune`: Launches **Optuna** hyperparameter search running 15 validation trials per learner to optimize cross-validated $R^2$ before stacking.
* `--explain`: Evaluates the **SHAP** TreeExplainer on the LightGBM model, generating the top 8 global features and exporting them to `insights.json` for frontend analytics.
* `--deep`: Trains PyTorch-based **Embedding MLP** and **TabNet** architectures, writing metrics to `metrics.json` for comparative research logs.
* `--custom-data <path>`: Integrates user-provided CSV listings with the main dataset before fitting models.
