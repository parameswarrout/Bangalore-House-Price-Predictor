import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import from our new src structure
from src.transformers import LocationTargetEncoder, InteractionFeatureTransformer
from src.preprocessing import clean_total_sqft, apply_feature_engineering

def run_training():
    print("Starting Property Price Model Training (Modular Structure)...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'bengaluru_house_prices.csv')
    model_dir = os.path.join(base_dir, 'backend', 'models')

    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    # Load & Clean
    df = pd.read_csv(data_path)
    df['total_sqft'] = df['total_sqft'].apply(clean_total_sqft)
    df = df.dropna(subset=['total_sqft', 'bath', 'location'])
    
    # Apply Feature Engineering Logic
    df = apply_feature_engineering(df)

    # Prepare features (Raw inputs only)
    features = [
        'location', 'total_sqft', 'bath', 'balcony', 'bhk', 
        'area_type_enc', 'is_ready_to_move'
    ]
    X = df[features]
    y = np.log1p(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data ready. Training on {len(X_train)} samples...")

    # Define Top 3 Models
    xgb_pipe = Pipeline([
        ('interactions', InteractionFeatureTransformer()),
        ('encoder', LocationTargetEncoder()),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])

    lgbm_pipe = Pipeline([
        ('interactions', InteractionFeatureTransformer()),
        ('encoder', LocationTargetEncoder()),
        ('scaler', StandardScaler()),
        ('model', LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1))
    ])

    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', xgb_pipe),
            ('lgbm', lgbm_pipe),
            ('rf', Pipeline([
                ('interactions', InteractionFeatureTransformer()),
                ('encoder', LocationTargetEncoder()),
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=50, random_state=42))
            ]))
        ],
        final_estimator=Ridge()
    )

    # Train
    print("Fitting models...")
    xgb_pipe.fit(X_train, y_train)
    lgbm_pipe.fit(X_train, y_train)
    stacking_model.fit(X_train, y_train)

    # Export
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(xgb_pipe, os.path.join(model_dir, 'xgb_model.pkl'))
    joblib.dump(lgbm_pipe, os.path.join(model_dir, 'lgbm_model.pkl'))
    joblib.dump(stacking_model, os.path.join(model_dir, 'stacking_model.pkl'))
    joblib.dump(lgbm_pipe, os.path.join(model_dir, 'bangalore_house_price_model.pkl'))

    # Save metadata (Locations list)
    locations = sorted(df['location'].unique().tolist())
    
    # Calculate Insights for UI Charts
    # 1. Price Distribution (Original scale)
    price_hist, bins = np.histogram(df['price'], bins=10, range=(0, 1000))
    price_dist = [
        {"range": f"{int(bins[i])}-{int(bins[i+1])}L", "count": int(price_hist[i])} 
        for i in range(len(price_hist))
    ]

    # 2. Top Locations by Price (Exclude locations with < 5 properties for stability)
    loc_stats = df.groupby('location')['price'].agg(['mean', 'count']).reset_index()
    top_locs = loc_stats[loc_stats['count'] >= 5].sort_values(by='mean', ascending=False).head(8)
    loc_data = [
        {"location": row['location'], "avg_price": round(row['mean'], 2)} 
        for _, row in top_locs.iterrows()
    ]

    insights = {
        "price_distribution": price_dist,
        "location_insights": loc_data,
        "model_performance": [
            {"name": "XGBoost", "r2": round(xgb_pipe.score(X_test, y_test), 3)},
            {"name": "LightGBM", "r2": round(lgbm_pipe.score(X_test, y_test), 3)},
            {"name": "Stacking", "r2": round(stacking_model.score(X_test, y_test), 3)}
        ]
    }

    import json
    with open(os.path.join(model_dir, 'locations.json'), 'w') as f:
        json.dump(locations, f)
    
    with open(os.path.join(model_dir, 'insights.json'), 'w') as f:
        json.dump(insights, f)

    print(f"Models and metadata saved to {model_dir}")
    print(f"Accuracy (Ensemble): {stacking_model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    run_training()
