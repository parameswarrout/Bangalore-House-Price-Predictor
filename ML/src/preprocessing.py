import pandas as pd
import numpy as np

def clean_total_sqft(x):
    """Handles range values like '1200-1400' and converts to float."""
    tokens = str(x).split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    """Removes outliers based on price per sqft per location."""
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def apply_feature_engineering(df):
    """Applies V2 feature engineering logic."""
    # BHK extraction
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 2)
    
    # Fill balcony
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    
    # Encodings
    df['area_type_enc'] = df['area_type'].map({
        'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3
    }).fillna(0)
    
    df['is_ready_to_move'] = df['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)
    
    # Size-based filtering
    df = df[df.total_sqft/df.bhk >= 300]
    
    # Price per sqft for outlier detection
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    df = remove_pps_outliers(df)
    
    # Interaction Features
    df['sqft_per_room'] = df['total_sqft'] / (df['bhk'] + df['bath'])
    df['room_density'] = df['bhk'] / (df['total_sqft'] / 1000)
    df['bath_to_bhk'] = df['bath'] / df['bhk'].apply(lambda x: max(x, 1))
    df['total_rooms'] = df['bhk'] + df['bath'] + df['balcony']
    
    return df
