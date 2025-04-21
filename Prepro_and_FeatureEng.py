# Prepro_and_FeatureEng.py

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

# Load dataset
def load_data():
    cali = fetch_california_housing()
    df = pd.DataFrame(data=cali.data, columns=cali.feature_names)
    df['Target'] = cali.target
    return df

# Handle missing values (if any)
def handle_missing_values(df):
    if df.isnull().sum().any():
        df = df.dropna()
    return df

# Remove outliers using IQR method
def remove_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Feature engineering (with proxy for Households)
def create_new_features(df):
    # Approximate households using Population / AveOccup
    df['Households_est'] = df['Population'] / df['AveOccup']

    # Derived features
    df['AveRoomsPerHousehold'] = df['AveRooms'] / df['Households_est']
    df['PopulationPerHousehold'] = df['Population'] / df['Households_est']
    df['AveBedrmsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    
    return df

# Feature scaling
def scale_features(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Save preprocessed data
def save_data(df, path='processed_california_housing.csv'):
    df.to_csv(path, index=False)
    print(f"Preprocessed data saved to '{path}'")

# Main preprocessing pipeline
def preprocess_pipeline():
    df = load_data()
    print("data Loaded data with shape:", df.shape)
    
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df)
    df = create_new_features(df)
    df = scale_features(df)

    save_data(df)
    return df

# Run when executed as a script
if __name__ == "__main__":
    final_df = preprocess_pipeline()
