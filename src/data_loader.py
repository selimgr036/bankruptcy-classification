import pandas as pd
import numpy as np
from pathlib import Path

def load_bankruptcy_data(data_path=None):
    """Load the bankruptcy dataset"""
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "american_bankruptcy.csv"
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()
    df_processed['target'] = df_processed['status_label'].map({'alive': 0, 'failed': 1})
    
    feature_cols = [f'X{i}' for i in range(1, 19)]
    X = df_processed[feature_cols].values
    y = df_processed['target'].values
    years = df_processed['year'].values
    
    print(f"Preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, years, feature_cols

def train_test_by_year(years, train_end_year=2011, test_start_year=2015, test_end_year=2018):
    """Create temporal train/test split"""
    years = np.array(years)
    
    train_mask = years <= train_end_year
    test_mask = (years >= test_start_year) & (years <= test_end_year)
    val_mask = (years > train_end_year) & (years < test_start_year)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    val_indices = np.where(val_mask)[0]
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, test_indices, val_indices

def load_and_split_data(data_path=None, train_end_year=2011, test_start_year=2015, test_end_year=2018):
    """Complete pipeline: load, preprocess, and split"""
    df = load_bankruptcy_data(data_path)
    X, y, years, feature_names = preprocess_data(df)
    
    train_idx, test_idx, val_idx = train_test_by_year(years, train_end_year, test_start_year, test_end_year)
    
    return {
        'X_train': X[train_idx],
        'X_test': X[test_idx],
        'X_val': X[val_idx],
        'y_train': y[train_idx],
        'y_test': y[test_idx],
        'y_val': y[val_idx],
        'feature_names': feature_names
    }