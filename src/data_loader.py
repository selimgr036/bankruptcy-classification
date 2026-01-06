"""
Data loading and preprocessing module for bankruptcy prediction.

This module handles loading the bankruptcy dataset, preprocessing features,
and performing temporal train/test splits based on year boundaries.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_bankruptcy_data(data_path=None):
    """Load the bankruptcy dataset from CSV file"""
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "american_bankruptcy.csv"
    
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df


def preprocess_data(df, add_engineered_features=False):
    """Preprocess the bankruptcy dataset"""
    df_processed = df.copy()
    
    df_processed['target'] = df_processed['status_label'].map({'alive': 0, 'failed': 1})
    
    feature_cols = [f'X{i}' for i in range(1, 19)]
    
    missing_cols = [col for col in feature_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    X = df_processed[feature_cols].values
    y = df_processed['target'].values
    years = df_processed['year'].values
    
    if np.isnan(X).any():
        print("Warning: Missing values detected in features.")
        print("  Missing values will be imputed after train/test split using training data only.")
    
    print(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution - Alive (0): {np.sum(y == 0)}, Failed (1): {np.sum(y == 1)}")
    
    return X, y, years, feature_cols


def train_test_by_year(years, train_end_year=2011, test_start_year=2015, test_end_year=2018):
    """Create temporal train/test split based on year boundaries"""
    years = np.array(years)
    
    train_mask = years <= train_end_year
    test_mask = (years >= test_start_year) & (years <= test_end_year)
    val_mask = (years > train_end_year) & (years < test_start_year)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    val_indices = np.where(val_mask)[0]
    
    print(f"Temporal split:")
    print(f"  Training set: {len(train_indices)} samples (years <= {train_end_year})")
    print(f"  Validation set: {len(val_indices)} samples (years {train_end_year+1}-{test_start_year-1})")
    print(f"  Test set: {len(test_indices)} samples (years {test_start_year}-{test_end_year})")
    
    return train_indices, test_indices, val_indices


def impute_missing_values(X_train, X_test, X_val=None, strategy='median'):
    """Impute missing values using training data statistics only"""
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    if X_val is not None:
        X_val_imputed = imputer.transform(X_val)
        return X_train_imputed, X_test_imputed, X_val_imputed, imputer
    else:
        return X_train_imputed, X_test_imputed, imputer


def scale_features(X_train, X_test, X_val=None, scaler=None):
    """Scale features using StandardScaler"""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled, scaler


def load_and_split_data(data_path=None, train_end_year=2011, test_start_year=2015, 
                        test_end_year=2018, scale=False):
    """Complete pipeline: load, preprocess, and split data temporally"""
    df = load_bankruptcy_data(data_path)
    
    X, y, years, feature_names = preprocess_data(df)
    
    train_idx, test_idx, val_idx = train_test_by_year(
        years, train_end_year, test_start_year, test_end_year
    )
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_val = y[val_idx]
    years_train = years[train_idx]
    years_test = years[test_idx]
    years_val = years[val_idx]
    
    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(X_val).any():
        print("Imputing missing values using training data statistics only...")
        X_train, X_test, X_val, imputer = impute_missing_values(X_train, X_test, X_val)
    else:
        imputer = None
    
    scaler = None
    if scale:
        X_train, X_test, X_val, scaler = scale_features(X_train, X_test, X_val)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'years_train': years_train,
        'years_test': years_test,
        'years_val': years_val,
        'feature_names': feature_names,
        'scaler': scaler
    }