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
    """
    Load the bankruptcy dataset from CSV file.
    
    Args:
        data_path (str, optional): Path to the CSV file. If None, uses default path.
        
    Returns:
        pd.DataFrame: Loaded dataset with columns including company_name, status_label, year, and features X1-X18.
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "american_bankruptcy.csv"
    
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df


def preprocess_data(df, add_engineered_features=False):
    """
    Preprocess the bankruptcy dataset.
    
    This function:
    - Maps status_label: 'alive' → 0, 'failed' → 1
    - Extracts feature columns (X1-X18)
    - Optionally adds engineered features (ratios, interactions)
    - Returns features, target, and year column separately
    
    NOTE: Missing value imputation should be done AFTER splitting to avoid data leakage.
    This function only does label mapping and feature extraction.
    
    Args:
        df (pd.DataFrame): Raw dataset
        add_engineered_features (bool): Whether to add engineered features (default: False)
        
    Returns:
        tuple: (X, y, years, feature_names) where:
            - X: Feature matrix (numpy array)
            - y: Target vector (numpy array, 0=alive, 1=bankruptcy)
            - years: Year column (numpy array)
            - feature_names: List of feature names
    """
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Map status_label to binary target
    # 'alive' → 0, 'failed' → 1
    df_processed['target'] = df_processed['status_label'].map({'alive': 0, 'failed': 1})
    
    # Extract feature columns (X1 through X18)
    feature_cols = [f'X{i}' for i in range(1, 19)]
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features, target, and year
    X = df_processed[feature_cols].values
    y = df_processed['target'].values
    years = df_processed['year'].values
    
    # REMOVED: Missing value imputation here (was causing data leakage)
    # Missing values should be imputed AFTER splitting, using training data only
    # Check for missing values and warn if found
    if np.isnan(X).any():
        print("Warning: Missing values detected in features.")
        print("  Missing values will be imputed after train/test split using training data only.")
    
    print(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution - Alive (0): {np.sum(y == 0)}, Failed (1): {np.sum(y == 1)}")
    
    return X, y, years, feature_cols


def train_test_by_year(years, train_end_year=2011, test_start_year=2015, test_end_year=2018):
    """
    Create temporal train/test split based on year boundaries.
    
    This function creates indices for temporal splitting where:
    - Training set: years <= train_end_year (1999-2011)
    - Test set: years >= test_start_year and <= test_end_year (2015-2018)
    - Validation set (optional): years between train_end_year and test_start_year (2012-2014)
    
    Args:
        years (np.array): Array of years for each sample
        train_end_year (int): Last year for training set (default: 2011)
        test_start_year (int): First year for test set (default: 2015)
        test_end_year (int): Last year for test set (default: 2018)
        
    Returns:
        tuple: (train_indices, test_indices) as numpy arrays of indices
    """
    years = np.array(years)
    
    # Training set: years <= train_end_year
    train_mask = years <= train_end_year
    train_indices = np.where(train_mask)[0]
    
    # Test set: years >= test_start_year and <= test_end_year
    test_mask = (years >= test_start_year) & (years <= test_end_year)
    test_indices = np.where(test_mask)[0]
    
    # Validation set (years between train_end_year and test_start_year)
    val_mask = (years > train_end_year) & (years < test_start_year)
    val_indices = np.where(val_mask)[0]
    
    print(f"Temporal split:")
    print(f"  Training set: {len(train_indices)} samples (years <= {train_end_year})")
    print(f"  Validation set: {len(val_indices)} samples (years {train_end_year+1}-{test_start_year-1})")
    print(f"  Test set: {len(test_indices)} samples (years {test_start_year}-{test_end_year})")
    
    return train_indices, test_indices, val_indices


def impute_missing_values(X_train, X_test, X_val=None, strategy='median'):
    """
    Impute missing values using training data statistics only.
    
    This function fits an imputer on the training set and transforms
    train, test, and validation sets. This prevents data leakage.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Test features
        X_val (np.array, optional): Validation features
        strategy (str): Imputation strategy - 'median' or 'mean' (default: 'median')
        
    Returns:
        tuple: (X_train_imputed, X_test_imputed, X_val_imputed, imputer) or
               (X_train_imputed, X_test_imputed, imputer) if X_val is None
    """
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
    """
    Scale features using StandardScaler.
    
    This is important for models like SVM and Logistic Regression that are
    sensitive to feature scales. Tree-based models (RF, XGBoost) don't require scaling,
    but we provide this function for consistency.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Test features
        X_val (np.array, optional): Validation features
        scaler (sklearn.preprocessing.StandardScaler, optional): Pre-fitted scaler.
            If None, fits a new scaler on X_train.
            
    Returns:
        tuple: (X_train_scaled, X_test_scaled, X_val_scaled, scaler) or 
               (X_train_scaled, X_test_scaled, scaler) if X_val is None
    """
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
    """
    Complete pipeline: load, preprocess, and split data temporally.
    
    This is the main function to use for data preparation. It combines
    loading, preprocessing, and temporal splitting into one convenient function.
    
    Args:
        data_path (str, optional): Path to CSV file
        train_end_year (int): Last year for training (default: 2011)
        test_start_year (int): First year for test (default: 2015)
        test_end_year (int): Last year for test (default: 2018)
        scale (bool): Whether to scale features (default: False)
        
    Returns:
        dict: Dictionary containing:
            - 'X_train', 'X_test', 'X_val': Feature arrays
            - 'y_train', 'y_test', 'y_val': Target arrays
            - 'years_train', 'years_test', 'years_val': Year arrays
            - 'feature_names': List of feature column names
            - 'scaler': Fitted scaler (if scale=True) or None
    """
    # Load data
    df = load_bankruptcy_data(data_path)
    
    # Preprocess (only label mapping and feature extraction - NO imputation)
    X, y, years, feature_names = preprocess_data(df)
    
    # Temporal split FIRST (before any imputation)
    train_idx, test_idx, val_idx = train_test_by_year(
        years, train_end_year, test_start_year, test_end_year
    )
    
    # Split data
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_val = y[val_idx]
    years_train = years[train_idx]
    years_test = years[test_idx]
    years_val = years[val_idx]
    
    # Impute missing values AFTER splitting (fit on train only)
    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(X_val).any():
        print("Imputing missing values using training data statistics only...")
        X_train, X_test, X_val, imputer = impute_missing_values(X_train, X_test, X_val)
    else:
        imputer = None
    
    # Optional scaling (already correctly implemented - fits on train only)
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

