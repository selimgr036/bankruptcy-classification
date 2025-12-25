import pandas as pd
import numpy as np
from pathlib import Path

def load_data(data_path=None):
    """Load bankruptcy dataset"""
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "american_bankruptcy.csv"
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    return df

def preprocess_data(df):
    """Basic preprocessing"""
    # Map labels
    df['target'] = df['status_label'].map({'alive': 0, 'failed': 1})
    
    # Get features
    feature_cols = [f'X{i}' for i in range(1, 19)]
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols