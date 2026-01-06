"""
Model definitions for bankruptcy prediction.

This module provides builder functions for machine learning models:
- Baseline: DummyClassifier (majority class)
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM with RBF kernel)

All models are configured with random_state=42 for reproducibility and
include class balancing to handle the imbalanced dataset.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np

# SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE resampling will not be available.")
    print("Install with: pip install imbalanced-learn")


def build_baseline(random_state=42, strategy='most_frequent'):
    """Build a minimal baseline model using DummyClassifier"""
    model = DummyClassifier(
        strategy=strategy,
        random_state=random_state
    )
    return model


def build_logistic_regression(random_state=42, class_weight='balanced', **kwargs):
    """Build a Logistic Regression model"""
    model = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        max_iter=1000,
        **kwargs
    )
    return model


def build_random_forest(random_state=42, n_estimators=100, class_weight='balanced', **kwargs):
    """Build a Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        **kwargs
    )
    return model


def build_xgboost(random_state=42, **kwargs):
    """Build an XGBoost classifier"""
    model_params = {
        'random_state': random_state,
        'seed': random_state,
        'eval_metric': 'logloss',
    }
    
    if 'use_label_encoder' not in kwargs:
        model_params['use_label_encoder'] = False
    
    model_params.update(kwargs)
    model = xgb.XGBClassifier(**model_params)
    return model


def build_svm(random_state=42, kernel='rbf', class_weight='balanced', probability=True, **kwargs):
    """Build a Support Vector Machine classifier with RBF kernel"""
    if 'C' not in kwargs:
        kwargs['C'] = 1.0
    if 'gamma' not in kwargs:
        kwargs['gamma'] = 'scale'
    if 'cache_size' not in kwargs:
        kwargs['cache_size'] = 500
    
    model = SVC(
        kernel=kernel,
        random_state=random_state,
        class_weight=class_weight,
        probability=probability,
        **kwargs
    )
    return model


def get_all_models(random_state=42, xgb_scale_pos_weight=None, include_baseline=True):
    """Get all models configured and ready for training"""
    models = {}
    
    if include_baseline:
        models['Baseline'] = build_baseline(random_state=random_state)
    
    models.update({
        'Logistic Regression': build_logistic_regression(random_state=random_state),
        'Random Forest': build_random_forest(random_state=random_state),
        'XGBoost': build_xgboost(random_state=random_state),
        'SVM': build_svm(random_state=random_state)
    })
    
    if xgb_scale_pos_weight is not None:
        models['XGBoost'].set_params(scale_pos_weight=xgb_scale_pos_weight)
    
    return models


def calculate_xgb_scale_pos_weight(y_train):
    """Calculate scale_pos_weight for XGBoost based on class distribution"""
    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)
    
    if positive_count == 0:
        return 1.0
    
    scale_pos_weight = negative_count / positive_count
    return scale_pos_weight


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes"""
    if not SMOTE_AVAILABLE:
        print("Warning: SMOTE not available. Returning original data.")
        return X_train, y_train
    
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"SMOTE applied: {len(X_train)} -> {len(X_resampled)} samples")
        print(f"  Original - Alive: {np.sum(y_train == 0)}, Failed: {np.sum(y_train == 1)}")
        print(f"  Resampled - Alive: {np.sum(y_resampled == 0)}, Failed: {np.sum(y_resampled == 1)}")
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Warning: SMOTE failed: {str(e)}. Using original data.")
        return X_train, y_train