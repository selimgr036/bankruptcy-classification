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
    """
    Build a minimal baseline model using DummyClassifier.
    
    This baseline always predicts the majority class (most_frequent strategy).
    It serves as a minimal reference point - any meaningful ML model should
    outperform this baseline on the temporal test set.
    
    Args:
        random_state (int): Random seed for reproducibility (default: 42)
        strategy (str): Strategy for DummyClassifier (default: 'most_frequent')
            Options: 'most_frequent', 'stratified', 'uniform', 'constant'
            
    Returns:
        sklearn.dummy.DummyClassifier: Configured baseline model
    """
    model = DummyClassifier(
        strategy=strategy,
        random_state=random_state
    )
    return model


def build_logistic_regression(random_state=42, class_weight='balanced', **kwargs):
    """
    Build a Logistic Regression model.
    
    Logistic Regression is a linear model that works well as a baseline.
    We use class_weight='balanced' to handle the imbalanced dataset.
    StandardScaler should be applied to features before training.
    
    Args:
        random_state (int): Random seed for reproducibility (default: 42)
        class_weight (str or dict): Class weight strategy (default: 'balanced')
        **kwargs: Additional arguments passed to LogisticRegression
        
    Returns:
        sklearn.linear_model.LogisticRegression: Configured model
    """
    model = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        max_iter=1000,  # Ensure convergence
        **kwargs
    )
    return model


def build_random_forest(random_state=42, n_estimators=100, class_weight='balanced', **kwargs):
    """
    Build a Random Forest classifier.
    
    Random Forest is an ensemble of decision trees that handles non-linear
    relationships well. It's robust to feature scaling and can provide
    feature importance scores.
    
    Args:
        random_state (int): Random seed for reproducibility (default: 42)
        n_estimators (int): Number of trees in the forest (default: 100)
        class_weight (str or dict): Class weight strategy (default: 'balanced')
        **kwargs: Additional arguments passed to RandomForestClassifier
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Configured model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,  # Use all available cores
        **kwargs
    )
    return model


def build_xgboost(random_state=42, **kwargs):
    """
    Build an XGBoost classifier.
    
    XGBoost is a gradient boosting framework that often achieves high
    performance. It handles class imbalance via scale_pos_weight parameter.
    Feature scaling is not required for tree-based models.
    
    Args:
        random_state (int): Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments passed to XGBClassifier
        
    Returns:
        xgboost.XGBClassifier: Configured model
    """
    # XGBoost configuration for reproducibility
    # Note: use_label_encoder is deprecated in XGBoost 1.5+ and removed in 2.0+
    # We include it for backward compatibility but it may show deprecation warnings
    # Set both random_state and seed for maximum reproducibility across XGBoost versions
    model_params = {
        'random_state': random_state,
        'seed': random_state,  # Some XGBoost versions also need 'seed' parameter
        'eval_metric': 'logloss',  # For binary classification
    }
    
    # Add use_label_encoder only if not already in kwargs (for older versions)
    if 'use_label_encoder' not in kwargs:
        model_params['use_label_encoder'] = False
    
    model_params.update(kwargs)
    model = xgb.XGBClassifier(**model_params)
    return model


def build_svm(random_state=42, kernel='rbf', class_weight='balanced', probability=True, **kwargs):
    """
    Build a Support Vector Machine classifier with RBF kernel.
    
    SVM with RBF kernel can capture complex non-linear patterns.
    It requires feature scaling (StandardScaler) for optimal performance.
    probability=True enables probability estimates for ROC-AUC calculation.
    
    Note: SVM training time scales O(n²) to O(n³) with dataset size, so it can be
    very slow on large datasets. The main pipeline uses a stratified sample of 20,000
    training instances for SVM to speed up training. However, SVM may still have
    lower performance than tree-based models (Random Forest, XGBoost) on large,
    imbalanced datasets like this one.
    
    Args:
        random_state (int): Random seed for reproducibility (default: 42)
        kernel (str): Kernel type (default: 'rbf')
        class_weight (str or dict): Class weight strategy (default: 'balanced')
        probability (bool): Whether to enable probability estimates (default: True)
        **kwargs: Additional arguments passed to SVC
        
    Returns:
        sklearn.svm.SVC: Configured model
    """
    # Use better default hyperparameters for imbalanced data
    # C controls regularization - lower C = more regularization (less overfitting)
    # For imbalanced data with class_weight='balanced', we want moderate C
    if 'C' not in kwargs:
        kwargs['C'] = 1.0
    if 'gamma' not in kwargs:
        kwargs['gamma'] = 'scale'  # 'scale' is usually better than 'auto'
    if 'cache_size' not in kwargs:
        kwargs['cache_size'] = 500  # Increase cache size for better performance
    
    model = SVC(
        kernel=kernel,
        random_state=random_state,
        class_weight=class_weight,
        probability=probability,
        **kwargs
    )
    return model


def get_all_models(random_state=42, xgb_scale_pos_weight=None, include_baseline=True):
    """
    Get all models configured and ready for training.
    
    This is a convenience function that returns all models in a dictionary.
    Useful for iterating over models in the main pipeline.
    
    Args:
        random_state (int): Random seed for all models (default: 42)
        xgb_scale_pos_weight (float, optional): Scale pos weight for XGBoost.
            If None, will be calculated from training data class distribution.
        include_baseline (bool): Whether to include baseline model (default: True)
            
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models = {}
    
    # Add baseline first if requested
    if include_baseline:
        models['Baseline'] = build_baseline(random_state=random_state)
    
    # Add ML models
    models.update({
        'Logistic Regression': build_logistic_regression(random_state=random_state),
        'Random Forest': build_random_forest(random_state=random_state),
        'XGBoost': build_xgboost(random_state=random_state),
        'SVM': build_svm(random_state=random_state)
    })
    
    # Set scale_pos_weight for XGBoost if provided
    if xgb_scale_pos_weight is not None:
        models['XGBoost'].set_params(scale_pos_weight=xgb_scale_pos_weight)
    
    return models


def calculate_xgb_scale_pos_weight(y_train):
    """
    Calculate scale_pos_weight for XGBoost based on class distribution.
    
    This helps XGBoost handle class imbalance. The formula is:
    scale_pos_weight = (number of negative samples) / (number of positive samples)
    
    Args:
        y_train (np.array): Training target vector
        
    Returns:
        float: Scale pos weight value
    """
    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)
    
    if positive_count == 0:
        return 1.0
    
    scale_pos_weight = negative_count / positive_count
    return scale_pos_weight


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    SMOTE creates synthetic samples of the minority class to balance the dataset.
    This can help models learn better patterns for the minority class.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) or (X_train, y_train) if SMOTE unavailable
    """
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

