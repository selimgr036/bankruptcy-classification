"""
Evaluation module for bankruptcy prediction models.

This module provides comprehensive evaluation metrics, SHAP analysis,
and visualization functions for model assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute comprehensive classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            metrics['pr_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    return metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal threshold for binary classification"""
    if y_pred_proba is None:
        return 0.5, {}
    
    thresholds = np.linspace(0.1, 0.9, 41)
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        metrics_thresh = compute_metrics(y_true, y_pred_thresh, y_pred_proba)
        
        if metric == 'f1':
            score = metrics_thresh['f1']
        elif metric == 'precision':
            score = metrics_thresh['precision']
        elif metric == 'recall':
            score = metrics_thresh['recall']
        elif metric == 'f1_precision_recall_balance':
            score = metrics_thresh['f1'] * (1 - abs(metrics_thresh['precision'] - metrics_thresh['recall']))
        else:
            score = metrics_thresh['f1']
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics_thresh.copy()
            best_metrics['threshold'] = threshold
    
    return best_threshold, best_metrics


def evaluate_model(model, X_test, y_test, model_name='Model', optimize_threshold=False, 
                   X_val=None, y_val=None):
    """Evaluate a trained model on test data"""
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    result = {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }
    
    if optimize_threshold and y_pred_proba is not None:
        if X_val is not None and y_val is not None:
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_val_proba = None
            
            if y_val_proba is not None:
                optimal_threshold, _ = find_optimal_threshold(
                    y_val, y_val_proba, metric='f1_precision_recall_balance'
                )
                y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
                metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_pred_proba)
                metrics_optimal['threshold'] = optimal_threshold
                cm_optimal = confusion_matrix(y_test, y_pred_optimal)
                
                default_f1 = metrics['f1']
                optimized_f1 = metrics_optimal['f1']
                optimized_thresh_value = optimal_threshold
                
                if optimized_f1 < default_f1:
                    metrics_optimal = metrics.copy()
                    metrics_optimal['threshold'] = 0.5
                    y_pred_optimal = y_pred
                    cm_optimal = cm
                    optimal_threshold = 0.5
                    result['threshold_optimization_warning'] = (
                        f"Optimized threshold ({optimized_thresh_value:.3f}) performed worse "
                        f"than default (0.5) on test set. Using default threshold."
                    )
                
                result['optimal_threshold'] = optimal_threshold
                result['metrics_at_optimal'] = metrics_optimal
                result['y_pred_optimal'] = y_pred_optimal
                result['confusion_matrix_optimal'] = cm_optimal
        else:
            optimal_threshold, metrics_optimal = find_optimal_threshold(
                y_test, y_pred_proba, metric='f1_precision_recall_balance'
            )
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            cm_optimal = confusion_matrix(y_test, y_pred_optimal)
            
            result['optimal_threshold'] = optimal_threshold
            result['metrics_at_optimal'] = metrics_optimal
            result['y_pred_optimal'] = y_pred_optimal
            result['confusion_matrix_optimal'] = cm_optimal
    
    return result


def compute_shap_values(model, X_sample, model_name='Model', feature_names=None):
    """Compute SHAP values for model interpretability"""
    np.random.seed(42)
    
    try:
        model_type = type(model).__name__
        X_sample_used = X_sample.copy()
        
        if 'RandomForest' in model_type or 'XGB' in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_used)
            
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values) == 1:
                    shap_values = shap_values[0]
            
            shap_values = np.array(shap_values)
        
        elif model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, X_sample_used)
            shap_values = explainer.shap_values(X_sample_used)
            
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values) == 1:
                    shap_values = shap_values[0]
            
            shap_values = np.array(shap_values)
        
        elif model_name == 'SVM':
            background_size = min(30, X_sample_used.shape[0])
            np.random.seed(42)
            bg_indices = np.random.choice(X_sample_used.shape[0], background_size, replace=False)
            X_background = X_sample_used[bg_indices]
            
            explainer = shap.PermutationExplainer(model.predict_proba, X_background)
            explain_size = min(20, X_sample_used.shape[0])
            X_explain = X_sample_used[:explain_size]
            shap_values = explainer.shap_values(X_explain)
            
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values) == 1:
                    shap_values = shap_values[0]
            
            shap_values = np.array(shap_values)
            X_sample_used = X_explain
                
        else:
            if X_sample_used.shape[0] > 100:
                sample_size = min(100, X_sample_used.shape[0])
                np.random.seed(42)
                sample_indices = np.random.choice(X_sample_used.shape[0], sample_size, replace=False)
                X_sample_used = X_sample_used[sample_indices]
            
            explainer = shap.KernelExplainer(model.predict_proba, X_sample_used)
            shap_values = explainer.shap_values(X_sample_used)
            
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values) == 1:
                    shap_values = shap_values[0]
            
            shap_values = np.array(shap_values)
        
        return shap_values, explainer, X_sample_used
        
    except Exception as e:
        print(f"Warning: SHAP computation failed for {model_name}: {str(e)}")
        return None, None, None


def plot_roc_curves(results_dict, save_path=None):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        y_test = results['y_test']
        y_pred_proba = results.get('y_pred_proba')
        
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results['metrics'].get('roc_auc', np.nan)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Bankruptcy Prediction Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.close()


def plot_pr_curves(results_dict, save_path=None):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        y_test = results['y_test']
        y_pred_proba = results.get('y_pred_proba')
        
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = results['metrics'].get('pr_auc', np.nan)
            plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Bankruptcy Prediction Models', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curves to {save_path}")
    
    plt.close()


def plot_feature_importance(model, feature_names, model_name='Model', save_path=None, top_n=15):
    """Plot feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Warning: {model_name} does not support feature importance extraction.")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.close()


def plot_shap_summary(shap_values, X_sample, feature_names, model_name='Model', save_path=None):
    """Plot SHAP summary plot"""
    if shap_values is None:
        print(f"Warning: No SHAP values available for {model_name}.")
        return
    
    try:
        shap_values = np.array(shap_values)
        X_sample = np.array(X_sample)
        
        if feature_names is not None:
            if isinstance(feature_names, (pd.Index, pd.Series)):
                feature_names = [str(f) for f in feature_names.tolist()]
            elif isinstance(feature_names, np.ndarray):
                feature_names = [str(f) for f in feature_names.tolist()]
            elif hasattr(feature_names, '__iter__') and not isinstance(feature_names, str):
                feature_names = [str(f) for f in feature_names]
            else:
                feature_names = [str(f) for f in feature_names] if feature_names else None
        
        if shap_values.ndim == 1:
            if shap_values.shape[0] == X_sample.shape[1]:
                shap_values = shap_values.reshape(1, -1)
                X_sample = X_sample[:1]
            elif shap_values.shape[0] == X_sample.shape[0] * X_sample.shape[1]:
                shap_values = shap_values.reshape(X_sample.shape[0], X_sample.shape[1])
            else:
                print(f"Warning: Cannot interpret 1D SHAP values shape {shap_values.shape} for {model_name}")
                return
        
        if shap_values.shape[0] != X_sample.shape[0]:
            min_samples = min(shap_values.shape[0], X_sample.shape[0])
            shap_values = shap_values[:min_samples]
            X_sample = X_sample[:min_samples]
        
        if shap_values.shape[1] != X_sample.shape[1]:
            min_features = min(shap_values.shape[1], X_sample.shape[1])
            shap_values = shap_values[:, :min_features]
            X_sample = X_sample[:, :min_features]
            if feature_names is not None:
                feature_names = feature_names[:min_features]
        
        if feature_names is not None:
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            elif not isinstance(feature_names, list):
                feature_names = list(feature_names)
            feature_names = [str(f) for f in feature_names]
        
        plt.figure(figsize=(10, 8))
        try:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        except TypeError as e:
            if "only integer scalar arrays" in str(e):
                feature_names_array = np.array(feature_names, dtype=object)
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names_array, show=False)
            else:
                raise
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to create SHAP plot for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_confusion_matrices(results_dict, save_path=None):
    """Plot confusion matrices for all models"""
    n_models = len(results_dict)
    
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Alive', 'Bankruptcy'],
                   yticklabels=['Alive', 'Bankruptcy'])
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path}")
    
    plt.close()


def save_metrics(results_dict, save_path, include_optimized=True):
    """Save all model metrics to a CSV file"""
    metrics_list = []
    
    for model_name, results in results_dict.items():
        metrics = results['metrics'].copy()
        metrics['model'] = model_name
        metrics['threshold'] = 0.5
        metrics['threshold_type'] = 'default'
        metrics_list.append(metrics)
        
        if include_optimized and 'metrics_at_optimal' in results:
            opt_metrics = results['metrics_at_optimal'].copy()
            opt_metrics['model'] = model_name
            opt_metrics['threshold_type'] = 'optimized'
            metrics_list.append(opt_metrics)
    
    df_metrics = pd.DataFrame(metrics_list)
    
    priority_cols = ['model', 'threshold_type', 'threshold']
    other_cols = [col for col in df_metrics.columns if col not in priority_cols]
    cols = priority_cols + other_cols
    df_metrics = df_metrics[cols]
    
    df_metrics.to_csv(save_path, index=False)
    print(f"Saved metrics to {save_path}")
    
    return df_metrics