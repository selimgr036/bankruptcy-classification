#!/usr/bin/env python3
"""
Main entry point for the bankruptcy prediction project.

This script orchestrates the complete ML pipeline:
1. Load and preprocess data
2. Perform temporal train/test split
3. Train baseline model (DummyClassifier - majority class) and four ML models
   (Logistic Regression, Random Forest, XGBoost, SVM)
4. Evaluate models with comprehensive metrics
5. Compare all models against baseline (SUCCESS = models beat baseline on temporal test set)
6. Generate visualizations (ROC, PR curves, SHAP, feature importance)
7. Save results to results/ directory
"""

import argparse
import json
import numpy as np
import warnings
# Filter only specific non-critical warnings
# Allow important warnings (SHAP, convergence, deprecations) to show
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib', message='.*findfont.*')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib', message='.*Glyph.*')

# Import project modules
from src.data_loader import load_and_split_data
from src.models import get_all_models, calculate_xgb_scale_pos_weight, apply_smote
from src.evaluation import (
    evaluate_model, compute_shap_values, plot_roc_curves, plot_pr_curves,
    plot_feature_importance, plot_shap_summary, plot_confusion_matrices,
    save_metrics
)
from src.utils import create_results_directories, print_metrics_summary, get_project_root, print_baseline_comparison


def main(use_smote=True, optimize_thresholds=True):
    """
    Main pipeline execution.
    
    Args:
        use_smote (bool): Whether to use SMOTE for class balancing (default: True)
        optimize_thresholds (bool): Whether to find optimal thresholds for each model (default: True)
    """
    print("="*80)
    print("BANKRUPTCY PREDICTION WITH TEMPORAL VALIDATION")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  - SMOTE resampling: {'Enabled' if use_smote else 'Disabled'}")
    print(f"  - Threshold optimization: {'Enabled' if optimize_thresholds else 'Disabled'}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get project root and create results directories
    project_root = get_project_root()
    create_results_directories(project_root)
    
    results_dir = project_root / "results"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    
    # ========================================================================
    # STEP 1: Load and preprocess data with temporal split
    # ========================================================================
    print("STEP 1: Loading and preprocessing data...")
    print("-" * 80)
    
    # Temporal split: Train (1999-2011), Validation (2012-2014), Test (2015-2018)
    # Validation set is used for threshold optimization to avoid overfitting to test set
    data = load_and_split_data(
        train_end_year=2011,
        test_start_year=2015,
        test_end_year=2018,
        scale=False  # We'll scale per model as needed
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    X_val = data['X_val']
    y_train = data['y_train']
    y_test = data['y_test']
    y_val = data['y_val']
    feature_names = data['feature_names']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()
    
    # ========================================================================
    # STEP 2: Prepare models
    # ========================================================================
    print("STEP 2: Preparing models...")
    print("-" * 80)
    
    # Apply SMOTE if requested
    if use_smote:
        print("Applying SMOTE for class balancing...")
        X_train, y_train = apply_smote(X_train, y_train, random_state=42)
        print()
    
    # Calculate XGBoost scale_pos_weight for class balancing
    xgb_scale_pos_weight = calculate_xgb_scale_pos_weight(y_train)
    
    # Scale features once for models that need it (SVM and Logistic Regression)
    # This is done outside the model loop to avoid recomputing scaling multiple times
    from src.data_loader import scale_features
    print("Scaling features for models that require it (Logistic Regression, SVM)...")
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(X_train, X_test, X_val)
    print()
    
    # Get all models (includes Baseline and SVM by default)
    models = get_all_models(random_state=42, xgb_scale_pos_weight=xgb_scale_pos_weight, include_baseline=True)
    
    print(f"Models to train: {list(models.keys())}")
    print("Note: Baseline is a minimal reference (majority class). Success = models beat baseline on temporal test set.")
    print()
    
    # ========================================================================
    # STEP 3: Train and evaluate models
    # ========================================================================
    print("STEP 3: Training and evaluating models...")
    print("-" * 80)
    
    # Dictionary to store all results
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Select appropriate data (scaled or unscaled) based on model requirements
        if model_name == 'Baseline':
            # Baseline doesn't need scaling or special handling
            X_train_model = X_train
            X_test_model = X_test
            X_val_model = X_val
        elif model_name in ['SVM', 'Logistic Regression']:
            # Use pre-scaled features (computed once outside the loop)
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
            X_val_model = X_val_scaled
        else:
            # Tree-based models (Random Forest, XGBoost) don't need scaling
            X_train_model = X_train
            X_test_model = X_test
            X_val_model = X_val
        
        # For SVM, use a stratified sample of training data to speed up training
        # SVM training time scales O(n²) to O(n³), so using a sample is necessary
        # We use stratified sampling to maintain class balance
        if model_name == 'SVM':
            from sklearn.model_selection import train_test_split
            # Use larger sample (20k) with stratified sampling for better performance
            sample_size = min(20000, X_train_model.shape[0])
            # Stratified split to maintain class distribution
            X_train_svm, _, y_train_svm, _ = train_test_split(
                X_train_model, y_train,
                train_size=sample_size,
                stratify=y_train,
                random_state=42
            )
            print(f"  Using {len(X_train_svm):,} samples for SVM training (out of {X_train_model.shape[0]:,}) to speed up training")
            print(f"  Class distribution in sample - Alive: {np.sum(y_train_svm == 0)}, Failed: {np.sum(y_train_svm == 1)}")
            model.fit(X_train_svm, y_train_svm)
        else:
            # Train model
            model.fit(X_train_model, y_train)
        print(f"  ✓ {model_name} trained successfully")
        
        # Evaluate model with threshold optimization on validation set
        # Skip threshold optimization for baseline (always predicts same class)
        should_optimize = optimize_thresholds and model_name != 'Baseline'
        if should_optimize:
            print(f"  Optimizing threshold on validation set...")
        results = evaluate_model(
            model, X_test_model, y_test, model_name, 
            optimize_threshold=should_optimize,
            X_val=X_val_model, y_val=y_val
        )
        results['y_test'] = y_test  # Store for plotting
        results['model'] = model  # Store model for SHAP
        results['X_test'] = X_test_model  # Store test data for SHAP
        
        all_results[model_name] = results
        
        # Print metrics at default threshold
        metrics = results['metrics']
        print(f"  Metrics (threshold=0.5):")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        if not np.isnan(metrics['roc_auc']):
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        if not np.isnan(metrics['pr_auc']):
            print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
        
        # Print optimized metrics if available
        if optimize_thresholds and 'metrics_at_optimal' in results:
            opt_metrics = results['metrics_at_optimal']
            opt_thresh = results['optimal_threshold']
            
            # Check for threshold optimization warning
            if 'threshold_optimization_warning' in results:
                print(f"  ⚠ {results['threshold_optimization_warning']}")
            
            print(f"  Metrics (optimized threshold={opt_thresh:.3f}):")
            print(f"    Accuracy: {opt_metrics['accuracy']:.4f}")
            print(f"    Precision: {opt_metrics['precision']:.4f}")
            print(f"    Recall: {opt_metrics['recall']:.4f}")
            print(f"    F1-Score: {opt_metrics['f1']:.4f}")
    
    print()
    
    # ========================================================================
    # STEP 3.5: Save thresholds to disk for reproducibility
    # ========================================================================
    print("Saving thresholds to disk...")
    thresholds_dict = {}
    for model_name, results in all_results.items():
        # Get optimal threshold if available, otherwise use default 0.5
        optimal_threshold = results.get('optimal_threshold', 0.5)
        thresholds_dict[model_name] = optimal_threshold
    
    # Save thresholds to JSON file
    thresholds_path = metrics_dir / "thresholds.json"
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"  ✓ Saved thresholds to {thresholds_path}")
    print()
    
    # ========================================================================
    # STEP 4: Compute SHAP values for interpretability
    # ========================================================================
    print("STEP 4: Computing SHAP values for model interpretability...")
    print("-" * 80)
    
    # Use a smaller sample of test data for SHAP (for efficiency)
    # Reduced from 100 to 50 to speed up computation
    # Get the number of test samples (same for all models, regardless of scaling)
    n_test = all_results[next(iter(all_results))]["X_test"].shape[0]
    sample_size = min(50, n_test)
    np.random.seed(42)
    sample_indices = np.random.choice(n_test, sample_size, replace=False)
    
    for model_name, results in all_results.items():
        # Skip SHAP for baseline (not meaningful)
        if model_name == 'Baseline':
            results['shap_values'] = None
            results['shap_explainer'] = None
            results['X_sample'] = None
            continue
            
        print(f"Computing SHAP values for {model_name}...")
        model = results['model']
        X_test_model = results['X_test']
        
        # Get sample from the test data used for this model (already scaled if needed)
        X_sample = X_test_model[sample_indices]
        
        shap_values, explainer, X_sample_used = compute_shap_values(
            model, X_sample, model_name, feature_names
        )
        
        if shap_values is not None:
            results['shap_values'] = shap_values
            results['shap_explainer'] = explainer
            results['X_sample'] = X_sample_used  # Use the actual sample that was explained
            print(f"  ✓ SHAP values computed for {model_name}")
        else:
            results['shap_values'] = None
            results['shap_explainer'] = None
            results['X_sample'] = None
            print(f"  ⚠ SHAP computation failed for {model_name}")
    
    print()
    
    # ========================================================================
    # STEP 5: Generate visualizations
    # ========================================================================
    print("STEP 5: Generating visualizations...")
    print("-" * 80)
    
    # ROC curves
    print("  Generating ROC curves...")
    plot_roc_curves(all_results, save_path=figures_dir / "roc_curves.png")
    
    # Precision-Recall curves
    print("  Generating Precision-Recall curves...")
    plot_pr_curves(all_results, save_path=figures_dir / "pr_curves.png")
    
    # Confusion matrices - default threshold
    print("  Generating confusion matrices (default threshold)...")
    plot_confusion_matrices(all_results, save_path=figures_dir / "confusion_matrices_default.png", threshold_type="default")
    
    # Confusion matrices - optimized threshold
    print("  Generating confusion matrices (optimized threshold)...")
    plot_confusion_matrices(all_results, save_path=figures_dir / "confusion_matrices_optimized.png", threshold_type="optimized")
    
    # Feature importance plots (for tree-based models)
    print("  Generating feature importance plots...")
    for model_name, results in all_results.items():
        model = results['model']
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model, feature_names, model_name,
                save_path=figures_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
            )
    
    # SHAP summary plots
    print("  Generating SHAP summary plots...")
    for model_name, results in all_results.items():
        shap_values = results.get('shap_values')
        X_sample = results.get('X_sample')
        if shap_values is not None and X_sample is not None:
            plot_shap_summary(
                shap_values, X_sample, feature_names, model_name,
                save_path=figures_dir / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
            )
    
    print()
    
    # ========================================================================
    # STEP 6: Save metrics
    # ========================================================================
    print("STEP 6: Saving metrics...")
    print("-" * 80)
    
    df_metrics = save_metrics(all_results, save_path=metrics_dir / "model_metrics.csv")
    
    # ========================================================================
    # STEP 7: Print summary and baseline comparison
    # ========================================================================
    print_metrics_summary(df_metrics)
    
    # Compare all models against baseline (focus on F1 score with optimized threshold)
    print_baseline_comparison(all_results, metric='f1', threshold_type='optimized')
    
    # Also compare using PR-AUC (better for imbalanced datasets, baseline has meaningful non-zero value)
    print_baseline_comparison(all_results, metric='pr_auc', threshold_type='optimized')
    
    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Metrics: {metrics_dir}")
    print(f"  - Figures: {figures_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bankruptcy prediction ML pipeline with temporal validation"
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE for class balancing (default: SMOTE is enabled)"
    )
    parser.add_argument(
        "--no-threshold-opt",
        action="store_true",
        help="Disable threshold optimization (default: threshold optimization is enabled)"
    )
    
    args = parser.parse_args()
    
    main(
        use_smote=not args.no_smote,
        optimize_thresholds=not args.no_threshold_opt
    )
