import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_and_split_data
from src.models import get_all_models
from src.evaluation import (
    evaluate_model, compute_shap_values, plot_roc_curves, plot_pr_curves,
    plot_feature_importance, plot_shap_summary, plot_confusion_matrices,
    save_metrics
)
from sklearn.preprocessing import StandardScaler

def main():
    print("="*80)
    print("BANKRUPTCY PREDICTION")
    print("="*80)
    
    np.random.seed(42)
    
    # Create directories
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = load_and_split_data()
    
    X_train = data['X_train']
    X_test = data['X_test']
    X_val = data['X_val']
    y_train = data['y_train']
    y_test = data['y_test']
    y_val = data['y_val']
    feature_names = data['feature_names']
    
    # Get models
    models = get_all_models(random_state=42, include_baseline=True)
    
    # Train and evaluate
    all_results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        if model_name == 'Baseline':
            X_tr, X_te, X_v = X_train, X_test, X_val
        elif model_name in ['SVM', 'Logistic Regression']:
            X_tr, X_te, X_v = X_train_scaled, X_test_scaled, X_val_scaled
        else:
            X_tr, X_te, X_v = X_train, X_test, X_val
        
        model.fit(X_tr, y_train)
        results = evaluate_model(model, X_te, y_test, model_name, 
                                optimize_threshold=True, X_val=X_v, y_val=y_val)
        results['y_test'] = y_test
        results['model'] = model
        results['X_test'] = X_te
        all_results[model_name] = results
    
    # SHAP analysis
    print("\nComputing SHAP values...")
    sample_size = min(50, X_test.shape[0])
    sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
    
    for model_name, results in all_results.items():
        if model_name == 'Baseline':
            continue
        
        X_sample = results['X_test'][sample_indices]
        shap_values, explainer, X_sample_used = compute_shap_values(
            results['model'], X_sample, model_name, feature_names
        )
        results['shap_values'] = shap_values
        results['X_sample'] = X_sample_used
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_roc_curves(all_results, "results/figures/roc_curves.png")
    plot_pr_curves(all_results, "results/figures/pr_curves.png")
    plot_confusion_matrices(all_results, "results/figures/confusion_matrices.png")
    
    for model_name, results in all_results.items():
        model = results['model']
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model, feature_names, model_name,
                f"results/figures/feature_importance_{model_name.lower().replace(' ', '_')}.png"
            )
        
        if results.get('shap_values') is not None:
            plot_shap_summary(
                results['shap_values'], results['X_sample'], feature_names, model_name,
                f"results/figures/shap_summary_{model_name.lower().replace(' ', '_')}.png"
            )
    
    # Save metrics
    save_metrics(all_results, "results/metrics/model_metrics.csv")
    print("\nDone!")

if __name__ == "__main__":
    main()