"""
Utility functions for the bankruptcy prediction project.

This module provides helper functions for directory management, table formatting,
and other common utilities.
"""

from pathlib import Path
import pandas as pd
from tabulate import tabulate


def ensure_directory(path):
    """Ensure a directory exists, creating it if necessary"""
    Path(path).mkdir(parents=True, exist_ok=True)


def create_results_directories(project_root=None):
    """Create results directory structure if it doesn't exist"""
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)
    
    ensure_directory(project_root / "results" / "metrics")
    ensure_directory(project_root / "results" / "figures")


def format_metrics_table(df_metrics):
    """Format metrics DataFrame as a nice table for terminal output"""
    numeric_cols = df_metrics.select_dtypes(include=['float64', 'int64']).columns
    df_formatted = df_metrics.copy()
    df_formatted[numeric_cols] = df_formatted[numeric_cols].round(4)
    
    table = tabulate(
        df_formatted,
        headers='keys',
        tablefmt='grid',
        showindex=False,
        floatfmt='.4f'
    )
    
    return table


def print_metrics_summary(df_metrics):
    """Print a formatted metrics summary table to terminal"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    if 'threshold_type' in df_metrics.columns:
        df_default = df_metrics[df_metrics['threshold_type'] == 'default'].copy()
        if not df_default.empty:
            df_default = df_default.drop(columns=['threshold_type', 'threshold'], errors='ignore')
            print("\nMetrics at Default Threshold (0.5):")
            print("-" * 80)
            print(format_metrics_table(df_default))
        
        df_optimized = df_metrics[df_metrics['threshold_type'] == 'optimized'].copy()
        if not df_optimized.empty:
            df_opt_display = df_optimized[['model', 'threshold', 'accuracy', 'precision', 'recall', 'f1']].copy()
            print("\nMetrics at Optimized Threshold:")
            print("-" * 80)
            print(format_metrics_table(df_opt_display))
    else:
        print(format_metrics_table(df_metrics))
    
    print("="*80 + "\n")


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def compare_against_baseline(all_results, metric='f1', threshold_type='optimized'):
    """Compare all models against the baseline model"""
    if 'Baseline' not in all_results:
        return {
            'baseline_metric': None,
            'models_beat_baseline': [],
            'models_below_baseline': [],
            'comparison': {},
            'error': 'Baseline model not found in results'
        }
    
    baseline_results = all_results['Baseline']
    baseline_metrics = baseline_results['metrics']
    
    if threshold_type == 'optimized':
        effective_threshold_type = 'optimized'
    else:
        effective_threshold_type = 'default'
    
    baseline_metric = baseline_metrics.get(metric, None)
    
    if baseline_metric is None or (isinstance(baseline_metric, float) and pd.isna(baseline_metric)):
        return {
            'baseline_metric': None,
            'models_beat_baseline': [],
            'models_below_baseline': [],
            'comparison': {},
            'error': f'Baseline {metric} metric not available'
        }
    
    models_beat_baseline = []
    models_below_baseline = []
    comparison = {}
    
    for model_name, results in all_results.items():
        if model_name == 'Baseline':
            continue
        
        if effective_threshold_type == 'optimized' and 'metrics_at_optimal' in results:
            model_metrics = results['metrics_at_optimal']
            default_metrics = results.get('metrics', {})
        else:
            model_metrics = results['metrics']
            default_metrics = None
        
        model_metric = model_metrics.get(metric, None)
        
        if model_metric is None or (isinstance(model_metric, float) and pd.isna(model_metric)):
            comparison[model_name] = {
                'metric_value': None,
                'beats_baseline': False,
                'status': 'N/A (metric unavailable)'
            }
            continue
        
        optimization_warning = None
        if effective_threshold_type == 'optimized' and default_metrics:
            default_metric = default_metrics.get(metric, None)
            if (default_metric is not None and model_metric is not None and 
                not (isinstance(default_metric, float) and pd.isna(default_metric)) and
                not (isinstance(model_metric, float) and pd.isna(model_metric))):
                if model_metric < default_metric:
                    optimization_warning = f"Optimized {metric} ({model_metric:.4f}) < Default {metric} ({default_metric:.4f}) - possible overfitting to validation set"
        
        beats_baseline = model_metric > baseline_metric
        improvement = model_metric - baseline_metric
        if baseline_metric > 0:
            improvement_pct = (improvement / baseline_metric * 100)
        else:
            improvement_pct = float('inf') if improvement > 0 else 0
        
        comparison[model_name] = {
            'metric_value': model_metric,
            'beats_baseline': beats_baseline,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'status': '✓ BEATS BASELINE' if beats_baseline else '✗ Below baseline',
            'optimization_warning': optimization_warning
        }
        
        if beats_baseline:
            models_beat_baseline.append(model_name)
        else:
            models_below_baseline.append(model_name)
    
    return {
        'baseline_metric': baseline_metric,
        'metric_name': metric,
        'threshold_type': threshold_type,
        'effective_threshold_type': effective_threshold_type,
        'models_beat_baseline': models_beat_baseline,
        'models_below_baseline': models_below_baseline,
        'comparison': comparison
    }


def print_baseline_comparison(all_results, metric='f1', threshold_type='optimized'):
    """Print a formatted comparison of all models against the baseline"""
    comparison_result = compare_against_baseline(all_results, metric=metric, threshold_type=threshold_type)
    
    if 'error' in comparison_result:
        print(f"\n⚠ Warning: {comparison_result['error']}")
        return
    
    baseline_metric = comparison_result['baseline_metric']
    metric_name = comparison_result['metric_name'].upper()
    threshold_type = comparison_result['threshold_type']
    effective_threshold_type = comparison_result.get('effective_threshold_type', threshold_type)
    models_beat = comparison_result['models_beat_baseline']
    models_below = comparison_result['models_below_baseline']
    comparison = comparison_result['comparison']
    
    print("\n" + "="*80)
    print(f"BASELINE COMPARISON - {metric_name}")
    if effective_threshold_type == 'optimized':
        print(f"  ML Models: {effective_threshold_type} thresholds | Baseline: default threshold")
    else:
        print(f"  All models: {effective_threshold_type} thresholds")
    print("="*80)
    print(f"\nBaseline {metric_name}: {baseline_metric:.4f} (always predicts majority class)")
    print()
    
    if models_beat:
        print(f"✓ SUCCESS: {len(models_beat)} model(s) beat baseline:")
        print("-" * 80)
        for model_name in models_beat:
            comp = comparison[model_name]
            if comp['improvement_pct'] == float('inf'):
                pct_str = "N/A (baseline=0)"
            else:
                pct_str = f"+{comp['improvement_pct']:.1f}%"
            print(f"  {model_name:25s} {metric_name}: {comp['metric_value']:.4f} "
                  f"(+{comp['improvement']:.4f}, {pct_str})")
            if comp.get('optimization_warning'):
                print(f"    ⚠ Warning: {comp['optimization_warning']}")
        print()
    
    if models_below:
        print(f"✗ {len(models_below)} model(s) below baseline:")
        print("-" * 80)
        for model_name in models_below:
            comp = comparison[model_name]
            if comp['improvement_pct'] == float('inf'):
                pct_str = "N/A (baseline=0)"
            elif comp['improvement_pct'] < 0:
                pct_str = f"{comp['improvement_pct']:.1f}%"
            else:
                pct_str = f"{comp['improvement_pct']:.1f}%"
            print(f"  {model_name:25s} {metric_name}: {comp['metric_value']:.4f} "
                  f"({comp['improvement']:.4f}, {pct_str})")
        print()
    
    total_models = len(comparison)
    success_rate = len(models_beat) / total_models * 100 if total_models > 0 else 0
    print(f"Success Rate: {len(models_beat)}/{total_models} models beat baseline ({success_rate:.1f}%)")
    print("="*80 + "\n")