def compare_against_baseline(all_results, metric='f1'):
    """Compare models against baseline"""
    if 'Baseline' not in all_results:
        return None
    
    baseline_metric = all_results['Baseline']['metrics'].get(metric, 0)
    comparison = {}
    
    for model_name, results in all_results.items():
        if model_name == 'Baseline':
            continue
        
        model_metric = results['metrics'].get(metric, 0)
        beats_baseline = model_metric > baseline_metric
        comparison[model_name] = {
            'metric': model_metric,
            'beats_baseline': beats_baseline
        }
    
    return baseline_metric, comparison

def print_baseline_comparison(all_results, metric='f1'):
    """Print baseline comparison"""
    result = compare_against_baseline(all_results, metric)
    if result is None:
        return
    
    baseline_metric, comparison = result
    print(f"\nBaseline {metric}: {baseline_metric:.4f}")
    print("\nModels that beat baseline:")
    for name, comp in comparison.items():
        if comp['beats_baseline']:
            print(f"  {name}: {comp['metric']:.4f} ✓")
    print("\nModels below baseline:")
    for name, comp in comparison.items():
        if not comp['beats_baseline']:
            print(f"  {name}: {comp['metric']:.4f} ✗")