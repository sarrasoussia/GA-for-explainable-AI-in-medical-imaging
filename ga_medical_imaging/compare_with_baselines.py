"""
Compare GA Model results with reported baseline results from:
1. DarkCovidNet (MDPI review)
2. VGG-19 and other CNNs

Baseline Results:
- DarkCovidNet: accuracy ≈ 98%, sensitivity ≈ 95%, specificity ≈ 95%, 
                precision ≈ 98%, F1 ≈ 0.97 (5-fold CV)
- VGG-19: accuracy ≈ 98.75% (2-class: COVID vs non-COVID)
"""

import json
import os
import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


# Baseline results from literature
BASELINE_RESULTS = {
    'DarkCovidNet': {
        'accuracy': 0.98,
        'sensitivity': 0.95,
        'specificity': 0.95,
        'precision': 0.98,
        'f1_score': 0.97,
        'roc_auc': None,  # Not reported
        'cv_method': '5-fold CV',
        'dataset': 'ieee8023/covid-chestxray-dataset',
        'task': 'Binary classification (COVID-19 vs no-findings)'
    },
    'VGG-19': {
        'accuracy': 0.9875,
        'sensitivity': None,  # Not reported
        'specificity': None,  # Not reported
        'precision': None,  # Not reported
        'f1_score': None,  # Not reported
        'roc_auc': None,  # Not reported
        'cv_method': 'Not specified',
        'dataset': 'ieee8023/covid-chestxray-dataset + internet-sourced',
        'task': '2-class (COVID vs non-COVID)'
    },
    'VGG16 + cGAN (Electronics 2022)': {
        'accuracy': 0.9976,
        'sensitivity': None,  # Not reported
        'specificity': None,  # Not reported
        'precision': None,  # Not reported
        'f1_score': None,  # Not reported
        'roc_auc': None,  # Not reported
        'cv_method': 'Not specified',
        'dataset': 'COVID-19 Radiography Database (Kaggle) + cGAN synthetic images',
        'task': 'Binary classification (COVID-19 vs normal)',
        'note': 'Uses cGAN to generate synthetic COVID-19 images for class balancing'
    }
}


def load_ga_results(cv_results_path: str) -> Dict:
    """Load GA model results from 5-fold CV."""
    with open(cv_results_path, 'r') as f:
        results = json.load(f)
    return results


def format_metric(value: float, std: float = None, as_percent: bool = False) -> str:
    """Format metric for display."""
    if value is None:
        return "N/A"
    
    if as_percent:
        value_str = f"{value*100:.2f}%"
        if std is not None:
            value_str += f" ± {std*100:.2f}%"
    else:
        value_str = f"{value:.4f}"
        if std is not None:
            value_str += f" ± {std:.4f}"
    
    return value_str


def print_comparison_table(ga_results: Dict, baselines: Dict = None):
    """Print formatted comparison table."""
    if baselines is None:
        baselines = BASELINE_RESULTS
    
    print("\n" + "="*90)
    print("COMPARISON WITH BASELINE METHODS")
    print("="*90)
    
    # Get GA summary
    ga_summary = ga_results['summary']
    
    # Metrics to compare
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    
    # Header
    header = f"{'Metric':<20}"
    header += f"{'GA Model':>15}"
    for baseline_name in baselines.keys():
        header += f"{baseline_name:>15}"
    print(header)
    print("-" * 90)
    
    # Rows
    for metric, label in zip(metrics, metric_labels):
        row = f"{label:<20}"
        
        # GA result
        if metric in ga_summary:
            ga_val = ga_summary[metric]['mean']
            ga_std = ga_summary[metric]['std']
            row += f"{format_metric(ga_val, ga_std):>15}"
        else:
            row += f"{'N/A':>15}"
        
        # Baseline results
        for baseline_name, baseline_data in baselines.items():
            if metric in baseline_data and baseline_data[metric] is not None:
                row += f"{format_metric(baseline_data[metric]):>15}"
            else:
                row += f"{'N/A':>15}"
        
        print(row)
    
    print("="*90)
    
    # Additional info
    print("\nEvaluation Protocol:")
    print(f"  GA Model: {ga_results['n_folds']}-fold Cross-Validation")
    for name, data in baselines.items():
        print(f"  {name}: {data.get('cv_method', 'Not specified')}")
    
    print("\nDataset:")
    for name, data in baselines.items():
        print(f"  {name}: {data.get('dataset', 'Not specified')}")


def plot_comparison(ga_results: Dict, baselines: Dict = None, save_path: str = None):
    """Plot comparison charts."""
    if baselines is None:
        baselines = BASELINE_RESULTS
    
    ga_summary = ga_results['summary']
    
    # Prepare data
    models = ['GA Model'] + list(baselines.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        values = []
        errors = []
        labels_list = []
        
        # GA Model
        if metric in ga_summary:
            values.append(ga_summary[metric]['mean'])
            errors.append(ga_summary[metric]['std'])
            labels_list.append('GA Model')
        
        # Baselines
        for name, data in baselines.items():
            if metric in data and data[metric] is not None:
                values.append(data[metric])
                errors.append(0)  # No std reported for baselines
                labels_list.append(name)
        
        # Plot
        x_pos = np.arange(len(values))
        colors = ['#2E86AB'] + ['#A23B72'] * (len(values) - 1)
        
        bars = ax.bar(x_pos, values, yerr=errors if any(e > 0 for e in errors) else None,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                     capsize=5, error_kw={'elinewidth': 2})
        
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            if err > 0:
                label_text = f'{val:.3f}\n±{err:.3f}'
            else:
                label_text = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    plt.suptitle('GA Model vs Baseline Methods - Performance Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_comparison_report(ga_results: Dict, baselines: Dict = None, 
                               output_path: str = None):
    """Generate a comprehensive comparison report."""
    if baselines is None:
        baselines = BASELINE_RESULTS
    
    ga_summary = ga_results['summary']
    
    report = []
    report.append("="*90)
    report.append("COMPREHENSIVE COMPARISON REPORT")
    report.append("="*90)
    report.append("")
    report.append("Geometric Algebra-Based Model vs Baseline Methods")
    report.append("")
    report.append("="*90)
    report.append("")
    
    # Performance metrics
    report.append("PERFORMANCE METRICS COMPARISON")
    report.append("-"*90)
    report.append("")
    
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    
    for metric, label in zip(metrics, metric_labels):
        report.append(f"{label}:")
        
        # GA Model
        if metric in ga_summary:
            ga_val = ga_summary[metric]['mean']
            ga_std = ga_summary[metric]['std']
            report.append(f"  GA Model:        {format_metric(ga_val, ga_std)}")
        
        # Baselines
        for name, data in baselines.items():
            if metric in data and data[metric] is not None:
                report.append(f"  {name:<15} {format_metric(data[metric])}")
        
        report.append("")
    
    # Evaluation protocol
    report.append("EVALUATION PROTOCOL")
    report.append("-"*90)
    report.append(f"  GA Model:        {ga_results['n_folds']}-fold Cross-Validation")
    for name, data in baselines.items():
        report.append(f"  {name:<15} {data.get('cv_method', 'Not specified')}")
    report.append("")
    
    # Dataset info
    report.append("DATASET INFORMATION")
    report.append("-"*90)
    for name, data in baselines.items():
        report.append(f"  {name}:")
        report.append(f"    Dataset: {data.get('dataset', 'Not specified')}")
        report.append(f"    Task:    {data.get('task', 'Not specified')}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS")
    report.append("-"*90)
    
    # Compare accuracy
    ga_acc = ga_summary['accuracy']['mean']
    dark_acc = baselines['DarkCovidNet']['accuracy']
    vgg_acc = baselines['VGG-19']['accuracy']
    
    report.append(f"1. Accuracy Comparison:")
    report.append(f"   - GA Model:      {ga_acc*100:.2f}%")
    report.append(f"   - DarkCovidNet:  {dark_acc*100:.2f}%")
    report.append(f"   - VGG-19:        {vgg_acc*100:.2f}%")
    
    if ga_acc >= dark_acc * 0.95:  # Within 5% of baseline
        report.append(f"   → GA Model achieves comparable accuracy to DarkCovidNet")
    if ga_acc >= vgg_acc * 0.95:
        report.append(f"   → GA Model achieves comparable accuracy to VGG-19")
    
    report.append("")
    
    # Compare other metrics
    if 'sensitivity' in ga_summary:
        ga_sens = ga_summary['sensitivity']['mean']
        dark_sens = baselines['DarkCovidNet']['sensitivity']
        if dark_sens is not None:
            report.append(f"2. Sensitivity Comparison:")
            report.append(f"   - GA Model:      {ga_sens*100:.2f}%")
            report.append(f"   - DarkCovidNet:  {dark_sens*100:.2f}%")
            report.append("")
    
    if 'f1_score' in ga_summary:
        ga_f1 = ga_summary['f1_score']['mean']
        dark_f1 = baselines['DarkCovidNet']['f1_score']
        if dark_f1 is not None:
            report.append(f"3. F1-Score Comparison:")
            report.append(f"   - GA Model:      {ga_f1:.4f}")
            report.append(f"   - DarkCovidNet:  {dark_f1:.4f}")
            report.append("")
    
    report.append("="*90)
    report.append("")
    report.append("Note: GA Model provides intrinsic explainability through geometric")
    report.append("      component analysis, which is not available in baseline methods.")
    report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nComparison report saved to: {output_path}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Compare GA Model with baseline methods'
    )
    parser.add_argument('--cv_results', type=str, required=True,
                       help='Path to 5-fold CV results JSON file')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--baselines_file', type=str, default=None,
                       help='Optional JSON file with custom baseline results')
    
    args = parser.parse_args()
    
    # Load GA results
    print(f"Loading GA model results from: {args.cv_results}")
    ga_results = load_ga_results(args.cv_results)
    
    # Load baselines (use custom if provided, otherwise use defaults)
    if args.baselines_file and os.path.exists(args.baselines_file):
        with open(args.baselines_file, 'r') as f:
            baselines = json.load(f)
    else:
        baselines = BASELINE_RESULTS
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print comparison table
    print_comparison_table(ga_results, baselines)
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, 'baseline_comparison.png')
    plot_comparison(ga_results, baselines, plot_path)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'comparison_report.txt')
    generate_comparison_report(ga_results, baselines, report_path)
    
    # Save JSON comparison
    comparison_json = {
        'ga_model': {
            'summary': {
                k: {'mean': float(v['mean']), 'std': float(v['std'])}
                for k, v in ga_results['summary'].items()
            },
            'n_folds': ga_results['n_folds']
        },
        'baselines': baselines
    }
    
    json_path = os.path.join(args.output_dir, 'baseline_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    
    print(f"\nAll comparison results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

