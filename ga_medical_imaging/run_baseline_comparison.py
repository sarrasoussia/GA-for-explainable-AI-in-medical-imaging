#!/usr/bin/env python3
"""
Complete evaluation and comparison workflow.
Runs 5-fold CV and compares with baselines in one command.
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete evaluation and baseline comparison workflow'
    )
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (if None, creates dummy dataset)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--output_base', type=str, default='results',
                       help='Base output directory')
    parser.add_argument('--skip_cv', action='store_true',
                       help='Skip CV evaluation (use existing results)')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip baseline comparison')
    
    args = parser.parse_args()
    
    # Set up paths
    cv_output_dir = os.path.join(args.output_base, '5fold_cv')
    comparison_output_dir = os.path.join(args.output_base, 'comparison')
    cv_results_path = os.path.join(cv_output_dir, 'cv_results.json')
    
    success = True
    
    # Step 1: Run 5-fold CV
    if not args.skip_cv:
        cv_cmd = [
            sys.executable, 'evaluate_5fold_cv.py',
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--device', args.device,
            '--output_dir', cv_output_dir
        ]
        
        if args.data_dir:
            cv_cmd.extend(['--data_dir', args.data_dir])
        
        success = run_command(cv_cmd, "5-Fold Cross-Validation Evaluation")
        
        if not success:
            print("\n❌ Cross-validation failed. Cannot proceed with comparison.")
            return 1
    
    # Step 2: Compare with baselines
    if not args.skip_comparison:
        if not os.path.exists(cv_results_path):
            print(f"\n❌ Error: CV results not found at {cv_results_path}")
            print("   Run 5-fold CV first or check the path.")
            return 1
        
        comparison_cmd = [
            sys.executable, 'compare_with_baselines.py',
            '--cv_results', cv_results_path,
            '--output_dir', comparison_output_dir
        ]
        
        success = run_command(comparison_cmd, "Baseline Comparison")
        
        if not success:
            print("\n⚠️  Baseline comparison failed, but CV results are available.")
            return 1
    
    # Summary
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    if not args.skip_cv:
        print(f"  - CV Results: {cv_output_dir}/cv_results.json")
    if not args.skip_comparison:
        print(f"  - Comparison: {comparison_output_dir}/")
        print(f"    - baseline_comparison.png")
        print(f"    - comparison_report.txt")
        print(f"    - baseline_comparison.json")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

