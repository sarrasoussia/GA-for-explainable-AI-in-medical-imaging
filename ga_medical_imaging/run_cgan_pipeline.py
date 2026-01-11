#!/usr/bin/env python3
"""
Complete pipeline for cGAN-based data augmentation and evaluation.

This script automates the entire workflow:
1. Train cGAN
2. Generate synthetic images
3. Create balanced dataset
4. Run 5-fold CV evaluation
5. Compare with baselines

Usage:
    python run_cgan_pipeline.py --data_dir data/covid_chestxray
"""

import os
import sys
import subprocess
import argparse
import time


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, check=check)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n❌ {description} failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Complete cGAN augmentation and evaluation pipeline'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Original dataset directory')
    parser.add_argument('--skip_cgan_training', action='store_true',
                       help='Skip cGAN training (use existing checkpoint)')
    parser.add_argument('--cgan_checkpoint', type=str, default=None,
                       help='Path to existing cGAN checkpoint')
    parser.add_argument('--cgan_epochs', type=int, default=100,
                       help='Number of cGAN training epochs')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                       help='Balance ratio for synthetic images (1.0 = fully balanced)')
    parser.add_argument('--cv_epochs', type=int, default=30,
                       help='Number of epochs for CV evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--output_base', type=str, default='results',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Set up paths
    cgan_checkpoint_dir = 'checkpoints/cgan'
    cgan_final_checkpoint = os.path.join(cgan_checkpoint_dir, 'cgan_final.pth')
    balanced_data_dir = args.data_dir + '_balanced'
    cv_output_dir = os.path.join(args.output_base, '5fold_cv_balanced')
    comparison_output_dir = os.path.join(args.output_base, 'comparison_balanced')
    
    print(f"\n{'='*70}")
    print("CGAN AUGMENTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Original dataset: {args.data_dir}")
    print(f"Balanced dataset: {balanced_data_dir}")
    print(f"Output directory: {args.output_base}")
    print(f"{'='*70}\n")
    
    # Step 1: Train cGAN (if not skipped)
    if not args.skip_cgan_training:
        if args.cgan_checkpoint:
            cgan_checkpoint_path = args.cgan_checkpoint
        else:
            cgan_checkpoint_path = cgan_final_checkpoint
            
            train_cmd = [
                sys.executable, 'train_cgan.py',
                '--data_dir', args.data_dir,
                '--num_epochs', str(args.cgan_epochs),
                '--device', args.device,
                '--save_dir', cgan_checkpoint_dir
            ]
            
            if not run_command(train_cmd, "Step 1: Training cGAN", check=False):
                print("\n⚠️  cGAN training failed. You can:")
                print("  1. Check the error messages above")
                print("  2. Use --skip_cgan_training and provide --cgan_checkpoint")
                return 1
    else:
        if args.cgan_checkpoint:
            cgan_checkpoint_path = args.cgan_checkpoint
        else:
            cgan_checkpoint_path = cgan_final_checkpoint
        
        if not os.path.exists(cgan_checkpoint_path):
            print(f"\n❌ Error: cGAN checkpoint not found: {cgan_checkpoint_path}")
            print("   Please train cGAN first or provide --cgan_checkpoint")
            return 1
        
        print(f"\n✓ Using existing cGAN checkpoint: {cgan_checkpoint_path}")
    
    # Step 2: Generate synthetic images and create balanced dataset
    generate_cmd = [
        sys.executable, 'generate_synthetic_dataset.py',
        '--original_data_dir', args.data_dir,
        '--output_dir', balanced_data_dir,
        '--cgan_checkpoint', cgan_checkpoint_path,
        '--target_class', '1',
        '--balance_ratio', str(args.balance_ratio),
        '--device', args.device
    ]
    
    if not run_command(generate_cmd, "Step 2: Generating synthetic images", check=False):
        print("\n❌ Synthetic image generation failed")
        return 1
    
    # Step 3: Run 5-fold CV evaluation
    cv_cmd = [
        sys.executable, 'evaluate_5fold_cv.py',
        '--data_dir', balanced_data_dir,
        '--num_epochs', str(args.cv_epochs),
        '--device', args.device,
        '--output_dir', cv_output_dir
    ]
    
    if not run_command(cv_cmd, "Step 3: 5-Fold Cross-Validation", check=False):
        print("\n⚠️  CV evaluation failed, but balanced dataset is available")
        return 1
    
    # Step 4: Compare with baselines
    cv_results_path = os.path.join(cv_output_dir, 'cv_results.json')
    if os.path.exists(cv_results_path):
        compare_cmd = [
            sys.executable, 'compare_with_baselines.py',
            '--cv_results', cv_results_path,
            '--output_dir', comparison_output_dir
        ]
        
        run_command(compare_cmd, "Step 4: Baseline Comparison", check=False)
    else:
        print(f"\n⚠️  CV results not found: {cv_results_path}")
        print("   Skipping baseline comparison")
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  - Balanced dataset: {balanced_data_dir}")
    print(f"  - CV results: {cv_output_dir}/cv_results.json")
    print(f"  - Comparison: {comparison_output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review CV results: {cv_output_dir}/cv_results.json")
    print(f"  2. Check comparison: {comparison_output_dir}/comparison_report.txt")
    print(f"  3. Compare with non-augmented results")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

