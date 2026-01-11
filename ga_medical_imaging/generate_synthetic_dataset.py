#!/usr/bin/env python3
"""
Generate synthetic COVID-19 images using trained cGAN and add them to the dataset.

This script:
1. Loads a trained cGAN generator
2. Generates synthetic COVID-19 images
3. Adds them to the dataset to balance classes
4. Creates a new balanced dataset directory
"""

import torch
import argparse
import os
import shutil
from pathlib import Path

from ga_medical_imaging.cgan_generator import ConditionalGAN
from ga_medical_imaging.data_utils import load_dataset_from_directory


def balance_dataset_with_synthetic(
    original_data_dir: str,
    output_dir: str,
    cgan_checkpoint: str,
    target_class: int = 1,
    balance_ratio: float = 1.0,  # 1.0 = fully balanced, 0.5 = half way
    img_size: int = 224,
    device: str = 'auto'
):
    """
    Generate synthetic images and create a balanced dataset.
    
    Args:
        original_data_dir: Original dataset directory
        output_dir: Output directory for balanced dataset
        cgan_checkpoint: Path to trained cGAN generator checkpoint
        target_class: Class to generate (1 = COVID-19)
        balance_ratio: Ratio for balancing (1.0 = fully balanced)
        img_size: Image size
        device: Device to use
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("GENERATING BALANCED DATASET WITH SYNTHETIC IMAGES")
    print(f"{'='*70}")
    
    # Load original dataset to count classes
    print(f"\nAnalyzing original dataset: {original_data_dir}")
    train_loader, val_loader = load_dataset_from_directory(
        original_data_dir,
        image_size=(img_size, img_size)
    )
    
    # Count classes in training set
    train_labels = [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
    class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    
    print(f"Original class distribution:")
    print(f"  Class 0 (No Finding): {class_counts[0]} images")
    print(f"  Class 1 (COVID-19): {class_counts[1]} images")
    
    # Calculate how many synthetic images to generate
    minority_class = 0 if class_counts[0] < class_counts[1] else 1
    majority_class = 1 - minority_class
    
    if target_class == minority_class:
        # Generate to balance minority class
        num_synthetic = int((class_counts[majority_class] - class_counts[minority_class]) * balance_ratio)
        print(f"\nGenerating {num_synthetic} synthetic images for class {target_class}")
    else:
        # Generate to balance majority class (less common)
        num_synthetic = int((class_counts[minority_class] - class_counts[majority_class]) * balance_ratio)
        print(f"\nGenerating {num_synthetic} synthetic images for class {target_class}")
    
    # Load cGAN
    print(f"\nLoading cGAN from: {cgan_checkpoint}")
    cgan = ConditionalGAN(
        latent_dim=100,
        num_classes=2,
        img_size=img_size,
        device=device
    )
    cgan.load_generator(cgan_checkpoint)
    
    # Create output directories
    output_covid_dir = os.path.join(output_dir, 'covid')
    output_no_findings_dir = os.path.join(output_dir, 'no_findings')
    os.makedirs(output_covid_dir, exist_ok=True)
    os.makedirs(output_no_findings_dir, exist_ok=True)
    
    # Copy original images
    print(f"\nCopying original images...")
    original_covid_dir = os.path.join(original_data_dir, 'covid')
    original_no_findings_dir = os.path.join(original_data_dir, 'no_findings')
    
    if os.path.exists(original_covid_dir):
        for img_file in os.listdir(original_covid_dir):
            src = os.path.join(original_covid_dir, img_file)
            dst = os.path.join(output_covid_dir, img_file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"  Copied {len(os.listdir(output_covid_dir))} COVID-19 images")
    
    if os.path.exists(original_no_findings_dir):
        for img_file in os.listdir(original_no_findings_dir):
            src = os.path.join(original_no_findings_dir, img_file)
            dst = os.path.join(output_no_findings_dir, img_file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"  Copied {len(os.listdir(output_no_findings_dir))} No Finding images")
    
    # Generate synthetic images
    if num_synthetic > 0:
        print(f"\nGenerating {num_synthetic} synthetic images...")
        cgan.save_synthetic_images(
            num_images=num_synthetic,
            output_dir=output_covid_dir if target_class == 1 else output_no_findings_dir,
            target_class=target_class,
            prefix="synthetic"
        )
    
    # Final statistics
    final_covid_count = len(os.listdir(output_covid_dir)) if os.path.exists(output_covid_dir) else 0
    final_no_findings_count = len(os.listdir(output_no_findings_dir)) if os.path.exists(output_no_findings_dir) else 0
    
    print(f"\n{'='*70}")
    print("BALANCED DATASET CREATED")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Class 0 (No Finding): {final_no_findings_count} images")
    print(f"Class 1 (COVID-19): {final_covid_count} images")
    print(f"Total images: {final_covid_count + final_no_findings_count}")
    print(f"Synthetic images added: {num_synthetic}")
    print(f"{'='*70}\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic images and create balanced dataset'
    )
    parser.add_argument('--original_data_dir', type=str, required=True,
                       help='Original dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for balanced dataset')
    parser.add_argument('--cgan_checkpoint', type=str, required=True,
                       help='Path to trained cGAN generator checkpoint')
    parser.add_argument('--target_class', type=int, default=1,
                       help='Class to generate (1=COVID-19, 0=No Finding)')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                       help='Balance ratio (1.0 = fully balanced)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    balance_dataset_with_synthetic(
        original_data_dir=args.original_data_dir,
        output_dir=args.output_dir,
        cgan_checkpoint=args.cgan_checkpoint,
        target_class=args.target_class,
        balance_ratio=args.balance_ratio,
        img_size=args.img_size,
        device=args.device
    )
    
    print("✅ Balanced dataset created successfully!")
    print(f"\nYou can now use the balanced dataset:")
    print(f"  python evaluate_5fold_cv.py --data_dir {args.output_dir}")


if __name__ == '__main__':
    main()

