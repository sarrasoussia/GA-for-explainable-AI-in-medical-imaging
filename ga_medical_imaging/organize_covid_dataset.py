#!/usr/bin/env python3
"""
Organize the COVID-19 Chest X-ray dataset for binary classification.

This script:
1. Reads metadata.csv from the covid-chestxray-dataset
2. Filters for COVID-19 positive cases
3. Filters for "No Finding" (normal/negative) cases
4. Organizes images into binary classification structure (covid/ and no_findings/)
5. Creates symlinks or copies images to the organized structure

Usage:
    python organize_covid_dataset.py \
        --dataset_dir data/covid-chestxray-dataset \
        --output_dir data/covid_chestxray \
        [--copy_images]  # Copy instead of symlink
"""

import os
import pandas as pd
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import sys


def filter_covid_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for COVID-19 positive cases.
    
    Looks for:
    - finding contains "COVID-19" or "COVID"
    - RT_PCR_positive == 'Y' (if available)
    """
    # Filter for COVID-19 cases
    covid_mask = (
        df['finding'].str.contains('COVID-19', case=False, na=False) |
        df['finding'].str.contains('COVID', case=False, na=False)
    )
    
    # Also check RT_PCR_positive if available
    if 'RT_PCR_positive' in df.columns:
        covid_mask = covid_mask | (df['RT_PCR_positive'] == 'Y')
    
    covid_df = df[covid_mask].copy()
    
    print(f"Found {len(covid_df)} COVID-19 positive cases")
    return covid_df


def filter_no_findings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for "No Finding" (normal/negative) cases.
    """
    # Filter for "No Finding" cases
    no_finding_mask = (
        df['finding'].str.contains('No Finding', case=False, na=False) |
        df['finding'].str.contains('Normal', case=False, na=False) |
        df['finding'].str.contains('no finding', case=False, na=False)
    )
    
    no_finding_df = df[no_finding_mask].copy()
    
    print(f"Found {len(no_finding_df)} 'No Finding' cases")
    return no_finding_df


def get_image_path(dataset_dir: str, row: pd.Series) -> str:
    """Get full path to image file."""
    folder = row.get('folder', 'images')
    filename = row.get('filename', '')
    
    # Try different possible locations
    possible_paths = [
        os.path.join(dataset_dir, folder, filename),
        os.path.join(dataset_dir, 'images', filename),
        os.path.join(dataset_dir, filename),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, return the most likely path
    return os.path.join(dataset_dir, 'images', filename)


def organize_dataset(
    dataset_dir: str,
    output_dir: str,
    copy_images: bool = False,
    min_samples_per_class: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Organize the dataset into binary classification structure.
    
    Returns:
        Tuple of (covid_image_paths, no_findings_image_paths)
    """
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print(f"Reading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Total entries in metadata: {len(df)}")
    
    # Filter for COVID-19 cases
    covid_df = filter_covid_cases(df)
    
    # Filter for No Finding cases
    no_findings_df = filter_no_findings(df)
    
    # Check if we have enough samples
    if len(covid_df) < min_samples_per_class:
        print(f"Warning: Only {len(covid_df)} COVID-19 cases found (minimum: {min_samples_per_class})")
    
    if len(no_findings_df) < min_samples_per_class:
        print(f"Warning: Only {len(no_findings_df)} No Finding cases found (minimum: {min_samples_per_class})")
    
    # Create output directories
    covid_dir = os.path.join(output_dir, 'covid')
    no_findings_dir = os.path.join(output_dir, 'no_findings')
    
    os.makedirs(covid_dir, exist_ok=True)
    os.makedirs(no_findings_dir, exist_ok=True)
    
    # Process COVID-19 images
    print(f"\nProcessing COVID-19 images...")
    covid_paths = []
    covid_processed = 0
    covid_missing = 0
    
    for idx, row in covid_df.iterrows():
        src_path = get_image_path(dataset_dir, row)
        
        if not os.path.exists(src_path):
            covid_missing += 1
            continue
        
        # Create destination filename (use original filename or generate one)
        filename = row.get('filename', f'covid_{idx}.jpg')
        # Ensure unique filename
        if filename in covid_paths:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{idx}{ext}"
        
        dst_path = os.path.join(covid_dir, filename)
        
        # Copy or symlink
        if copy_images:
            shutil.copy2(src_path, dst_path)
        else:
            # Create symlink (relative path for portability)
            rel_path = os.path.relpath(src_path, covid_dir)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(rel_path, dst_path)
        
        covid_paths.append(dst_path)
        covid_processed += 1
    
    print(f"  Processed: {covid_processed} images")
    if covid_missing > 0:
        print(f"  Missing: {covid_missing} images")
    
    # Process No Finding images
    print(f"\nProcessing No Finding images...")
    no_findings_paths = []
    no_findings_processed = 0
    no_findings_missing = 0
    
    for idx, row in no_findings_df.iterrows():
        src_path = get_image_path(dataset_dir, row)
        
        if not os.path.exists(src_path):
            no_findings_missing += 1
            continue
        
        # Create destination filename
        filename = row.get('filename', f'no_finding_{idx}.jpg')
        # Ensure unique filename
        if filename in no_findings_paths:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{idx}{ext}"
        
        dst_path = os.path.join(no_findings_dir, filename)
        
        # Copy or symlink
        if copy_images:
            shutil.copy2(src_path, dst_path)
        else:
            # Create symlink (relative path for portability)
            rel_path = os.path.relpath(src_path, no_findings_dir)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(rel_path, dst_path)
        
        no_findings_paths.append(dst_path)
        no_findings_processed += 1
    
    print(f"  Processed: {no_findings_processed} images")
    if no_findings_missing > 0:
        print(f"  Missing: {no_findings_missing} images")
    
    # Summary
    print(f"\n{'='*70}")
    print("DATASET ORGANIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"COVID-19 images: {covid_processed} (in {covid_dir})")
    print(f"No Finding images: {no_findings_processed} (in {no_findings_dir})")
    print(f"Total images: {covid_processed + no_findings_processed}")
    print(f"{'='*70}\n")
    
    return covid_paths, no_findings_paths


def main():
    parser = argparse.ArgumentParser(
        description='Organize COVID-19 Chest X-ray dataset for binary classification'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='data/covid-chestxray-dataset',
        help='Path to the downloaded covid-chestxray-dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/covid_chestxray',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--copy_images',
        action='store_true',
        help='Copy images instead of creating symlinks (uses more disk space)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='Minimum samples per class to proceed (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        print("Please download the dataset first:")
        print("  git clone https://github.com/ieee8023/covid-chestxray-dataset.git data/covid-chestxray-dataset")
        return 1
    
    # Organize dataset
    try:
        covid_paths, no_findings_paths = organize_dataset(
            args.dataset_dir,
            args.output_dir,
            copy_images=args.copy_images,
            min_samples_per_class=args.min_samples
        )
        
        print("✅ Dataset organization successful!")
        print(f"\nYou can now use the dataset with:")
        print(f"  python evaluate_5fold_cv.py --data_dir {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

