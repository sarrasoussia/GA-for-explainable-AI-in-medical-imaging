"""
5-Fold Cross-Validation Evaluation Script
Matches the evaluation protocol used in DarkCovidNet and VGG-19 baselines.

Reports: Accuracy, Sensitivity, Specificity, Precision, F1-Score
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, Tuple, List
import argparse

from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.data_utils import MedicalImageDataset, load_dataset_from_directory, create_dummy_dataset
from torchvision import transforms


def calculate_metrics_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics matching baseline reporting.
    
    Returns:
        Dictionary with: accuracy, sensitivity, specificity, precision, f1_score, roc_auc
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate specificity and sensitivity from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Ensure recall == sensitivity
        metrics['recall'] = metrics['sensitivity']
    else:
        metrics['specificity'] = 0.0
        metrics['sensitivity'] = metrics['recall']
    
    # ROC AUC
    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str
) -> nn.Module:
    """Train model for one fold."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_fold(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on test fold."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    metrics = calculate_metrics_comprehensive(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def cross_validate_5fold(
    model_class,
    model_kwargs: dict,
    dataset: MedicalImageDataset,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Perform 5-fold cross-validation.
    
    Returns:
        Dictionary with fold results and summary statistics
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_fold_metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    print(f"\n{'='*70}")
    print(f"5-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of folds: {n_splits}")
    print(f"Epochs per fold: {num_epochs}")
    print(f"{'='*70}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*70}")
        print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Create data loaders for this fold
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        # Split train into train/val (80/20 of train fold)
        val_size = int(len(train_subset) * 0.2)
        train_size = len(train_subset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_subset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Create fresh model for this fold
        model = model_class(**model_kwargs).to(device)
        
        # Train
        print(f"\nTraining fold {fold}...")
        model = train_fold(model, train_loader, val_loader, num_epochs, learning_rate, device)
        
        # Evaluate
        print(f"Evaluating fold {fold}...")
        metrics, y_true, y_pred, y_probs = evaluate_fold(model, test_loader, device)
        
        # Store results
        fold_results.append({
            'fold': fold,
            'metrics': metrics,
            'n_test_samples': len(test_idx)
        })
        
        # Accumulate metrics
        for key in all_fold_metrics:
            all_fold_metrics[key].append(metrics[key])
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
        print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    # Calculate summary statistics
    summary = {}
    for metric_name, values in all_fold_metrics.items():
        summary[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY (5-Fold)")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*70}")
    
    for metric_name in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']:
        stats = summary[metric_name]
        print(f"{metric_name.capitalize():<15} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
              f"{stats['min']:<12.4f} {stats['max']:<12.4f}")
    
    print(f"{'='*70}\n")
    
    return {
        'fold_results': fold_results,
        'summary': summary,
        'n_folds': n_splits
    }


def main():
    parser = argparse.ArgumentParser(
        description='5-Fold Cross-Validation Evaluation'
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
    parser.add_argument('--output_dir', type=str, default='results/5fold_cv',
                       help='Output directory for results')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("Creating dummy dataset...")
        image_paths, labels = create_dummy_dataset(
            num_samples=200,
            image_size=(224, 224),
            output_dir='data/dummy'
        )
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        dataset = MedicalImageDataset(
            image_paths, labels, transform, (224, 224)
        )
    else:
        # Load from directory
        train_loader, val_loader = load_dataset_from_directory(
            args.data_dir,
            image_size=(224, 224)
        )
        # Combine train and val for CV
        dataset = train_loader.dataset
        # Optionally combine with val dataset if needed
        if hasattr(val_loader, 'dataset'):
            # For CV, we'll use the full dataset
            pass
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Model configuration
    model_kwargs = {
        'num_classes': 2,
        'multivector_dim': 8,
        'feature_dim': 128,
        'device': device
    }
    
    # Run 5-fold CV
    results = cross_validate_5fold(
        model_class=GAMedicalClassifier,
        model_kwargs=model_kwargs,
        dataset=dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device,
        n_splits=args.n_splits
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    results_to_save = {
        'n_folds': results['n_folds'],
        'summary': {
            k: {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'min': float(v['min']),
                'max': float(v['max'])
            }
            for k, v in results['summary'].items()
        },
        'fold_results': [
            {
                'fold': r['fold'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()},
                'n_test_samples': r['n_test_samples']
            }
            for r in results['fold_results']
        ]
    }
    
    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}/cv_results.json")
    
    # Print final summary in baseline format
    print("\n" + "="*70)
    print("FINAL SUMMARY (Matching Baseline Format)")
    print("="*70)
    acc = results['summary']['accuracy']
    sens = results['summary']['sensitivity']
    spec = results['summary']['specificity']
    prec = results['summary']['precision']
    f1 = results['summary']['f1_score']
    
    print(f"Accuracy:    {acc['mean']:.4f} ± {acc['std']:.4f} ({acc['mean']*100:.2f}%)")
    print(f"Sensitivity: {sens['mean']:.4f} ± {sens['std']:.4f} ({sens['mean']*100:.2f}%)")
    print(f"Specificity: {spec['mean']:.4f} ± {spec['std']:.4f} ({spec['mean']*100:.2f}%)")
    print(f"Precision:   {prec['mean']:.4f} ± {prec['std']:.4f} ({prec['mean']*100:.2f}%)")
    print(f"F1-Score:    {f1['mean']:.4f} ± {f1['std']:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

