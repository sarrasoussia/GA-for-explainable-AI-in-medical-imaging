"""
Script to compare GA model against baseline methods.
Implements the experimental framework described in RESEARCH_REPORT.md Section 5.1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.data_utils import load_dataset_from_directory, MedicalImageDataset


class BaselineResNet(nn.Module):
    """ResNet baseline for comparison."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class BaselineEfficientNet(nn.Module):
    """EfficientNet baseline for comparison."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier[1] = nn.Linear(
                self.backbone.classifier[1].in_features, num_classes
            )
        except ImportError:
            raise ImportError("EfficientNet requires torchvision >= 0.13.0")
    
    def forward(self, x):
        return self.backbone(x)


def evaluate_model(model, dataloader, device, model_name="Model"):
    """
    Evaluate a model and return comprehensive metrics.
    
    Returns:
        dict: Metrics including accuracy, precision, recall, F1, AUC-ROC, AUC-PR
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    # Calculate AUC-PR
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
    except:
        metrics['auc_pr'] = 0.0
    
    return metrics, all_probs, all_labels


def train_baseline_model(model, train_loader, val_loader, num_epochs, device, model_name):
    """Train a baseline model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    
    print(f"\nTraining {model_name}...")
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Val Acc = {val_acc:.2f}%")
    
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    return model


def plot_comparison(results, output_dir):
    """Plot comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Bar chart comparison
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for model_name in models:
        if 'probs' in results[model_name] and 'labels' in results[model_name]:
            fpr, tpr, _ = roc_curve(
                results[model_name]['labels'],
                results[model_name]['probs']
            )
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={results[model_name]['auc_roc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare GA model with baselines')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train/val/test splits')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['ga', 'resnet', 'efficientnet'],
                       choices=['ga', 'resnet', 'efficientnet'],
                       help='Models to compare')
    parser.add_argument('--output_dir', type=str, default='results/baseline_comparison',
                       help='Output directory for results')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate (requires checkpoints)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader = load_dataset_from_directory(
        args.data_dir,
        image_size=(224, 224)
    )
    
    # Create test loader (use validation for now if no separate test set)
    test_loader = val_loader
    
    results = {}
    
    # Train and evaluate each model
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name.upper()}")
        print(f"{'='*60}")
        
        if model_name == 'ga':
            model = GAMedicalClassifier(
                num_classes=2,
                multivector_dim=8,
                feature_dim=128,
                device=device
            ).to(device)
        elif model_name == 'resnet':
            model = BaselineResNet(num_classes=2, pretrained=True).to(device)
        elif model_name == 'efficientnet':
            model = BaselineEfficientNet(num_classes=2, pretrained=True).to(device)
        else:
            continue
        
        # Train if not skipping
        if not args.skip_training:
            model = train_baseline_model(
                model, train_loader, val_loader, 
                args.num_epochs, device, model_name
            )
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name
            }, f'checkpoints/{model_name}_baseline.pth')
        else:
            # Load checkpoint
            checkpoint_path = f'checkpoints/{model_name}_baseline.pth'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"Warning: No checkpoint found for {model_name}, skipping...")
                continue
        
        # Evaluate
        metrics, probs, labels = evaluate_model(model, test_loader, device, model_name)
        results[model_name] = metrics
        results[model_name]['probs'] = probs
        results[model_name]['labels'] = labels
        
        print(f"\n{model_name.upper()} Results:")
        for key, value in metrics.items():
            if key not in ['probs', 'labels']:
                print(f"  {key}: {value:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save JSON
    results_json = {k: {k2: v2 for k2, v2 in v.items() 
                       if k2 not in ['probs', 'labels']} 
                   for k, v in results.items()}
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Create comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("-" * 75)
    for model_name in args.models:
        if model_name in results:
            r = results[model_name]
            print(f"{model_name:<15} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                  f"{r['recall']:<12.4f} {r['f1_score']:<12.4f} {r['auc_roc']:<12.4f}")
    
    # Plot comparisons
    plot_comparison(results, args.output_dir)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

