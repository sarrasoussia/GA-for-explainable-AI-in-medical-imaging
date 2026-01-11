"""
Comprehensive comparison between GA model and traditional CNN baseline.
Implements evaluation metrics: Accuracy, Precision, Recall, Specificity, F1-Score, ROC AUC, Confusion Matrix.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from tqdm import tqdm
import os
import json
from typing import Dict, Tuple, List

from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.data_utils import create_dummy_dataset, MedicalImageDataset, load_dataset_from_directory


class TraditionalCNN(nn.Module):
    """
    Traditional Convolutional Neural Network baseline for comparison.
    Similar architecture complexity to GAMedicalClassifier.
    """
    
    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            # Fourth block
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Ensure input has channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities for positive class (for ROC AUC)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate specificity manually (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Recall is same as sensitivity
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


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_name: str = "Model"
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a model and return metrics, predictions, and probabilities.
    
    Returns:
        metrics: Dictionary of evaluation metrics
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities for positive class
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    model_name: str = "Model"
) -> nn.Module:
    """Train a model and return the best model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
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
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    return model


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sain', 'Tumeur'],
                yticklabels=['Sain', 'Tumeur'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: str = None
):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        if 'y_true' in result and 'y_probs' in result:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_probs'])
            auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(
    results: Dict[str, Dict],
    save_path: str = None
):
    """Plot bar chart comparing all metrics."""
    models = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1-Score', 'ROC AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [results[m]['metrics'][metric] for m in models]
        colors = ['#2E86AB', '#A23B72'] if len(models) == 2 else None
        
        bars = axes[idx].bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[idx].set_title(label, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Comparison - Evaluation Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_comparison_table(results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC AUC']
    
    # Header
    header = f"{'Metric':<25}"
    for model in models:
        header += f"{model:>15}"
    print(header)
    print("-" * 80)
    
    # Rows
    for metric, name in zip(metrics, metric_names):
        row = f"{name:<25}"
        for model in models:
            value = results[model]['metrics'][metric]
            row += f"{value:>15.4f}"
        print(row)
    
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare GA model with traditional CNN')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (if None, creates dummy dataset)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
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
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("Creating dummy dataset...")
        image_paths, labels = create_dummy_dataset(
            num_samples=200,
            image_size=(224, 224),
            output_dir='data/dummy'
        )
        
        # Split train/val
        split_idx = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_idx]
        train_labels = labels[:split_idx]
        val_paths = image_paths[split_idx:]
        val_labels = labels[split_idx:]
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        train_dataset = MedicalImageDataset(
            train_paths, train_labels, train_transform, (224, 224)
        )
        val_dataset = MedicalImageDataset(
            val_paths, val_labels, val_transform, (224, 224)
        )
    else:
        train_loader, val_loader = load_dataset_from_directory(
            args.data_dir,
            image_size=(224, 224)
        )
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize models
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)
    
    ga_model = GAMedicalClassifier(
        num_classes=2,
        multivector_dim=8,
        feature_dim=128,
        device=device
    ).to(device)
    
    cnn_model = TraditionalCNN(num_classes=2, input_channels=1).to(device)
    
    ga_params = sum(p.numel() for p in ga_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    
    print(f"GA Model parameters: {ga_params:,}")
    print(f"CNN Model parameters: {cnn_params:,}")
    
    # Train models
    if not args.skip_training:
        ga_model = train_model(
            ga_model, train_loader, val_loader,
            args.num_epochs, args.learning_rate, device, "GA Model"
        )
        
        cnn_model = train_model(
            cnn_model, train_loader, val_loader,
            args.num_epochs, args.learning_rate, device, "Traditional CNN"
        )
        
        # Save models
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(ga_model.state_dict(), 'checkpoints/ga_model_comparison.pth')
        torch.save(cnn_model.state_dict(), 'checkpoints/cnn_model_comparison.pth')
    else:
        # Load models
        ga_model.load_state_dict(torch.load('checkpoints/ga_model_comparison.pth', map_location=device))
        cnn_model.load_state_dict(torch.load('checkpoints/cnn_model_comparison.pth', map_location=device))
    
    # Evaluate models
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    ga_metrics, ga_y_true, ga_y_pred, ga_y_probs = evaluate_model(
        ga_model, val_loader, device, "GA Model"
    )
    
    cnn_metrics, cnn_y_true, cnn_y_pred, cnn_y_probs = evaluate_model(
        cnn_model, val_loader, device, "Traditional CNN"
    )
    
    # Store results
    results = {
        'GA Model': {
            'metrics': ga_metrics,
            'y_true': ga_y_true,
            'y_pred': ga_y_pred,
            'y_probs': ga_y_probs
        },
        'Traditional CNN': {
            'metrics': cnn_metrics,
            'y_true': cnn_y_true,
            'y_pred': cnn_y_pred,
            'y_probs': cnn_y_probs
        }
    }
    
    # Print comparison
    print_comparison_table(results)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Confusion matrices
    plot_confusion_matrix(
        ga_y_true, ga_y_pred, "GA Model",
        os.path.join(args.output_dir, 'confusion_matrix_ga.png')
    )
    plot_confusion_matrix(
        cnn_y_true, cnn_y_pred, "Traditional CNN",
        os.path.join(args.output_dir, 'confusion_matrix_cnn.png')
    )
    print("  ✓ Confusion matrices saved")
    
    # ROC curves
    plot_roc_curves(
        results,
        os.path.join(args.output_dir, 'roc_curves.png')
    )
    print("  ✓ ROC curves saved")
    
    # Metrics comparison
    plot_metrics_comparison(
        results,
        os.path.join(args.output_dir, 'metrics_comparison.png')
    )
    print("  ✓ Metrics comparison saved")
    
    # Save results to JSON
    results_json = {
        'GA Model': {
            'metrics': ga_metrics,
            'num_parameters': ga_params
        },
        'Traditional CNN': {
            'metrics': cnn_metrics,
            'num_parameters': cnn_params
        }
    }
    
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    print("  ✓ Results saved to JSON")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nGA Model Strengths:")
    if ga_metrics['accuracy'] >= cnn_metrics['accuracy']:
        print(f"  • Higher or equal accuracy: {ga_metrics['accuracy']:.4f} vs {cnn_metrics['accuracy']:.4f}")
    if ga_metrics['roc_auc'] >= cnn_metrics['roc_auc']:
        print(f"  • Higher or equal ROC AUC: {ga_metrics['roc_auc']:.4f} vs {cnn_metrics['roc_auc']:.4f}")
    if ga_metrics['f1_score'] >= cnn_metrics['f1_score']:
        print(f"  • Higher or equal F1-Score: {ga_metrics['f1_score']:.4f} vs {cnn_metrics['f1_score']:.4f}")
    
    print("\nTraditional CNN Strengths:")
    if cnn_metrics['accuracy'] > ga_metrics['accuracy']:
        print(f"  • Higher accuracy: {cnn_metrics['accuracy']:.4f} vs {ga_metrics['accuracy']:.4f}")
    if cnn_metrics['roc_auc'] > ga_metrics['roc_auc']:
        print(f"  • Higher ROC AUC: {cnn_metrics['roc_auc']:.4f} vs {ga_metrics['roc_auc']:.4f}")
    if cnn_metrics['f1_score'] > ga_metrics['f1_score']:
        print(f"  • Higher F1-Score: {cnn_metrics['f1_score']:.4f} vs {ga_metrics['f1_score']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

