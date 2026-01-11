"""
Professional Training Script for Geometric Algebra Medical Imaging Model

This script implements production-ready training with:
- Early stopping
- Advanced learning rate scheduling (Cosine Annealing with Warm Restarts)
- Gradient clipping
- Mixed precision training (optional)
- Comprehensive logging
- Reproducibility (random seeds)
- Professional data augmentation
- Model checkpointing and resume capability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
from typing import Optional, Dict, Tuple
import random
from datetime import datetime

from .model import GAMedicalClassifier, GAMedicalClassifierWithAttention
from .data_utils import load_dataset_from_directory
from .device_utils import get_device, print_device_info
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


def train_epoch_professional(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    use_mixed_precision: bool = False,
    max_grad_norm: float = 1.0,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, float]:
    """
    Professional training epoch with mixed precision and gradient clipping.
    
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_professional(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_mixed_precision: bool = False
) -> Tuple[float, float]:
    """Professional validation with mixed precision support."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_mixed_precision:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_professional(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    save_dir: str = 'checkpoints',
    use_tensorboard: bool = True,
    use_mixed_precision: bool = False,
    early_stopping_patience: int = 15,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    resume_from: Optional[str] = None
) -> Dict:
    """
    Professional training function with all best practices.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        device: PyTorch device
        save_dir: Directory to save checkpoints
        use_tensorboard: Enable TensorBoard logging
        use_mixed_precision: Use mixed precision training (AMP)
        early_stopping_patience: Patience for early stopping
        weight_decay: L2 regularization
        max_grad_norm: Maximum gradient norm for clipping
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Dictionary with training history and best metrics
    """
    # Set reproducibility
    set_seed(42)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Mixed precision setup
    scaler = GradScaler() if use_mixed_precision and device == 'cuda' else None
    
    # Loss and optimizer
    # Handle class imbalance with weighted loss
    try:
        # Extract labels from dataset
        labels = [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
        from .imbalance_handling import calculate_class_weights
        class_weights = calculate_class_weights(labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    except (AttributeError, IndexError, TypeError):
        # Fallback if dataset structure is different
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced learning rate scheduler: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Initial restart period
        T_mult=2,  # Period multiplier
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.001,
        mode='max'
    )
    
    # TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir=log_dir)
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"PROFESSIONAL TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_mixed_precision}")
    print(f"Max Epochs: {num_epochs}")
    print(f"Initial LR: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Gradient Clipping: {max_grad_norm}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss, train_acc = train_epoch_professional(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixed_precision, max_grad_norm, scaler
        )
        
        # Validation
        val_loss, val_acc = validate_professional(
            model, val_loader, criterion, device, use_mixed_precision
        )
        
        # Learning rate scheduling (step after validation)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print progress
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'best_val_acc': best_val_acc,
                'training_history': training_history
            }
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'training_history': training_history
            }
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    if writer:
        writer.close()
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total epochs trained: {epoch}")
    print(f"Model saved to: {save_dir}/best_model.pth")
    print(f"{'='*70}\n")
    
    return {
        'best_val_acc': best_val_acc,
        'final_epoch': epoch,
        'training_history': training_history
    }


def main():
    parser = argparse.ArgumentParser(
        description='Professional Training for GA Medical Imaging Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory with class subdirectories')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (increase if GPU memory allows)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2 regularization weight decay')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, mps, or auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/professional',
                       help='Directory to save checkpoints')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (H W)')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='Use mixed precision training (AMP) for faster training')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'attention'],
                       help='Model architecture type')
    
    args = parser.parse_args()
    
    # Device
    device = get_device(args.device)
    print_device_info(device)
    
    # Disable mixed precision if not on CUDA
    if args.use_mixed_precision and device != 'cuda':
        print("⚠ Mixed precision only available on CUDA. Disabling...")
        args.use_mixed_precision = False
    
    # Load dataset
    print(f"\n{'='*70}")
    print(f"LOADING DATASET")
    print(f"{'='*70}")
    train_loader, val_loader = load_dataset_from_directory(
        args.data_dir,
        image_size=tuple(args.image_size),
        train_split=0.8
    )
    
    # Create model
    model_kwargs = {
        'num_classes': 2,
        'multivector_dim': 8,
        'feature_dim': 256,  # Increased for better capacity
        'device': device
    }
    
    if args.model_type == 'attention':
        model = GAMedicalClassifierWithAttention(**model_kwargs).to(device)
        print("Using GAMedicalClassifierWithAttention")
    else:
        model = GAMedicalClassifier(**model_kwargs).to(device)
        print("Using GAMedicalClassifier")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Training
    results = train_professional(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir,
        use_mixed_precision=args.use_mixed_precision,
        early_stopping_patience=args.early_stopping_patience,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        resume_from=args.resume_from
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")


if __name__ == '__main__':
    main()

