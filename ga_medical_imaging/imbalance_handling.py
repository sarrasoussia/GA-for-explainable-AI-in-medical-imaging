"""
Class imbalance handling utilities.

Provides simple, reliable alternatives to cGAN for handling class imbalance:
1. Class-weighted loss function
2. Weighted random sampling
3. Focal loss (optional)

These methods are more stable and reliable than cGAN for handling imbalance.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import List, Tuple, Optional


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calculate class weights for weighted loss function.
    
    Weights are inversely proportional to class frequency:
    weight[i] = total_samples / (num_classes * class_i_count)
    
    Args:
        labels: List of class labels
    
    Returns:
        Tensor of class weights (shape: [num_classes])
    """
    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)
    
    total = len(labels_array)
    num_classes = len(unique)
    
    weights = torch.zeros(num_classes, dtype=torch.float32)
    
    for i, class_label in enumerate(unique):
        class_count = counts[i]
        # Inverse frequency weighting
        weights[int(class_label)] = total / (num_classes * class_count)
    
    return weights


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced training.
    
    This ensures each batch has roughly balanced classes during training.
    
    Args:
        labels: List of class labels
    
    Returns:
        WeightedRandomSampler instance
    """
    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)
    
    # Calculate sample weights (inverse frequency)
    sample_weights = np.zeros(len(labels_array))
    for i, class_label in enumerate(unique):
        class_count = counts[i]
        # Weight inversely proportional to frequency
        weight = 1.0 / class_count
        sample_weights[labels_array == class_label] = weight
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels_array),
        replacement=True
    )


def create_weighted_loss(labels: List[int], device: str = 'cpu') -> nn.Module:
    """
    Create a weighted CrossEntropyLoss to handle class imbalance.
    
    Args:
        labels: List of class labels
        device: PyTorch device
    
    Returns:
        Weighted CrossEntropyLoss
    """
    class_weights = calculate_class_weights(labels)
    class_weights = class_weights.to(device)
    
    return nn.CrossEntropyLoss(weight=class_weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    which helps with imbalanced datasets.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        """
        Args:
            alpha: Class weighting (None = no class weighting)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Class indices (B,)
        
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def create_focal_loss(
    labels: List[int],
    gamma: float = 2.0,
    device: str = 'cpu'
) -> FocalLoss:
    """
    Create Focal Loss with class weighting.
    
    Args:
        labels: List of class labels
        gamma: Focusing parameter (default: 2.0)
        device: PyTorch device
    
    Returns:
        FocalLoss instance
    """
    class_weights = calculate_class_weights(labels)
    class_weights = class_weights.to(device)
    
    return FocalLoss(alpha=class_weights, gamma=gamma)


def get_imbalance_handling_strategy(
    labels: List[int],
    strategy: str = 'weighted_loss',
    device: str = 'cpu'
) -> Tuple[Optional[WeightedRandomSampler], nn.Module]:
    """
    Get imbalance handling strategy (sampler + loss).
    
    Args:
        labels: List of class labels
        strategy: 'weighted_loss', 'weighted_sampling', 'focal_loss', or 'both'
        device: PyTorch device
    
    Returns:
        Tuple of (sampler, loss_function)
        - sampler: None or WeightedRandomSampler
        - loss_function: Weighted loss or FocalLoss
    """
    if strategy == 'weighted_loss':
        sampler = None
        loss = create_weighted_loss(labels, device)
    
    elif strategy == 'weighted_sampling':
        sampler = create_weighted_sampler(labels)
        loss = nn.CrossEntropyLoss()
    
    elif strategy == 'focal_loss':
        sampler = None
        loss = create_focal_loss(labels, gamma=2.0, device=device)
    
    elif strategy == 'both':
        sampler = create_weighted_sampler(labels)
        loss = create_weighted_loss(labels, device)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'weighted_loss', 'weighted_sampling', 'focal_loss', or 'both'")
    
    return sampler, loss


def print_imbalance_info(labels: List[int]):
    """
    Print information about class imbalance.
    
    Args:
        labels: List of class labels
    """
    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)
    
    print("=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    
    for i, class_label in enumerate(unique):
        count = counts[i]
        percentage = 100 * count / len(labels_array)
        print(f"Class {int(class_label)}: {count} samples ({percentage:.2f}%)")
    
    if len(unique) == 2:
        ratio = max(counts) / min(counts)
        print(f"\nImbalance ratio: {ratio:.2f}:1")
        
        if ratio > 2.0:
            print("⚠️ Significant class imbalance detected!")
            print("   Recommendation: Use weighted loss or weighted sampling")
        else:
            print("✓ Classes are relatively balanced")
    
    # Calculate recommended weights
    class_weights = calculate_class_weights(labels)
    print(f"\nRecommended class weights: {class_weights.numpy()}")
    print("=" * 60)

