"""
Standardized preprocessing for fair comparison between models.

This module provides consistent preprocessing transforms that ensure
all models (GA and CNN baselines) are evaluated on the same preprocessed data.
"""

import torch
from torchvision import transforms
from typing import Tuple, Optional


# Standard preprocessing configuration
STANDARD_PREPROCESSING = {
    'mean': [0.485],  # Standard for grayscale medical images (ImageNet-like)
    'std': [0.229],
    'image_size': (224, 224),
    'normalization_range': 'imagenet'  # Alternative: 'simple' for [-1, 1]
}


def get_train_transform(
    image_size: Tuple[int, int] = (224, 224),
    augmentation: bool = True,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> transforms.Compose:
    """
    Get standardized training transform with data augmentation.
    
    Args:
        image_size: Target image size (H, W)
        augmentation: Whether to apply data augmentation
        mean: Normalization mean (default: STANDARD_PREPROCESSING['mean'])
        std: Normalization std (default: STANDARD_PREPROCESSING['std'])
    
    Returns:
        Compose transform for training
    """
    if mean is None:
        mean = STANDARD_PREPROCESSING['mean']
    if std is None:
        std = STANDARD_PREPROCESSING['std']
    
    if augmentation:
        return transforms.Compose([
            transforms.Resize((image_size[0] + 32, image_size[1] + 32)),  # Slightly larger for crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Medical images can be flipped vertically
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Small translations
                scale=(0.9, 1.1),  # Slight scaling
                shear=5  # Small shearing
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Intensity variations
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_val_transform(
    image_size: Tuple[int, int] = (224, 224),
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> transforms.Compose:
    """
    Get standardized validation/test transform (minimal augmentation).
    
    Args:
        image_size: Target image size (H, W)
        mean: Normalization mean (default: STANDARD_PREPROCESSING['mean'])
        std: Normalization std (default: STANDARD_PREPROCESSING['std'])
    
    Returns:
        Compose transform for validation/testing
    """
    if mean is None:
        mean = STANDARD_PREPROCESSING['mean']
    if std is None:
        std = STANDARD_PREPROCESSING['std']
    
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_test_transform(
    image_size: Tuple[int, int] = (224, 224),
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> transforms.Compose:
    """
    Get standardized test transform (identical to validation).
    
    Args:
        image_size: Target image size (H, W)
        mean: Normalization mean (default: STANDARD_PREPROCESSING['mean'])
        std: Normalization std (default: STANDARD_PREPROCESSING['std'])
    
    Returns:
        Compose transform for testing
    """
    return get_val_transform(image_size, mean, std)


def denormalize(
    tensor: torch.Tensor,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean used
        std: Normalization std used
    
    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = STANDARD_PREPROCESSING['mean']
    if std is None:
        std = STANDARD_PREPROCESSING['std']
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean

