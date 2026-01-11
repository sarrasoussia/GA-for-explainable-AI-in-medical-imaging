"""
Utilities for loading and preprocessing image datasets for classification.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
import glob


class MedicalImageDataset(Dataset):
    """
    Dataset for image classification with binary labels.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0=negative class, 1=positive class)
            transform: Transformations to apply
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        # Transform par défaut si non fourni
        if self.transform is None:
            from .preprocessing import get_val_transform
            self.transform = get_val_transform(image_size=image_size)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor (C, H, W)
            label: int (0 ou 1)
        """
        # Charger l'image
        img_path = self.image_paths[idx]
        
        try:
            if img_path.endswith('.npy'):
                # Image numpy
                image = np.load(img_path)
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)  # Convertir en grayscale
                image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                # Image standard (PNG, JPG, etc.)
                image = Image.open(img_path).convert('L')  # Grayscale
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            else:
                # Conversion manuelle si pas de transform
                image = transforms.functional.resize(image, self.image_size)
                image = transforms.functional.to_tensor(image)
            
            label = self.labels[idx]
            
            return image, label
            
        except Exception as e:
            print(f"Erreur lors du chargement de {img_path}: {e}")
            # Retourner une image noire en cas d'erreur
            image = torch.zeros(1, *self.image_size)
            return image, self.labels[idx]


def create_dummy_dataset(
    num_samples: int = 100,
    image_size: Tuple[int, int] = (224, 224),
    output_dir: str = 'data/dummy'
) -> Tuple[List[str], List[int]]:
    """
    Create a dummy dataset for testing the model.
    Generates synthetic images for binary classification testing.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Image size
        output_dir: Output directory
        
    Returns:
        Tuple (image_paths, labels)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'class_0'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'class_1'), exist_ok=True)
    
    image_paths = []
    labels = []
    
    np.random.seed(42)
    
    for i in range(num_samples):
        # Generate a synthetic image
        image = np.random.rand(*image_size) * 0.3  # Dark background
        
        # Add Gaussian noise
        image += np.random.normal(0, 0.1, image_size)
        
        # For class 1, add a brighter and textured region
        label = i % 2  # Alternate between class 0 and class 1
        
        if label == 1:  # Class 1
            # Add an anomalous region
            center_x, center_y = np.random.randint(50, image_size[0]-50, 2)
            radius = np.random.randint(20, 40)
            
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            image[mask] += np.random.uniform(0.3, 0.6, size=mask.sum())
            # Add texture
            image += np.random.normal(0, 0.15, image_size) * mask
        
        # Normalize and save
        image = np.clip(image, 0, 1)
        image_uint8 = (image * 255).astype(np.uint8)
        
        subdir = 'class_1' if label == 1 else 'class_0'
        filename = os.path.join(output_dir, subdir, f'image_{i:04d}.png')
        
        Image.fromarray(image_uint8).save(filename)
        
        image_paths.append(filename)
        labels.append(label)
    
    return image_paths, labels


def load_dataset_from_directory(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a dataset from a directory organized by classes.
    
    Expected structure (supports multiple naming conventions):
        data_dir/
            no_findings/ or normal/ or negative/  (label=0)
                *.png, *.jpg, etc.
            covid/ or covid-19/ or positive/    (label=1)
                *.png, *.jpg, etc.
    
    Args:
        data_dir: Root directory of the dataset
        image_size: Target image size
        train_split: Proportion for training
        
    Returns:
        Tuple (train_loader, val_loader)
    """
    # Try different naming conventions for negative class (label=0)
    negative_dirs = ['sain', 'no_findings', 'normal', 'negative', 'healthy']
    positive_dirs = ['tumeur', 'covid', 'covid-19', 'positive']
    
    # Find negative class images
    negative_paths = []
    for neg_dir in negative_dirs:
        neg_path = os.path.join(data_dir, neg_dir)
        if os.path.exists(neg_path):
            negative_paths = glob.glob(os.path.join(neg_path, '*'))
            # Filter for image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            negative_paths = [p for p in negative_paths if any(p.endswith(ext.replace('*', '')) for ext in image_extensions)]
            if negative_paths:
                print(f"Found {len(negative_paths)} negative class images in {neg_dir}/")
                break
    
    # Find positive class images
    positive_paths = []
    for pos_dir in positive_dirs:
        pos_path = os.path.join(data_dir, pos_dir)
        if os.path.exists(pos_path):
            positive_paths = glob.glob(os.path.join(pos_path, '*'))
            # Filter for image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            positive_paths = [p for p in positive_paths if any(p.endswith(ext.replace('*', '')) for ext in image_extensions)]
            if positive_paths:
                print(f"Found {len(positive_paths)} positive class images in {pos_dir}/")
                break
    
    if not negative_paths:
        raise ValueError(f"No negative class images found. Tried: {negative_dirs}")
    if not positive_paths:
        raise ValueError(f"No positive class images found. Tried: {positive_dirs}")
    
    # Créer les labels
    negative_labels = [0] * len(negative_paths)
    positive_labels = [1] * len(positive_paths)
    
    # Combiner
    all_paths = negative_paths + positive_paths
    all_labels = negative_labels + positive_labels
    
    # Mélanger avec seed fixe pour reproductibilité
    rng = np.random.RandomState(42)  # Use fixed seed for reproducibility
    indices = rng.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split train/val
    split_idx = int(len(all_paths) * train_split)
    train_paths = all_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_paths[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Créer les datasets
    # Use standardized preprocessing for fair comparison
    from .preprocessing import get_train_transform, get_val_transform
    
    train_transform = get_train_transform(image_size=image_size, augmentation=True)
    val_transform = get_val_transform(image_size=image_size)
    
    train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = MedicalImageDataset(val_paths, val_labels, val_transform, image_size)
    
    # Créer les dataloaders avec mélange déterministe
    # Note: num_workers=0 sur macOS pour éviter les problèmes de mutex
    import platform
    num_workers = 0 if platform.system() == 'Darwin' else 2
    
    # Use fixed generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(42)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=num_workers,
        generator=generator  # Deterministic shuffling
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

