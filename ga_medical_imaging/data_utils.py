"""
Utilitaires pour charger et préprocesser les données d'imagerie médicale.
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
    Dataset pour les images médicales avec labels binaires (sain/tumeur).
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
            image_paths: Liste des chemins vers les images
            labels: Liste des labels (0=sain, 1=tumeur)
            transform: Transformations à appliquer
            image_size: Taille cible des images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        # Transform par défaut si non fourni
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation [-1, 1]
            ])
    
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
    Crée un dataset factice pour tester le modèle.
    Génère des images synthétiques simulant des images médicales.
    
    Args:
        num_samples: Nombre d'échantillons à générer
        image_size: Taille des images
        output_dir: Répertoire de sortie
        
    Returns:
        Tuple (image_paths, labels)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sain'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tumeur'), exist_ok=True)
    
    image_paths = []
    labels = []
    
    np.random.seed(42)
    
    for i in range(num_samples):
        # Générer une image synthétique
        image = np.random.rand(*image_size) * 0.3  # Fond sombre
        
        # Ajouter du bruit gaussien
        image += np.random.normal(0, 0.1, image_size)
        
        # Pour les tumeurs, ajouter une région plus brillante et texturée
        label = i % 2  # Alterner entre sain et tumeur
        
        if label == 1:  # Tumeur
            # Ajouter une région anormale
            center_x, center_y = np.random.randint(50, image_size[0]-50, 2)
            radius = np.random.randint(20, 40)
            
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            image[mask] += np.random.uniform(0.3, 0.6, size=mask.sum())
            # Ajouter de la texture
            image += np.random.normal(0, 0.15, image_size) * mask
        
        # Normaliser et sauvegarder
        image = np.clip(image, 0, 1)
        image_uint8 = (image * 255).astype(np.uint8)
        
        subdir = 'tumeur' if label == 1 else 'sain'
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
    Charge un dataset depuis un répertoire organisé par classes.
    
    Structure attendue:
        data_dir/
            sain/
                *.png
            tumeur/
                *.png
    
    Args:
        data_dir: Répertoire racine des données
        image_size: Taille des images
        train_split: Proportion pour l'entraînement
        
    Returns:
        Tuple (train_loader, val_loader)
    """
    # Trouver toutes les images
    sain_paths = glob.glob(os.path.join(data_dir, 'sain', '*'))
    tumeur_paths = glob.glob(os.path.join(data_dir, 'tumeur', '*'))
    
    # Créer les labels
    sain_labels = [0] * len(sain_paths)
    tumeur_labels = [1] * len(tumeur_paths)
    
    # Combiner
    all_paths = sain_paths + tumeur_paths
    all_labels = sain_labels + tumeur_labels
    
    # Mélanger
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split train/val
    split_idx = int(len(all_paths) * train_split)
    train_paths = all_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_paths[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Créer les datasets
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = MedicalImageDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = MedicalImageDataset(val_paths, val_labels, val_transform, image_size)
    
    # Créer les dataloaders
    # Note: num_workers=0 sur macOS pour éviter les problèmes de mutex
    import platform
    num_workers = 0 if platform.system() == 'Darwin' else 2
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

