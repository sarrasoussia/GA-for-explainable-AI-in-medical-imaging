"""
Module pour convertir les images médicales en représentations multivecteurs
utilisant l'algèbre géométrique (GA).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F


class GeometricAlgebraRepresentation:
    """
    Convertit les caractéristiques d'images médicales en multivecteurs GA.
    
    Utilise l'algèbre de Clifford pour représenter:
    - Les intensités de pixels (scalaires)
    - Les gradients spatiaux (vecteurs)
    - Les orientations et textures (bivecteurs)
    - Les relations géométriques complexes (multivecteurs)
    """
    
    def __init__(self, dim: int = 3, device: str = 'cpu'):
        """
        Args:
            dim: Dimension de l'algèbre géométrique (3D pour images 2D + intensité)
            device: Device PyTorch ('cpu' ou 'cuda')
        """
        self.dim = dim
        self.device = device
        # Note: L'implémentation utilise une construction manuelle des multivecteurs
        # pour une meilleure compatibilité avec PyTorch
        
    def image_to_multivector(self, image: np.ndarray) -> torch.Tensor:
        """
        Convertit une image en représentation multivecteur.
        
        Pour chaque pixel/patch:
        - Scalar: intensité normalisée
        - Vector: gradients (dx, dy)
        - Bivector: orientation et texture
        
        Args:
            image: Image numpy array (H, W) ou (H, W, C)
            
        Returns:
            Tensor multivecteur de forme (H, W, multivector_dim)
        """
        if len(image.shape) == 3:
            # Image RGB/grayscale multi-channel
            image = np.mean(image, axis=2) if image.shape[2] > 1 else image[:, :, 0]
        
        image = image.astype(np.float32)
        h, w = image.shape
        
        # Normaliser l'intensité
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Calculer les gradients (vecteurs)
        grad_x = np.gradient(image_norm, axis=1)
        grad_y = np.gradient(image_norm, axis=0)
        
        # Calculer les dérivées secondes pour les bivecteurs (courbure, texture)
        grad_xx = np.gradient(grad_x, axis=1)
        grad_yy = np.gradient(grad_y, axis=0)
        grad_xy = np.gradient(grad_x, axis=0)
        
        # Construire les multivecteurs
        # Pour GA 3D: [scalar, e1, e2, e3, e12, e13, e23, e123]
        multivectors = np.zeros((h, w, 8), dtype=np.float32)
        
        # Scalar (grade 0)
        multivectors[:, :, 0] = image_norm
        
        # Vectors (grade 1): gradients
        multivectors[:, :, 1] = grad_x
        multivectors[:, :, 2] = grad_y
        multivectors[:, :, 3] = 0  # e3 non utilisé pour images 2D
        
        # Bivectors (grade 2): orientations et textures
        multivectors[:, :, 4] = grad_xy  # e12: rotation/courbure
        multivectors[:, :, 5] = grad_xx  # e13: texture horizontale
        multivectors[:, :, 6] = grad_yy  # e23: texture verticale
        
        # Trivector (grade 3): volume/relation complexe
        multivectors[:, :, 7] = image_norm * grad_xy  # e123: combinaison
        
        return torch.from_numpy(multivectors).to(self.device)
    
    def batch_to_multivectors(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Convertit un batch d'images en multivecteurs.
        
        Args:
            batch: Tensor de forme (B, C, H, W) ou (B, H, W)
            
        Returns:
            Tensor de forme (B, H, W, multivector_dim)
        """
        batch_np = batch.cpu().numpy()
        if len(batch_np.shape) == 4:
            batch_np = np.mean(batch_np, axis=1)  # Moyenne sur les canaux
        
        multivectors = []
        for img in batch_np:
            mv = self.image_to_multivector(img)
            multivectors.append(mv)
        
        return torch.stack(multivectors)


class GAMultivectorLayer(nn.Module):
    """
    Couche de réseau de neurones qui opère sur des multivecteurs GA.
    """
    
    def __init__(self, in_dim: int, out_dim: int, multivector_dim: int = 8):
        """
        Args:
            in_dim: Dimension d'entrée (nombre de multivecteurs)
            out_dim: Dimension de sortie
            multivector_dim: Dimension d'un multivecteur (8 pour GA 3D)
        """
        super().__init__()
        self.multivector_dim = multivector_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Poids pour chaque composante du multivecteur
        # Permet d'apprendre les relations géométriques
        self.weights = nn.Parameter(
            torch.randn(in_dim, out_dim, multivector_dim) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, multivector_dim))
        
    def geometric_product(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Produit géométrique simplifié entre multivecteurs.
        Pour GA 3D, on utilise une approximation du produit géométrique.
        """
        # x: (..., multivector_dim), w: (multivector_dim)
        # Produit géométrique simplifié
        result = torch.zeros_like(x[..., :1]).expand(*x.shape[:-1], self.multivector_dim)
        
        # Scalar * Scalar
        result[..., 0] = x[..., 0] * w[0]
        
        # Vector * Vector (produit scalaire + bivector)
        result[..., 0] += x[..., 1] * w[1] + x[..., 2] * w[2]  # produit scalaire
        result[..., 4] = x[..., 1] * w[2] - x[..., 2] * w[1]  # bivector e12
        
        # Bivector contributions
        result[..., 4] += x[..., 4] * w[0]  # bivector * scalar
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forme (..., in_dim, multivector_dim)
            
        Returns:
            Tensor de forme (..., out_dim, multivector_dim)
        """
        # Appliquer le produit géométrique pour chaque poids
        outputs = []
        for i in range(self.out_dim):
            out = torch.zeros(*x.shape[:-2], self.multivector_dim, device=x.device)
            for j in range(self.in_dim):
                gp = self.geometric_product(x[..., j, :], self.weights[j, i, :])
                out = out + gp
            out = out + self.bias[i, :].to(x.device)  # Ensure bias is on same device
            outputs.append(out)
        
        return torch.stack(outputs, dim=-2)


class GAFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques utilisant l'algèbre géométrique.
    """
    
    def __init__(self, multivector_dim: int = 8, feature_dim: int = 128):
        super().__init__()
        self.multivector_dim = multivector_dim
        
        # Enhanced GA layers with residual connections for better feature extraction
        self.ga_layers = nn.Sequential(
            GAMultivectorLayer(1, 64, multivector_dim),
            nn.ReLU(inplace=True),
            GAMultivectorLayer(64, 128, multivector_dim),
            nn.ReLU(inplace=True),
            GAMultivectorLayer(128, 256, multivector_dim),
            nn.ReLU(inplace=True),
            GAMultivectorLayer(256, feature_dim, multivector_dim),
        )
        
        # Enhanced projection with normalization
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * multivector_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, multivectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multivectors: (B, H, W, multivector_dim)
            
        Returns:
            features: (B, feature_dim)
        """
        B, H, W, MV_DIM = multivectors.shape
        
        # Reshape pour les couches GA: (B*H*W, 1, MV_DIM)
        x = multivectors.view(B * H * W, 1, MV_DIM)
        
        # Passer à travers les couches GA
        x = self.ga_layers(x)  # (B*H*W, feature_dim, MV_DIM)
        
        # Flatten et projeter
        x = x.view(B * H * W, -1)
        x = self.projection(x)  # (B*H*W, feature_dim)
        
        # Pooling spatial (moyenne globale)
        x = x.view(B, H * W, -1)
        x = torch.mean(x, dim=1)  # (B, feature_dim)
        
        return x

