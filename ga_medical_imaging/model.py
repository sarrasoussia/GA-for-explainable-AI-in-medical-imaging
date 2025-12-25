"""
Modèle de classification basé sur l'algèbre géométrique pour l'imagerie médicale.
"""

import torch
import torch.nn as nn
from .ga_representation import GAFeatureExtractor, GeometricAlgebraRepresentation


class GAMedicalClassifier(nn.Module):
    """
    Classificateur utilisant l'algèbre géométrique pour la classification
    d'images médicales (tissu sain vs tumeur).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        multivector_dim: int = 8,
        feature_dim: int = 128,
        device: str = 'cpu'
    ):
        """
        Args:
            num_classes: Nombre de classes (2 pour binaire: sain/tumeur)
            multivector_dim: Dimension des multivecteurs GA
            feature_dim: Dimension des caractéristiques extraites
            device: Device PyTorch
        """
        super().__init__()
        self.device = device
        self.multivector_dim = multivector_dim
        self.num_classes = num_classes
        
        # Convertisseur image -> multivecteur
        self.ga_representation = GeometricAlgebraRepresentation(dim=3, device=device)
        
        # Extracteur de caractéristiques GA
        self.feature_extractor = GAFeatureExtractor(
            multivector_dim=multivector_dim,
            feature_dim=feature_dim
        )
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Stockage des activations pour l'explicabilité
        self.activations = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Enregistre des hooks pour capturer les activations intermédiaires."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook sur la dernière couche GA pour l'explicabilité
        self.feature_extractor.ga_layers[-1].register_forward_hook(
            get_activation('ga_features')
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            images: Tensor d'images (B, C, H, W) ou (B, H, W)
            
        Returns:
            logits: (B, num_classes)
        """
        # Convertir les images en multivecteurs
        multivectors = self.ga_representation.batch_to_multivectors(images)
        
        # Extraire les caractéristiques géométriques
        features = self.feature_extractor(multivectors)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_multivector_components(self, images: torch.Tensor) -> dict:
        """
        Retourne les composantes multivecteurs pour l'analyse.
        
        Returns:
            dict avec les différentes composantes géométriques
        """
        multivectors = self.ga_representation.batch_to_multivectors(images)
        
        return {
            'scalars': multivectors[..., 0],  # Intensités
            'vectors': multivectors[..., 1:4],  # Gradients
            'bivectors': multivectors[..., 4:7],  # Orientations/textures
            'trivector': multivectors[..., 7],  # Relations complexes
            'full_multivectors': multivectors
        }


class GAMedicalClassifierWithAttention(nn.Module):
    """
    Version améliorée avec mécanisme d'attention pour mieux identifier
    les régions importantes.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        multivector_dim: int = 8,
        feature_dim: int = 128,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.multivector_dim = multivector_dim
        
        self.ga_representation = GeometricAlgebraRepresentation(dim=3, device=device)
        self.feature_extractor = GAFeatureExtractor(
            multivector_dim=multivector_dim,
            feature_dim=feature_dim
        )
        
        # Mécanisme d'attention spatiale
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * multivector_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.activations = {}
        self.register_hooks()
    
    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        self.feature_extractor.ga_layers[-1].register_forward_hook(
            get_activation('ga_features')
        )
    
    def forward(self, images: torch.Tensor) -> tuple:
        """
        Returns:
            logits: (B, num_classes)
            attention_weights: (B, H, W) - poids d'attention pour explicabilité
        """
        multivectors = self.ga_representation.batch_to_multivectors(images)
        B, H, W, MV_DIM = multivectors.shape
        
        # Extraire les caractéristiques par pixel
        x = multivectors.view(B * H * W, 1, MV_DIM)
        x = self.feature_extractor.ga_layers(x)  # (B*H*W, feature_dim, MV_DIM)
        x = x.view(B * H * W, -1)
        x = self.feature_extractor.projection(x)  # (B*H*W, feature_dim)
        x = x.view(B, H * W, -1)
        
        # Calculer les poids d'attention
        attention_scores = self.attention(x)  # (B, H*W, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, H*W, 1)
        
        # Pooling pondéré par attention
        features = torch.sum(x * attention_weights, dim=1)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(features)
        
        attention_map = attention_weights.view(B, H, W)
        
        return logits, attention_map

