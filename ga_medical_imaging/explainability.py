"""
Module d'explicabilité pour identifier quelles composantes géométriques
influencent les décisions du modèle.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import os
# Configurer matplotlib pour éviter les problèmes sur macOS
import matplotlib
matplotlib.use('Agg')  # Utiliser backend non-interactif
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GAExplainabilityAnalyzer:
    """
    Analyseur d'explicabilité pour les modèles basés sur l'algèbre géométrique.
    Identifie quelles composantes géométriques (scalaires, vecteurs, bivecteurs)
    influencent le diagnostic.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Modèle GA entraîné
            device: Device PyTorch
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_component_importance(
        self,
        images: torch.Tensor,
        target_class: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule l'importance de chaque composante géométrique du multivecteur
        en utilisant des gradients (Gradient-weighted Class Activation Mapping).
        
        Args:
            images: Tensor d'images (B, C, H, W)
            target_class: Classe cible pour le calcul des gradients (None = classe prédite)
            
        Returns:
            Dict avec l'importance de chaque composante:
            - scalars_importance: importance des scalaires (intensités)
            - vectors_importance: importance des vecteurs (gradients)
            - bivectors_importance: importance des bivecteurs (orientations)
            - trivector_importance: importance du trivecteur
        """
        images = images.to(self.device)
        images.requires_grad = True
        
        # Forward pass
        logits = self.model(images)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        
        # Calculer les gradients pour chaque composante
        importance = {}
        
        # Obtenir les multivecteurs
        multivectors = self.model.ga_representation.batch_to_multivectors(images)
        
        # Calculer l'importance de chaque composante
        for i, comp_name in enumerate(['scalars', 'vectors', 'bivectors', 'trivector']):
            if comp_name == 'scalars':
                comp = multivectors[..., 0:1]
            elif comp_name == 'vectors':
                comp = multivectors[..., 1:4]
            elif comp_name == 'bivectors':
                comp = multivectors[..., 4:7]
            else:  # trivector
                comp = multivectors[..., 7:8]
            
            # Calculer le gradient
            if images.grad is not None:
                images.grad.zero_()
            
            # Score pour la classe cible
            if len(target_class.shape) == 0:
                score = logits[0, target_class]
            else:
                score = logits[0, target_class[0]]
            
            score.backward(retain_graph=True)
            
            # L'importance est la magnitude du gradient
            if images.grad is not None:
                grad_magnitude = torch.abs(images.grad).mean(dim=1)  # (B, H, W)
                importance[f'{comp_name}_importance'] = grad_magnitude.detach()
            else:
                importance[f'{comp_name}_importance'] = torch.zeros(
                    images.shape[0], images.shape[-2], images.shape[-1]
                ).to(self.device)
        
        return importance
    
    def analyze_geometric_components(
        self,
        images: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Analyse détaillée des composantes géométriques et leur contribution
        à la décision de classification.
        
        Returns:
            Dict avec:
            - component_contributions: contribution de chaque composante
            - spatial_importance: carte d'importance spatiale
            - geometric_features: caractéristiques géométriques extraites
        """
        self.model.eval()
        with torch.no_grad():
            # Obtenir les multivecteurs
            components = self.model.get_multivector_components(images)
            
            # Forward pass pour obtenir les prédictions
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Calculer la contribution de chaque composante
            contributions = {}
            
            # Importance relative de chaque grade
            scalars_mag = torch.abs(components['scalars']).mean()
            vectors_mag = torch.abs(components['vectors']).mean()
            bivectors_mag = torch.abs(components['bivectors']).mean()
            trivector_mag = torch.abs(components['trivector']).mean()
            
            total_mag = scalars_mag + vectors_mag + bivectors_mag + trivector_mag + 1e-8
            
            contributions['scalars_contribution'] = (scalars_mag / total_mag).item()
            contributions['vectors_contribution'] = (vectors_mag / total_mag).item()
            contributions['bivectors_contribution'] = (bivectors_mag / total_mag).item()
            contributions['trivector_contribution'] = (trivector_mag / total_mag).item()
            
            # Carte d'importance spatiale (magnitude des multivecteurs)
            spatial_importance = torch.norm(
                components['full_multivectors'], dim=-1
            ).cpu().numpy()
            
            return {
                'component_contributions': contributions,
                'spatial_importance': spatial_importance,
                'geometric_features': {
                    k: v.cpu().numpy() for k, v in components.items()
                },
                'predictions': {
                    'logits': logits.cpu().numpy(),
                    'probabilities': probs.cpu().numpy()
                }
            }
    
    def visualize_explanations(
        self,
        image: np.ndarray,
        analysis: Dict,
        save_path: str = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualise les explications du modèle.
        
        Args:
            image: Image originale (H, W) ou (H, W, C)
            analysis: Résultats de analyze_geometric_components
            save_path: Chemin pour sauvegarder la figure
            figsize: Taille de la figure
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2) if image.shape[2] > 1 else image[:, :, 0]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Image originale
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Image Originale')
        axes[0, 0].axis('off')
        
        # Carte d'importance spatiale
        spatial_imp = analysis['spatial_importance'][0]
        im1 = axes[0, 1].imshow(spatial_imp, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Importance Spatiale (Magnitude Multivecteurs)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Contributions des composantes
        contributions = analysis['component_contributions']
        comp_names = ['Scalaires\n(Intensités)', 'Vecteurs\n(Gradients)', 
                     'Bivecteurs\n(Orientations)', 'Trivecteur\n(Relations)']
        comp_values = [
            contributions['scalars_contribution'],
            contributions['vectors_contribution'],
            contributions['bivectors_contribution'],
            contributions['trivector_contribution']
        ]
        
        axes[0, 2].bar(comp_names, comp_values, color=['blue', 'green', 'orange', 'red'])
        axes[0, 2].set_title('Contribution des Composantes Géométriques')
        axes[0, 2].set_ylabel('Importance Relative')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Visualisation des composantes individuelles
        features = analysis['geometric_features']
        
        # Scalaires (intensités)
        axes[1, 0].imshow(features['scalars'][0], cmap='gray')
        axes[1, 0].set_title('Scalaires (Intensités)')
        axes[1, 0].axis('off')
        
        # Vecteurs (magnitude des gradients)
        vectors_mag = np.linalg.norm(features['vectors'][0], axis=-1)
        im2 = axes[1, 1].imshow(vectors_mag, cmap='viridis', interpolation='bilinear')
        axes[1, 1].set_title('Vecteurs (Gradients)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Bivecteurs (magnitude)
        bivectors_mag = np.linalg.norm(features['bivectors'][0], axis=-1)
        im3 = axes[1, 2].imshow(bivectors_mag, cmap='plasma', interpolation='bilinear')
        axes[1, 2].set_title('Bivecteurs (Orientations/Textures)')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        
        # Ajouter les prédictions
        probs = analysis['predictions']['probabilities'][0]
        pred_text = f"Probabilités:\n"
        pred_text += f"  Sain: {probs[0]:.3f}\n"
        pred_text += f"  Tumeur: {probs[1]:.3f}"
        axes[0, 2].text(0.5, -0.3, pred_text, transform=axes[0, 2].transAxes,
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_explanation_report(
        self,
        images: torch.Tensor,
        class_names: List[str] = ['Sain', 'Tumeur']
    ) -> str:
        """
        Génère un rapport textuel d'explication.
        
        Returns:
            Rapport d'explication en texte
        """
        analysis = self.analyze_geometric_components(images)
        
        probs = analysis['predictions']['probabilities'][0]
        pred_class = np.argmax(probs)
        
        contributions = analysis['component_contributions']
        
        report = f"""
=== RAPPORT D'EXPLICATION - DIAGNOSTIC MÉDICAL ===

PRÉDICTION:
  Classe prédite: {class_names[pred_class]}
  Confiance: {probs[pred_class]:.1%}
  
CONTRIBUTION DES COMPOSANTES GÉOMÉTRIQUES:

1. Scalaires (Intensités de pixels):
   Contribution: {contributions['scalars_contribution']:.1%}
   Interprétation: Représente les niveaux d'intensité bruts de l'image.
   
2. Vecteurs (Gradients spatiaux):
   Contribution: {contributions['vectors_contribution']:.1%}
   Interprétation: Capture les changements d'intensité (bords, contours).
   
3. Bivecteurs (Orientations et textures):
   Contribution: {contributions['bivectors_contribution']:.1%}
   Interprétation: Représente les orientations et les patterns de texture.
   
4. Trivecteur (Relations complexes):
   Contribution: {contributions['trivector_contribution']:.1%}
   Interprétation: Capture les relations géométriques complexes entre régions.

ANALYSE:
"""
        # Identifier la composante la plus importante
        max_comp = max(contributions.items(), key=lambda x: x[1])
        comp_names = {
            'scalars_contribution': 'les intensités de pixels',
            'vectors_contribution': 'les gradients spatiaux',
            'bivectors_contribution': 'les orientations et textures',
            'trivector_contribution': 'les relations géométriques complexes'
        }
        
        report += f"La composante la plus influente est {comp_names[max_comp[0]]} "
        report += f"({max_comp[1]:.1%} de la contribution totale).\n"
        
        if pred_class == 1:  # Tumeur
            report += "\nLe modèle a identifié des caractéristiques géométriques "
            report += "suggérant la présence d'une tumeur.\n"
        else:
            report += "\nLe modèle n'a pas détecté de caractéristiques "
            report += "géométriques anormales suggérant une tumeur.\n"
        
        return report

