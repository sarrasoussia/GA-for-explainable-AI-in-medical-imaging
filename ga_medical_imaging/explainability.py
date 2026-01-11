"""
Explainability module for identifying which geometric components
influence model decisions.

Provides intrinsic explainability through algebraic component decomposition,
measuring the contribution of scalars, vectors, bivectors, and trivectors
to model predictions.
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
    Explainability analyzer for Geometric Algebra-based models.
    Identifies which geometric components (scalars, vectors, bivectors, trivector)
    influence model predictions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Trained GA model
            device: PyTorch device
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
        Computes the importance of each geometric component of the multivector
        using gradients (Gradient-weighted Class Activation Mapping).
        
        Args:
            images: Image tensor (B, C, H, W)
            target_class: Target class for gradient computation (None = predicted class)
            
        Returns:
            Dict with importance of each component:
            - scalars_importance: importance of scalars (intensities)
            - vectors_importance: importance of vectors (gradients)
            - bivectors_importance: importance of bivectors (orientations/textures)
            - trivector_importance: importance of trivector
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
        Detailed analysis of geometric components and their contribution
        to classification decisions.
        
        Returns:
            Dict with:
            - component_contributions: contribution of each component
            - spatial_importance: spatial importance map
            - geometric_features: extracted geometric features
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
        Visualize model explanations.
        
        Args:
            image: Original image (H, W) or (H, W, C)
            analysis: Results from analyze_geometric_components
            save_path: Path to save the figure
            figsize: Figure size
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2) if image.shape[2] > 1 else image[:, :, 0]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Spatial importance map
        spatial_imp = analysis['spatial_importance'][0]
        im1 = axes[0, 1].imshow(spatial_imp, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Spatial Importance (Multivector Magnitude)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Component contributions
        contributions = analysis['component_contributions']
        comp_names = ['Scalars\n(Intensities)', 'Vectors\n(Gradients)', 
                     'Bivectors\n(Orientations)', 'Trivector\n(Relations)']
        comp_values = [
            contributions['scalars_contribution'],
            contributions['vectors_contribution'],
            contributions['bivectors_contribution'],
            contributions['trivector_contribution']
        ]
        
        axes[0, 2].bar(comp_names, comp_values, color=['blue', 'green', 'orange', 'red'])
        axes[0, 2].set_title('Geometric Component Contributions')
        axes[0, 2].set_ylabel('Relative Importance')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Individual component visualization
        features = analysis['geometric_features']
        
        # Scalars (intensities)
        axes[1, 0].imshow(features['scalars'][0], cmap='gray')
        axes[1, 0].set_title('Scalars (Intensities)')
        axes[1, 0].axis('off')
        
        # Vectors (gradient magnitude)
        vectors_mag = np.linalg.norm(features['vectors'][0], axis=-1)
        im2 = axes[1, 1].imshow(vectors_mag, cmap='viridis', interpolation='bilinear')
        axes[1, 1].set_title('Vectors (Gradients)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Bivectors (magnitude)
        bivectors_mag = np.linalg.norm(features['bivectors'][0], axis=-1)
        im3 = axes[1, 2].imshow(bivectors_mag, cmap='plasma', interpolation='bilinear')
        axes[1, 2].set_title('Bivectors (Orientations/Textures)')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        
        # Add predictions
        probs = analysis['predictions']['probabilities'][0]
        pred_text = f"Probabilities:\n"
        pred_text += f"  Class 0: {probs[0]:.3f}\n"
        pred_text += f"  Class 1: {probs[1]:.3f}"
        axes[0, 2].text(0.5, -0.3, pred_text, transform=axes[0, 2].transAxes,
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_explanation_report(
        self,
        images: torch.Tensor,
        class_names: List[str] = ['Class 0', 'Class 1']
    ) -> str:
        """
        Generate a textual explanation report.
        
        Returns:
            Explanation report as text
        """
        analysis = self.analyze_geometric_components(images)
        
        probs = analysis['predictions']['probabilities'][0]
        pred_class = np.argmax(probs)
        
        contributions = analysis['component_contributions']
        
        report = f"""
=== EXPLANATION REPORT ===

PREDICTION:
  Predicted class: {class_names[pred_class]}
  Confidence: {probs[pred_class]:.1%}
  
GEOMETRIC COMPONENT CONTRIBUTIONS:

1. Scalars (Pixel intensities):
   Contribution: {contributions['scalars_contribution']:.1%}
   Interpretation: Represents raw intensity levels in the image.
   
2. Vectors (Spatial gradients):
   Contribution: {contributions['vectors_contribution']:.1%}
   Interpretation: Captures intensity changes (edges, contours).
   
3. Bivectors (Orientations and textures):
   Contribution: {contributions['bivectors_contribution']:.1%}
   Interpretation: Represents orientations and texture patterns.
   
4. Trivector (Complex relationships):
   Contribution: {contributions['trivector_contribution']:.1%}
   Interpretation: Captures complex geometric relationships between regions.

ANALYSIS:
"""
        # Identify the most important component
        max_comp = max(contributions.items(), key=lambda x: x[1])
        comp_names = {
            'scalars_contribution': 'pixel intensities',
            'vectors_contribution': 'spatial gradients',
            'bivectors_contribution': 'orientations and textures',
            'trivector_contribution': 'complex geometric relationships'
        }
        
        report += f"The most influential component is {comp_names[max_comp[0]]} "
        report += f"({max_comp[1]:.1%} of total contribution).\n"
        
        report += "\nGA-based explanations decompose predictions into algebraic components, "
        report += "allowing direct inspection of contributing factors, unlike post-hoc saliency methods.\n"
        
        return report

