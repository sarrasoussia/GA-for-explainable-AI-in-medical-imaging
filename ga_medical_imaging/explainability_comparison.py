"""
Compare GA intrinsic explanations with CNN post-hoc explanations (Grad-CAM).

This module enables direct comparison between:
- GA: Intrinsic explainability through algebraic component decomposition
- Grad-CAM: Post-hoc explainability through gradient-weighted activations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import pearsonr

from .explainability import GAExplainabilityAnalyzer
from .gradcam import GradCAM


class ExplainabilityComparator:
    """
    Compare GA intrinsic explanations with CNN post-hoc explanations (Grad-CAM).
    
    This enables quantitative and qualitative comparison of explainability methods,
    demonstrating GA advantages in interpretability and stability.
    """
    
    def __init__(
        self,
        ga_model: nn.Module,
        cnn_model: nn.Module,
        gradcam: GradCAM,
        device: str = 'cpu'
    ):
        """
        Args:
            ga_model: Trained GA model
            cnn_model: Trained CNN model
            gradcam: GradCAM instance for CNN model
            device: PyTorch device
        """
        self.ga_model = ga_model
        self.cnn_model = cnn_model
        self.gradcam = gradcam
        self.device = device
        
        self.ga_analyzer = GAExplainabilityAnalyzer(ga_model, device)
    
    def compare_explanations(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Generate both GA and Grad-CAM explanations for the same image.
        
        Args:
            image_tensor: Input image tensor (B, C, H, W) or (1, C, H, W)
            target_class: Target class for explanation (None = predicted class)
        
        Returns:
            Dictionary with:
            - ga: GA explanation results
            - gradcam: Grad-CAM explanation results
            - comparison_metrics: Quantitative comparison metrics
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # GA explanation
        ga_analysis = self.ga_analyzer.analyze_geometric_components(image_tensor)
        ga_importance = ga_analysis['spatial_importance'][0]  # (H, W)
        
        # Get prediction for target class
        with torch.no_grad():
            ga_logits = self.ga_model(image_tensor)
            if target_class is None:
                target_class = ga_logits.argmax(dim=1).item()
        
        # Grad-CAM explanation
        gradcam_heatmap = self.gradcam.generate_cam(image_tensor, target_class)
        
        # Resize to same size if needed
        if ga_importance.shape != gradcam_heatmap.shape:
            gradcam_heatmap = self._resize_to_match(gradcam_heatmap, ga_importance.shape)
        
        # Compute comparison metrics
        comparison_metrics = self._compute_comparison_metrics(
            ga_importance, gradcam_heatmap
        )
        
        return {
            'ga': {
                'component_contributions': ga_analysis['component_contributions'],
                'spatial_importance': ga_importance,
                'explanation_type': 'intrinsic',
                'geometric_features': ga_analysis['geometric_features']
            },
            'gradcam': {
                'heatmap': gradcam_heatmap,
                'explanation_type': 'post_hoc'
            },
            'comparison_metrics': comparison_metrics,
            'target_class': target_class
        }
    
    def _resize_to_match(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize array to match target shape."""
        if array.shape == target_shape:
            return array
        
        zoom_factors = [
            target_shape[0] / array.shape[0],
            target_shape[1] / array.shape[1]
        ]
        return zoom(array, zoom_factors, order=1)
    
    def _compute_comparison_metrics(
        self,
        ga_importance: np.ndarray,
        gradcam_heatmap: np.ndarray
    ) -> Dict:
        """
        Compute quantitative comparison metrics between GA and Grad-CAM.
        
        Metrics:
        - spatial_correlation: Pearson correlation between heatmaps
        - sparsity_ga: Entropy of GA importance distribution
        - sparsity_gradcam: Entropy of Grad-CAM distribution
        - overlap: Intersection over Union of top-k% important regions
        """
        # Flatten for correlation
        ga_flat = ga_importance.flatten()
        gradcam_flat = gradcam_heatmap.flatten()
        
        # Normalize
        ga_norm = (ga_flat - ga_flat.min()) / (ga_flat.max() - ga_flat.min() + 1e-8)
        gradcam_norm = (gradcam_flat - gradcam_flat.min()) / (gradcam_flat.max() - gradcam_flat.min() + 1e-8)
        
        # Spatial correlation
        correlation, p_value = pearsonr(ga_norm, gradcam_norm)
        
        # Sparsity (entropy)
        def compute_sparsity(arr):
            arr_norm = arr / (arr.sum() + 1e-8)
            arr_norm = arr_norm[arr_norm > 1e-8]  # Remove zeros
            if len(arr_norm) == 0:
                return 0.0
            return -np.sum(arr_norm * np.log(arr_norm + 1e-8))
        
        sparsity_ga = compute_sparsity(ga_flat)
        sparsity_gradcam = compute_sparsity(gradcam_flat)
        
        # Overlap (IoU of top 20% important regions)
        k = 0.2
        ga_threshold = np.percentile(ga_importance, (1 - k) * 100)
        gradcam_threshold = np.percentile(gradcam_heatmap, (1 - k) * 100)
        
        ga_mask = ga_importance >= ga_threshold
        gradcam_mask = gradcam_heatmap >= gradcam_threshold
        
        intersection = np.logical_and(ga_mask, gradcam_mask).sum()
        union = np.logical_or(ga_mask, gradcam_mask).sum()
        iou = intersection / (union + 1e-8)
        
        return {
            'spatial_correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'sparsity_ga': float(sparsity_ga),
            'sparsity_gradcam': float(sparsity_gradcam),
            'overlap_iou': float(iou),
            'overlap_k': k
        }
    
    def compare_explanation_stability(
        self,
        image_tensor: torch.Tensor,
        perturbations: Dict[str, callable],
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Compare explanation stability under perturbations.
        
        GA explanations should be more stable (intrinsic vs post-hoc).
        
        Args:
            image_tensor: Original image tensor
            perturbations: Dictionary of {name: perturbation_function}
            target_class: Target class for explanation
        
        Returns:
            Dictionary with stability metrics for both methods
        """
        # Original explanations
        original_comparison = self.compare_explanations(image_tensor, target_class)
        original_ga = original_comparison['ga']['spatial_importance']
        original_gradcam = original_comparison['gradcam']['heatmap']
        
        stability_scores_ga = []
        stability_scores_gradcam = []
        
        for pert_name, pert_func in perturbations.items():
            # Apply perturbation
            pert_image = pert_func(image_tensor)
            
            # Get explanations for perturbed image
            pert_comparison = self.compare_explanations(pert_image, target_class)
            pert_ga = pert_comparison['ga']['spatial_importance']
            pert_gradcam = pert_comparison['gradcam']['heatmap']
            
            # Resize if needed
            if pert_ga.shape != original_ga.shape:
                pert_ga = self._resize_to_match(pert_ga, original_ga.shape)
            if pert_gradcam.shape != original_gradcam.shape:
                pert_gradcam = self._resize_to_match(pert_gradcam, original_gradcam.shape)
            
            # Compute correlation (stability metric)
            ga_corr, _ = pearsonr(original_ga.flatten(), pert_ga.flatten())
            gradcam_corr, _ = pearsonr(original_gradcam.flatten(), pert_gradcam.flatten())
            
            stability_scores_ga.append(ga_corr)
            stability_scores_gradcam.append(gradcam_corr)
        
        return {
            'ga_stability': {
                'mean': float(np.mean(stability_scores_ga)),
                'std': float(np.std(stability_scores_ga)),
                'scores': [float(s) for s in stability_scores_ga]
            },
            'gradcam_stability': {
                'mean': float(np.mean(stability_scores_gradcam)),
                'std': float(np.std(stability_scores_gradcam)),
                'scores': [float(s) for s in stability_scores_gradcam]
            },
            'stability_advantage': float(np.mean(stability_scores_ga) - np.mean(stability_scores_gradcam)),
            'perturbations': list(perturbations.keys())
        }
    
    def visualize_comparison(
        self,
        image: np.ndarray,
        comparison_result: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12)
    ):
        """
        Create side-by-side comparison visualization.
        
        Shows:
        - Original image
        - GA spatial importance
        - Grad-CAM heatmap
        - Overlay comparisons
        - Component contributions (GA only)
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Ensure image is 2D
        if len(image.shape) == 3:
            image = np.mean(image, axis=2) if image.shape[2] > 1 else image[:, :, 0]
        
        ga_importance = comparison_result['ga']['spatial_importance']
        gradcam_heatmap = comparison_result['gradcam']['heatmap']
        metrics = comparison_result['comparison_metrics']
        contributions = comparison_result['ga']['component_contributions']
        
        # Row 1: Original, GA importance, Grad-CAM
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(ga_importance, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('GA Spatial Importance (Intrinsic)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        im2 = axes[0, 2].imshow(gradcam_heatmap, cmap='hot', interpolation='bilinear')
        axes[0, 2].set_title('Grad-CAM Heatmap (Post-hoc)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Row 2: Overlays and component analysis
        # GA overlay
        ga_overlay = self._overlay_heatmap(image, ga_importance, alpha=0.4)
        axes[1, 0].imshow(ga_overlay)
        axes[1, 0].set_title('GA Overlay', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Grad-CAM overlay
        gradcam_overlay = self._overlay_heatmap(image, gradcam_heatmap, alpha=0.4)
        axes[1, 1].imshow(gradcam_overlay)
        axes[1, 1].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Component contributions
        comp_names = ['Scalars', 'Vectors', 'Bivectors', 'Trivector']
        comp_values = [
            contributions['scalars_contribution'],
            contributions['vectors_contribution'],
            contributions['bivectors_contribution'],
            contributions['trivector_contribution']
        ]
        
        axes[1, 2].bar(comp_names, comp_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 2].set_title('GA Component Contributions', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('Relative Importance')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add metrics text
        metrics_text = (
            f"Correlation: {metrics['spatial_correlation']:.3f}\n"
            f"IoU (top 20%): {metrics['overlap_iou']:.3f}\n"
            f"GA Sparsity: {metrics['sparsity_ga']:.3f}\n"
            f"Grad-CAM Sparsity: {metrics['sparsity_gradcam']:.3f}"
        )
        axes[1, 2].text(0.5, -0.3, metrics_text, transform=axes[1, 2].transAxes,
                       ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.suptitle('GA vs Grad-CAM Explanation Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison visualization saved to: {save_path}")
        
        return fig
    
    def _overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay heatmap on image."""
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Resize heatmap if needed
        if heatmap.shape != image.shape:
            heatmap = self._resize_to_match(heatmap, image.shape)
        
        # Create RGB image
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=2)
        else:
            image_rgb = image
        
        # Create heatmap colormap
        try:
            import matplotlib.cm as cm
            colormap = cm.get_cmap('hot')
        except AttributeError:
            # For newer matplotlib versions
            import matplotlib.pyplot as plt
            colormap = plt.cm.get_cmap('hot')
        heatmap_rgb = colormap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Overlay
        overlaid = alpha * heatmap_rgb + (1 - alpha) * image_rgb
        return np.clip(overlaid, 0, 1)

