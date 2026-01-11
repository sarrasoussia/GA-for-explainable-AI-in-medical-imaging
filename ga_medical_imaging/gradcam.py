"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN models.

This module provides post-hoc explainability for CNN baselines,
enabling direct comparison with GA intrinsic explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    
    Provides post-hoc explainability for comparison with GA intrinsic explanations.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Trained CNN model
            target_layer: Target convolutional layer to generate CAM from
                         (typically the last or second-to-last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def save_activation(module, input, output):
            """Save activations during forward pass."""
            self.activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            """Save gradients during backward pass."""
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W) or (1, C, H, W)
            target_class: Target class for gradient computation (None = predicted class)
            retain_graph: Whether to retain computation graph
        
        Returns:
            cam: (H, W) numpy array with importance scores (normalized to [0, 1])
        """
        self.model.eval()
        
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # Compute gradient for target class
        if output.dim() == 2:
            score = output[0, target_class]
        else:
            score = output[0, target_class]
        
        score.backward(retain_graph=retain_graph)
        
        # Check if we have gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. "
                "Make sure the target layer is a convolutional layer that was used in forward pass."
            )
        
        # Get gradients and activations for the first (and only) sample
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        
        return cam_np
    
    def generate_cam_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for a batch of images.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_classes: List of target classes (None = predicted classes)
        
        Returns:
            cams: (B, H, W) numpy array with importance scores
        """
        batch_size = input_tensor.shape[0]
        cams = []
        
        for i in range(batch_size):
            single_image = input_tensor[i:i+1]
            target_class = target_classes[i] if target_classes is not None else None
            cam = self.generate_cam(single_image, target_class)
            cams.append(cam)
        
        return np.stack(cams)
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image (H, W) or (H, W, C) in range [0, 1] or [0, 255]
            cam: Grad-CAM heatmap (H, W) in range [0, 1]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap (cv2.COLORMAP_JET, cv2.COLORMAP_HOT, etc.)
        
        Returns:
            Overlaid image (H, W, 3) in range [0, 255]
        """
        # Ensure image is in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize CAM to match image size if needed
        if cam.shape != image.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid


def find_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Find a suitable target layer for Grad-CAM.
    
    Args:
        model: CNN model
        layer_name: Name of specific layer to use (None = auto-detect)
    
    Returns:
        Target layer module
    """
    if layer_name is not None:
        # Find layer by name
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Auto-detect: find the last convolutional layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("No convolutional layer found in model")
    
    return last_conv


def create_gradcam_for_cnn(
    model: nn.Module,
    layer_name: Optional[str] = None
) -> GradCAM:
    """
    Create Grad-CAM instance for a CNN model.
    
    Args:
        model: Trained CNN model
        layer_name: Name of target layer (None = auto-detect last conv layer)
    
    Returns:
        GradCAM instance
    """
    target_layer = find_target_layer(model, layer_name)
    return GradCAM(model, target_layer)

