"""
Explainable AI: Grad-CAM (Gradient-weighted Class Activation Mapping)
Implementation for MedVision AI Chest X-ray Interpretability.

Reference: Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization. ICCV 2017.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM class to produce visual saliency heatmaps for chest X-ray predictions.
    Supports DenseNet121, ResNet50, and EfficientNet-B0 architectures.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] contains the gradients with respect to the output feature map
            self.gradients = grad_out[0].detach()

        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fwd, handle_bwd])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap array of shape (H, W) in [0, 1].
        input_tensor: shape [1, 3, H, W]
        target_class_idx: class index to explain. If None, uses top predicting class.
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        # Handle binary classification (single output logit) vs multi-label (num_classes logits)
        if logits.shape[1] == 1:
            score = logits[0, 0]
        else:
            if target_class_idx is None:
                target_class_idx = int(torch.argmax(logits, dim=1).item())
            score = logits[0, target_class_idx]

        # Backward pass to compute gradients
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in Grad-CAM.")

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: [1, C, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]

        # Apply ReLU to retain only features with positive influence on class score
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        original_image_pil: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Resize heatmap to match original image, apply color palette, and blend.
        Returns:
            heatmap_colored_pil: Colored heatmap image (PIL)
            overlay_pil: Heatmap overlaid on original image (PIL)
        """
        w, h = original_image_pil.size
        # Resize heatmap array to match original image resolution
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        # Apply colormap (Jet: blue=cold, green=moderate, red=high importance)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Convert original image to RGB numpy array
        original_np = np.array(original_image_pil.convert("RGB"))

        # Blend
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blended = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * original_np)

        heatmap_pil = Image.fromarray(colored_heatmap)
        overlay_pil = Image.fromarray(blended)

        return heatmap_pil, overlay_pil
