"""
src/models/gradcam.py
Grad-CAM explainability for EfficientNet-B0 (pytorch-grad-cam)

Only runs when the classifier is actually loaded.
Returns heatmap and overlay on the original MRI.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


class GradCAMGenerator:
    """
    Generates Grad-CAM heatmaps using the pytorch-grad-cam library.

    Targets the last convolutional block of EfficientNet-B0
    (features[-1] → the final MBConv block).

    Usage:
        gen = GradCAMGenerator(classifier.get_model(), device)
        heatmap, overlay = gen.generate(tensor, display_rgb, class_idx)
    """

    def __init__(self, model: nn.Module, device: torch.device):
        if not GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam is not installed. "
                "Run: pip install grad-cam"
            )
        self.model = model
        self.device = device

        # Target the last convolutional block of EfficientNet-B0
        # EfficientNet-B0 features[-1] is the final MBConv block
        self.target_layers = [model.features[-1]]

    def generate(
        self,
        tensor: torch.Tensor,
        display_image_rgb: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap and overlay.

        Args:
            tensor:             [1, 3, 224, 224] classifier input tensor
            display_image_rgb:  H×W×3 uint8 RGB image for overlay
            class_idx:          target class index (None → argmax)

        Returns:
            heatmap:  H×W×3 uint8 RGB heatmap (jet colormap)
            overlay:  H×W×3 uint8 RGB overlay on original image
        """
        tensor = tensor.to(self.device)

        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None

        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)

        grayscale_cam = grayscale_cam[0]  # [H, W] float in [0,1]

        # Resize cam to match display image
        h, w = display_image_rgb.shape[:2]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))

        # Create colormap heatmap
        heatmap_uint8 = np.uint8(255 * grayscale_cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Create overlay
        display_float = display_image_rgb.astype(np.float32) / 255.0
        overlay = show_cam_on_image(display_float, grayscale_cam_resized, use_rgb=True)

        return heatmap_rgb, overlay


def is_gradcam_available() -> bool:
    """Check if pytorch-grad-cam is importable."""
    return GRADCAM_AVAILABLE
