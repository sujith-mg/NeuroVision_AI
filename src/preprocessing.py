"""
src/preprocessing.py
MRI Image Preprocessing Pipeline

Preprocessing steps:
  1. Open image (PIL)
  2. Produce RGB version for classifier (224×224)
  3. Produce grayscale version for segmenter (256×256)
  4. Apply CLAHE contrast enhancement
  5. Normalize classifier image to ImageNet stats
  6. Return display image + classifier tensor + segmentation tensor
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASSIFIER_SIZE = (224, 224)       # EfficientNet-B0 input
SEGMENTER_SIZE  = (256, 256)       # U-Net input

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# torchvision transform for classifier
_classifier_transform = transforms.Compose([
    transforms.Resize(CLASSIFIER_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# CLAHE helper
# ─────────────────────────────────────────────
def apply_clahe(image_np: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works on grayscale or RGB uint8 images.
    """
    if image_np.ndim == 3:  # RGB
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image_np)


# ─────────────────────────────────────────────
# Main preprocessing function
# ─────────────────────────────────────────────
class PreprocessingResult:
    """Container for all preprocessing outputs."""

    def __init__(
        self,
        display_image: np.ndarray,          # H×W×3 uint8 RGB (original scale)
        display_clahe: np.ndarray,           # H×W×3 uint8 RGB (CLAHE enhanced)
        classifier_tensor: torch.Tensor,     # [1, 3, 224, 224] float32
        segmenter_tensor: torch.Tensor,      # [1, 1, 256, 256] float32
        original_size: tuple[int, int],      # (W, H)
    ):
        self.display_image = display_image
        self.display_clahe = display_clahe
        self.classifier_tensor = classifier_tensor
        self.segmenter_tensor = segmenter_tensor
        self.original_size = original_size


def preprocess_mri(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    apply_clahe_enhancement: bool = True,
) -> PreprocessingResult:
    """
    Full MRI preprocessing pipeline.

    Args:
        image_input:  file path, PIL Image, or numpy array (uint8)
        apply_clahe_enhancement: whether to apply CLAHE before feeding to models

    Returns:
        PreprocessingResult with all tensors and display images
    """
    # ── Step 1: Load image ───────────────────
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    original_size = pil_img.size  # (W, H)

    # ── Step 2: Display image (RGB numpy) ────
    display_np = np.array(pil_img, dtype=np.uint8)  # H×W×3

    # ── Step 3: CLAHE enhancement ────────────
    if apply_clahe_enhancement:
        enhanced_np = apply_clahe(display_np)
    else:
        enhanced_np = display_np.copy()

    display_clahe = enhanced_np

    # ── Step 4: Classifier tensor ─────────────
    # From CLAHE-enhanced image
    enhanced_pil = Image.fromarray(enhanced_np)
    classifier_tensor = _classifier_transform(enhanced_pil).unsqueeze(0)  # [1,3,224,224]

    # ── Step 5: Segmenter tensor ─────────────
    # Grayscale, resized to 256×256, normalised to [0,1]
    gray_np = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2GRAY)         # H×W
    gray_resized = cv2.resize(gray_np, SEGMENTER_SIZE)               # 256×256
    seg_float = gray_resized.astype(np.float32) / 255.0
    segmenter_tensor = torch.from_numpy(seg_float).unsqueeze(0).unsqueeze(0)  # [1,1,256,256]

    return PreprocessingResult(
        display_image=display_np,
        display_clahe=display_clahe,
        classifier_tensor=classifier_tensor,
        segmenter_tensor=segmenter_tensor,
        original_size=original_size,
    )


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised classifier tensor back to a displayable uint8 image.
    Reverses ImageNet normalisation.
    """
    t = tensor.squeeze(0).clone()  # [3, H, W]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t * std + mean
    t = t.clamp(0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img
