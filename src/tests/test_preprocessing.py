"""
src/tests/test_preprocessing.py
Unit tests for the MRI preprocessing pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import (
    preprocess_mri,
    apply_clahe,
    tensor_to_display,
    CLASSIFIER_SIZE,
    SEGMENTER_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def rgb_image_256():
    """256×256 synthetic RGB image."""
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def gray_image_128():
    """128×128 synthetic grayscale image."""
    arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    return Image.fromarray(arr).convert("L")


# ─────────────────────────────────────────────
# CLAHE tests
# ─────────────────────────────────────────────
class TestCLAHE:
    def test_rgb_output_shape(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.shape == img.shape, "CLAHE must preserve RGB shape"

    def test_rgb_output_dtype(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.dtype == np.uint8, "CLAHE output must be uint8"

    def test_grayscale_output_shape(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.shape == (64, 64), "CLAHE grayscale must preserve 2D shape"


# ─────────────────────────────────────────────
# Preprocessing pipeline tests
# ─────────────────────────────────────────────
class TestPreprocessMRI:
    def test_returns_result_object(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        assert result is not None

    def test_classifier_tensor_shape(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        t = result.classifier_tensor
        assert t.shape == (1, 3, 224, 224), f"Expected [1,3,224,224], got {tuple(t.shape)}"

    def test_segmenter_tensor_shape(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        t = result.segmenter_tensor
        assert t.shape == (1, 1, 256, 256), f"Expected [1,1,256,256], got {tuple(t.shape)}"

    def test_classifier_tensor_dtype(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        import torch
        assert result.classifier_tensor.dtype == torch.float32

    def test_segmenter_tensor_range(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        t = result.segmenter_tensor
        assert float(t.min()) >= 0.0, "Segmenter tensor minimum must be >= 0"
        assert float(t.max()) <= 1.0, "Segmenter tensor maximum must be <= 1"

    def test_display_image_shape(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        assert result.display_image.ndim == 3
        assert result.display_image.shape[2] == 3

    def test_display_clahe_shape(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256)
        assert result.display_clahe.shape == result.display_image.shape

    def test_accepts_pil_grayscale(self, gray_image_128):
        result = preprocess_mri(gray_image_128)
        assert result.classifier_tensor.shape == (1, 3, 224, 224)

    def test_accepts_numpy_array(self):
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocess_mri(arr)
        assert result.classifier_tensor.shape == (1, 3, 224, 224)

    def test_no_clahe_mode(self, rgb_image_256):
        result = preprocess_mri(rgb_image_256, apply_clahe_enhancement=False)
        assert result.classifier_tensor.shape == (1, 3, 224, 224)

    def test_normalisation_applied(self, rgb_image_256):
        """Classifier tensor should NOT be in [0,1] range due to ImageNet normalisation."""
        result = preprocess_mri(rgb_image_256)
        t = result.classifier_tensor
        # After normalisation, values can be negative
        assert float(t.min()) < 0.0 or float(t.max()) > 1.0, \
            "Classifier tensor should be ImageNet-normalised (not raw [0,1])"


# ─────────────────────────────────────────────
# tensor_to_display tests
# ─────────────────────────────────────────────
class TestTensorToDisplay:
    def test_output_shape(self):
        import torch
        t = torch.randn(1, 3, 224, 224)
        result = tensor_to_display(t)
        assert result.shape == (224, 224, 3)

    def test_output_dtype(self):
        import torch
        t = torch.randn(1, 3, 224, 224)
        result = tensor_to_display(t)
        assert result.dtype == np.uint8

    def test_output_range(self):
        import torch
        t = torch.randn(1, 3, 224, 224)
        result = tensor_to_display(t)
        assert result.min() >= 0
        assert result.max() <= 255
