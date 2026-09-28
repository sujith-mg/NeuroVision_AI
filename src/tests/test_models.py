"""
src/tests/test_models.py
Unit tests for classifier, segmenter, and no-fake-prediction behaviour.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classifier import BrainTumorClassifier, build_efficientnet_b0, CLASSES, NUM_CLASSES
from src.models.segmenter import BrainTumorSegmenter, UNet


# ─────────────────────────────────────────────
# EfficientNet-B0 architecture tests
# ─────────────────────────────────────────────
class TestEfficientNetB0Architecture:
    def test_model_builds(self):
        model = build_efficientnet_b0()
        assert model is not None

    def test_output_shape(self):
        model = build_efficientnet_b0()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, NUM_CLASSES), f"Expected [1,{NUM_CLASSES}], got {tuple(out.shape)}"

    def test_num_classes_is_four(self):
        model = build_efficientnet_b0()
        # Last linear layer should have 4 outputs
        head = model.classifier[-1]
        assert head.out_features == 4, "Must have exactly 4 output classes"

    def test_classes_list(self):
        assert len(CLASSES) == 4
        assert "Glioma" in CLASSES
        assert "Meningioma" in CLASSES
        assert "Pituitary Tumor" in CLASSES
        assert "Healthy Brain" in CLASSES


# ─────────────────────────────────────────────
# BrainTumorClassifier wrapper tests
# ─────────────────────────────────────────────
class TestBrainTumorClassifier:
    def test_is_trained_false_when_no_weights(self, tmp_path):
        clf = BrainTumorClassifier(weights_path=tmp_path / "nonexistent.pth")
        assert clf.is_trained() is False, "is_trained() must be False when weights are missing"

    def test_is_trained_true_when_weights_exist(self, tmp_path):
        # Create a real (valid) checkpoint
        model = build_efficientnet_b0()
        weights_path = tmp_path / "classifier_efficientnet.pth"
        torch.save(model.state_dict(), weights_path)

        clf = BrainTumorClassifier(weights_path=weights_path)
        assert clf.is_trained() is True

    def test_predict_raises_when_no_weights(self, tmp_path):
        clf = BrainTumorClassifier(weights_path=tmp_path / "nonexistent.pth")
        dummy = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(dummy)

    def test_predict_returns_correct_structure(self, tmp_path):
        model = build_efficientnet_b0()
        weights_path = tmp_path / "classifier_efficientnet.pth"
        torch.save(model.state_dict(), weights_path)

        clf = BrainTumorClassifier(weights_path=weights_path)
        dummy = torch.randn(1, 3, 224, 224)
        result = clf.predict(dummy)

        assert "class" in result
        assert "class_index" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["class"] in CLASSES
        assert 0 <= result["class_index"] <= 3
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, tmp_path):
        model = build_efficientnet_b0()
        weights_path = tmp_path / "classifier_efficientnet.pth"
        torch.save(model.state_dict(), weights_path)

        clf = BrainTumorClassifier(weights_path=weights_path)
        dummy = torch.randn(1, 3, 224, 224)
        result = clf.predict(dummy)

        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5, f"Probabilities must sum to 1.0, got {total}"

    def test_probabilities_all_four_classes(self, tmp_path):
        model = build_efficientnet_b0()
        weights_path = tmp_path / "classifier_efficientnet.pth"
        torch.save(model.state_dict(), weights_path)

        clf = BrainTumorClassifier(weights_path=weights_path)
        dummy = torch.randn(1, 3, 224, 224)
        result = clf.predict(dummy)

        for cls in CLASSES:
            assert cls in result["probabilities"]

    def test_no_randomness_between_calls(self, tmp_path):
        """Same input must produce same output (deterministic inference)."""
        model = build_efficientnet_b0()
        weights_path = tmp_path / "classifier_efficientnet.pth"
        torch.save(model.state_dict(), weights_path)

        clf = BrainTumorClassifier(weights_path=weights_path)
        dummy = torch.randn(1, 3, 224, 224)
        r1 = clf.predict(dummy)
        r2 = clf.predict(dummy)
        assert r1["class"] == r2["class"]
        assert abs(r1["confidence"] - r2["confidence"]) < 1e-6


# ─────────────────────────────────────────────
# U-Net architecture tests
# ─────────────────────────────────────────────
class TestUNetArchitecture:
    def test_model_builds(self):
        model = UNet(in_channels=1, out_channels=1)
        assert model is not None

    def test_output_shape_256(self):
        model = UNet()
        model.eval()
        dummy = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 1, 256, 256), f"Expected [1,1,256,256], got {tuple(out.shape)}"

    def test_output_range_sigmoid(self):
        model = UNet()
        model.eval()
        dummy = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert float(out.min()) >= 0.0, "U-Net output min must be >= 0 (sigmoid)"
        assert float(out.max()) <= 1.0, "U-Net output max must be <= 1 (sigmoid)"


# ─────────────────────────────────────────────
# BrainTumorSegmenter wrapper tests
# ─────────────────────────────────────────────
class TestBrainTumorSegmenter:
    def test_is_trained_false_when_no_weights(self, tmp_path):
        seg = BrainTumorSegmenter(weights_path=tmp_path / "nonexistent.pth")
        assert seg.is_trained() is False

    def test_is_trained_true_when_weights_exist(self, tmp_path):
        model = UNet()
        weights_path = tmp_path / "segmenter_unet.pth"
        torch.save(model.state_dict(), weights_path)
        seg = BrainTumorSegmenter(weights_path=weights_path)
        assert seg.is_trained() is True

    def test_segment_raises_when_no_weights(self, tmp_path):
        seg = BrainTumorSegmenter(weights_path=tmp_path / "nonexistent.pth")
        dummy = torch.randn(1, 1, 256, 256)
        with pytest.raises(RuntimeError, match="not trained"):
            seg.segment(dummy)

    def test_segment_returns_binary_mask(self, tmp_path):
        model = UNet()
        weights_path = tmp_path / "segmenter_unet.pth"
        torch.save(model.state_dict(), weights_path)

        seg = BrainTumorSegmenter(weights_path=weights_path)
        dummy = torch.randn(1, 1, 256, 256)
        mask = seg.segment(dummy)

        assert mask.shape == (256, 256), f"Mask must be [256,256], got {mask.shape}"
        assert mask.dtype == np.uint8
        unique = np.unique(mask)
        assert all(v in [0, 255] for v in unique), f"Mask must be binary (0/255), got {unique}"
