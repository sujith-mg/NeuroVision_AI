"""
src/models/classifier.py
EfficientNet-B0 Brain Tumor Classifier (PyTorch)

Four classes:
  0 → Glioma
  1 → Meningioma
  2 → Pituitary Tumor
  3 → Healthy Brain

NO fake predictions. If weights are missing → is_trained() returns False.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASSES = ["Glioma", "Meningioma", "Pituitary Tumor", "Healthy Brain"]
NUM_CLASSES = 4
CLASSIFIER_INPUT_SIZE = (224, 224)

DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent.parent / "models" / "classifier_efficientnet.pth"


# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Construct EfficientNet-B0 with a custom 4-class head.
    ImageNet pretrained weights are used for the backbone.
    """
    # Load EfficientNet-B0 backbone with ImageNet weights (for training/init)
    backbone = models.efficientnet_b0(weights=None)

    # Replace classifier head
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone


# ─────────────────────────────────────────────
# Wrapper class
# ─────────────────────────────────────────────
class BrainTumorClassifier:
    """
    Wrapper around EfficientNet-B0 for brain tumor classification.

    Usage:
        clf = BrainTumorClassifier()
        if clf.is_trained():
            result = clf.predict(tensor)
        else:
            print("Model not trained")
    """

    def __init__(self, weights_path: Optional[Path] = None, device: Optional[str] = None):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Optional[nn.Module] = None
        self._loaded = False
        self.classes = CLASSES
        self.num_classes = NUM_CLASSES

    # ── Status ──────────────────────────────
    def is_trained(self) -> bool:
        """Return True only if weights file actually exists on disk."""
        return self.weights_path.exists()

    # ── Model loading ────────────────────────
    def load_model(self) -> bool:
        """
        Load the trained weights.
        Returns True on success, False if weights are missing.
        Raises RuntimeError if file is corrupt.
        """
        if not self.is_trained():
            return False

        model = build_efficientnet_b0(self.num_classes)

        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            # Accept both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Failed to load classifier weights: {exc}") from exc

        model.eval()
        model.to(self.device)
        self._model = model
        self._loaded = True
        return True

    def get_model(self) -> Optional[nn.Module]:
        """Return the loaded nn.Module (or None if not loaded)."""
        if not self._loaded:
            self.load_model()
        return self._model

    # ── Inference ────────────────────────────
    def predict(self, tensor: torch.Tensor) -> dict:
        """
        Run inference on a preprocessed image tensor.

        Args:
            tensor: shape [1, 3, 224, 224], float32, ImageNet-normalised

        Returns:
            {
                "class":       str   — predicted class name,
                "class_index": int   — index 0-3,
                "confidence":  float — probability of predicted class,
                "probabilities": {class_name: probability, ...}
            }

        Raises:
            RuntimeError if model is not loaded / weights missing.
        """
        if not self._loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError(
                    "Classifier model is not trained. "
                    "Train the model first: py -3 train_classifier.py"
                )

        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)          # [1, 4]
            probs = F.softmax(logits, dim=1)[0]   # [4]

        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        probabilities = {cls: float(probs[i].item()) for i, cls in enumerate(self.classes)}

        return {
            "class":         self.classes[class_idx],
            "class_index":   class_idx,
            "confidence":    confidence,
            "probabilities": probabilities,
        }

    def get_probabilities(self, tensor: torch.Tensor) -> dict[str, float]:
        """Convenience: return only the probability dict."""
        return self.predict(tensor)["probabilities"]
