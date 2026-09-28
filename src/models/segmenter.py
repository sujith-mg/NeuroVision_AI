"""
src/models/segmenter.py
U-Net Brain Tumor Segmenter (PyTorch)

Input:  1 × 256 × 256  (grayscale MRI)
Output: 1 × 256 × 256  (binary tumor mask, sigmoid output)

NO fake masks. If weights are missing → is_trained() returns False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SEGMENTER_INPUT_SIZE = (256, 256)
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).parent.parent.parent / "models" / "segmenter_unet.pth"
)


# ─────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Double convolution block: Conv→BN→ReLU → Conv→BN→ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool (returns features and pooled for skip connections)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Upsample + concat skip + ConvBlock"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ (can happen with odd input sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────
class UNet(nn.Module):
    """
    Standard U-Net (encoder-decoder with skip connections).

    Architecture:
      Encoder: 4 × EncoderBlock  (1→64→128→256→512)
      Bottleneck:               (512→1024)
      Decoder: 4 × DecoderBlock (1024→512→256→128→64)
      Head:    Conv 1×1 → sigmoid
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # Output head
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return torch.sigmoid(self.head(x))


# ─────────────────────────────────────────────
# Wrapper class
# ─────────────────────────────────────────────
class BrainTumorSegmenter:
    """
    Wrapper around U-Net for brain tumor segmentation.

    Usage:
        seg = BrainTumorSegmenter()
        if seg.is_trained():
            mask = seg.segment(tensor)
        else:
            print("Segmentation model not trained")
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self._model: Optional[nn.Module] = None
        self._loaded = False

    # ── Status ──────────────────────────────
    def is_trained(self) -> bool:
        """Return True only if weights file actually exists on disk."""
        return self.weights_path.exists()

    # ── Model loading ────────────────────────
    def load_model(self) -> bool:
        """Load trained U-Net weights. Returns False if missing."""
        if not self.is_trained():
            return False

        model = UNet(in_channels=1, out_channels=1)

        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Failed to load segmenter weights: {exc}") from exc

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
    def segment(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run segmentation on a preprocessed grayscale tensor.

        Args:
            tensor: shape [1, 1, 256, 256], float32, values in [0, 1]

        Returns:
            binary_mask: np.ndarray shape [256, 256], dtype uint8, values 0/255

        Raises:
            RuntimeError if model is not trained / weights missing.
        """
        if not self._loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError(
                    "Segmentation model is not trained. "
                    "Train the model first: py -3 train_segmenter.py"
                )

        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self._model(tensor)   # [1, 1, 256, 256]

        prob_map = output.squeeze().cpu().numpy()          # [256, 256]
        binary_mask = (prob_map >= self.threshold).astype(np.uint8) * 255
        return binary_mask

    def segment_with_probs(self, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns both (binary_mask, probability_map).
        binary_mask: uint8 [256,256] values 0/255
        prob_map:    float32 [256,256] values in [0,1]
        """
        if not self._loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError("Segmentation model is not trained.")

        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self._model(tensor)

        prob_map = output.squeeze().cpu().numpy()
        binary_mask = (prob_map >= self.threshold).astype(np.uint8) * 255
        return binary_mask, prob_map
