"""
src/visualization.py
Visualization utilities for NeuroVision AI

Provides:
  - Plotly probability chart
  - Grad-CAM overlay display
  - Segmentation overlay
  - Result card rendering helpers
"""

from __future__ import annotations

import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image


# ─────────────────────────────────────────────
# Class colour palette
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "Glioma":          "#EF4444",   # red
    "Meningioma":      "#F97316",   # orange
    "Pituitary Tumor": "#A855F7",   # purple
    "Healthy Brain":   "#22C55E",   # green
}

ALL_CLASSES = ["Glioma", "Meningioma", "Pituitary Tumor", "Healthy Brain"]


# ─────────────────────────────────────────────
# Plotly probability chart
# ─────────────────────────────────────────────
def make_probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """
    Create a horizontal bar chart showing class probabilities.

    Args:
        probabilities: {class_name: probability (0-1)}

    Returns:
        plotly Figure
    """
    classes = ALL_CLASSES
    values  = [probabilities.get(c, 0.0) for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]
    pct     = [v * 100 for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pct,
        y=classes,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{p:.1f}%" for p in pct],
        textposition="outside",
        textfont=dict(color="white", size=13),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        cliponaxis=False,
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter, sans-serif"),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 115],
            gridcolor="rgba(255,255,255,0.1)",
            showline=False,
            tickfont=dict(color="rgba(255,255,255,0.7)"),
        ),
        yaxis=dict(
            tickfont=dict(color="white", size=13),
            showgrid=False,
        ),
        margin=dict(l=10, r=60, t=20, b=40),
        height=240,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# Segmentation overlay
# ─────────────────────────────────────────────
def overlay_mask_on_image(
    original_rgb: np.ndarray,
    mask_binary: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary segmentation mask over an RGB image.

    Args:
        original_rgb: H×W×3 uint8
        mask_binary:  H×W uint8 (0 or 255)
        color:        RGB tuple for mask overlay
        alpha:        blend factor for the mask region

    Returns:
        H×W×3 uint8 composite image
    """
    img = original_rgb.copy().astype(np.float32)

    # Resize mask to match image if needed
    if mask_binary.shape[:2] != original_rgb.shape[:2]:
        mask_resized = cv2.resize(mask_binary, (original_rgb.shape[1], original_rgb.shape[0]))
    else:
        mask_resized = mask_binary

    mask_bool = mask_resized > 127

    overlay = img.copy()
    overlay[mask_bool] = (
        (1 - alpha) * img[mask_bool]
        + alpha * np.array(color, dtype=np.float32)
    )

    # Draw contour
    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, color, 2)

    return overlay_uint8


def mask_to_rgb(mask_binary: np.ndarray, color: tuple[int, int, int] = (0, 255, 128)) -> np.ndarray:
    """Convert binary mask to coloured RGB for display."""
    h, w = mask_binary.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask_binary > 127] = color
    return rgb


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────
def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert H×W×3 uint8 numpy array to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def resize_for_display(arr: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Resize image so the longest side is max_size pixels."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_size:
        return arr
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(arr, (new_w, new_h))
