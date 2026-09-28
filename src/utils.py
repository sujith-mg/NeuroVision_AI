"""
src/utils.py
Shared utilities for NeuroVision AI
"""

from __future__ import annotations

import sys
from pathlib import Path


# ─────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def get_models_dir() -> Path:
    return PROJECT_ROOT / "models"


def get_assets_dir() -> Path:
    return PROJECT_ROOT / "assets"


def classifier_weights_path() -> Path:
    return get_models_dir() / "classifier_efficientnet.pth"


def segmenter_weights_path() -> Path:
    return get_models_dir() / "segmenter_unet.pth"


# ─────────────────────────────────────────────
# Path setup (add project root to sys.path)
# ─────────────────────────────────────────────
def ensure_project_root_in_path() -> None:
    """
    Ensure PROJECT_ROOT is on sys.path so `src.*` imports work
    when running from sub-directories (e.g., pages/).
    """
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ─────────────────────────────────────────────
# CSS loader
# ─────────────────────────────────────────────
def load_css() -> str:
    """Load assets/styles.css and return as a string."""
    css_path = get_assets_dir() / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def inject_css(st_module) -> None:
    """Inject custom CSS into a Streamlit app."""
    css = load_css()
    if css:
        st_module.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model status helpers
# ─────────────────────────────────────────────
def classifier_model_status() -> dict:
    path = classifier_weights_path()
    return {
        "exists": path.exists(),
        "path":   str(path),
        "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
    }


def segmenter_model_status() -> dict:
    path = segmenter_weights_path()
    return {
        "exists": path.exists(),
        "path":   str(path),
        "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
    }


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────
def format_confidence(value: float) -> str:
    return f"{value * 100:.1f}%"


def get_class_emoji(class_name: str) -> str:
    mapping = {
        "Glioma":          "🔴",
        "Meningioma":      "🟠",
        "Pituitary Tumor": "🟣",
        "Healthy Brain":   "🟢",
    }
    return mapping.get(class_name, "⚪")


def get_class_color(class_name: str) -> str:
    mapping = {
        "Glioma":          "#EF4444",
        "Meningioma":      "#F97316",
        "Pituitary Tumor": "#A855F7",
        "Healthy Brain":   "#22C55E",
    }
    return mapping.get(class_name, "#94A3B8")
