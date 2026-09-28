"""
pages/2_🔬_Analysis.py — NeuroVision AI Main Analysis Page

Full pipeline:
  Upload MRI → Preprocess → EfficientNet-B0 → Grad-CAM → U-Net segmentation
  Shows Model-Not-Trained state when weights are missing.
  NEVER produces fake predictions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image

from src.utils import (
    classifier_model_status,
    segmenter_model_status,
    format_confidence,
    get_class_emoji,
    get_class_color,
)
from src.preprocessing import preprocess_mri
from src.visualization import (
    make_probability_chart,
    overlay_mask_on_image,
    mask_to_rgb,
    numpy_to_pil,
    resize_for_display,
)
from src.models.classifier import BrainTumorClassifier
from src.models.segmenter import BrainTumorSegmenter
from src.models.gradcam import GradCAMGenerator, is_gradcam_available

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Analysis — NeuroVision AI",
    page_icon="🔬",
    layout="wide",
)

css_path = PROJECT_ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-weight:700;font-size:1.1rem;
            background:linear-gradient(135deg,#00D4FF,#7C3AED);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            NeuroVision AI
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    clf_status = classifier_model_status()
    seg_status = segmenter_model_status()

    st.markdown("**Model Status**")
    if clf_status["exists"]:
        st.success(f"✅ Classifier ({clf_status['size_mb']} MB)")
    else:
        st.error("❌ Classifier: Not Trained")

    if seg_status["exists"]:
        st.success(f"✅ Segmenter ({seg_status['size_mb']} MB)")
    else:
        st.error("❌ Segmenter: Not Trained")

    st.markdown("---")
    st.markdown("""
    <div class="nv-disclaimer" style="font-size:0.72rem;">
        ⚠️ <b>Academic prototype only.</b><br>
        Not for medical diagnosis or clinical use.
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2rem; font-weight:700;
    background:linear-gradient(135deg,#00D4FF,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    🔬 MRI Analysis
</h1>
<p style="color:#94A3B8; margin-top:-0.5rem;">
    Upload a brain MRI image to run the full AI detection pipeline.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nv-disclaimer">
    ⚠️ <b>Academic prototype.</b> Results are <b>NOT medical diagnoses</b>.
    This tool is for educational demonstration only.
</div>
""", unsafe_allow_html=True)

# ── Upload section ───────────────────────────────────────────
st.markdown("### 📁 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an MRI image file",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Upload a brain MRI scan image (PNG, JPG, JPEG, or BMP format)",
)

if uploaded_file is None:
    st.markdown("""
    <div class="nv-card" style="text-align:center; padding:3rem; border-style:dashed;">
        <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
        <div style="font-size:1.1rem; color:#94A3B8; margin-bottom:0.5rem;">
            No MRI image uploaded yet
        </div>
        <div style="font-size:0.85rem; color:#475569;">
            Upload a PNG, JPG, JPEG, or BMP brain MRI image to begin analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Image loaded — run pipeline ──────────────────────────────
try:
    pil_image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"❌ Could not read the uploaded image: {e}")
    st.stop()

# ── Preprocessing ─────────────────────────────────────────────
with st.spinner("🔧 Preprocessing MRI image..."):
    try:
        prep = preprocess_mri(pil_image, apply_clahe_enhancement=True)
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

# ── Show original + preprocessed ────────────────────────────
st.markdown("---")
st.markdown("### 🖼️ Image Preview")

col_orig, col_clahe = st.columns(2)
with col_orig:
    st.image(
        numpy_to_pil(resize_for_display(prep.display_image, 400)),
        caption="Original MRI (Uploaded)",
        use_container_width=True,
    )
with col_clahe:
    st.image(
        numpy_to_pil(resize_for_display(prep.display_clahe, 400)),
        caption="Preprocessed MRI (CLAHE Enhanced)",
        use_container_width=True,
    )

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ════════════════════════════════════════════════════════════
st.markdown("### 🤖 Classification — EfficientNet-B0")

clf_status = classifier_model_status()

if not clf_status["exists"]:
    # ── MODEL NOT TRAINED state ──────────────────────────────
    st.markdown("""
    <div class="nv-not-trained">
        <div class="nv-not-trained-icon">⚠️</div>
        <div class="nv-not-trained-title">CLASSIFIER MODEL NOT TRAINED</div>
        <div style="color:#FCA5A5; font-size:0.9rem; margin-bottom:1rem;">
            The file <code>models/classifier_efficientnet.pth</code> was not found.
        </div>
        <div style="color:#94A3B8; font-size:0.85rem; text-align:left; max-width:600px; margin:0 auto;">
            <b>To train the classifier:</b><br>
            1. Prepare dataset in <code>dataset/Training/{glioma,meningioma,pituitary,notumor}/</code><br>
            2. Run: <code>py -3 train_classifier.py</code><br>
            3. Trained weights will be saved to <code>models/classifier_efficientnet.pth</code><br>
            4. Reload this page to run inference
        </div>
    </div>
    """, unsafe_allow_html=True)
    classification_result = None
else:
    # ── Run inference ────────────────────────────────────────
    with st.spinner("🤖 Running EfficientNet-B0 classification..."):
        try:
            classifier = BrainTumorClassifier()
            classification_result = classifier.predict(prep.classifier_tensor)
        except Exception as e:
            st.error(f"❌ Classification error: {e}")
            classification_result = None

    if classification_result is not None:
        pred_class  = classification_result["class"]
        confidence  = classification_result["confidence"]
        probs       = classification_result["probabilities"]
        class_emoji = get_class_emoji(pred_class)
        class_color = get_class_color(pred_class)

        # Result card
        st.markdown(f"""
        <div class="nv-result-card">
            <div style="font-size:0.8rem; color:#64748B; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:0.4rem;">
                Predicted Class
            </div>
            <div class="nv-result-class" style="color:{class_color};">
                {class_emoji} {pred_class}
            </div>
            <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:1rem;">
                Confidence: <b style="color:{class_color};">{format_confidence(confidence)}</b>
            </div>
            <div style="background:rgba(255,255,255,0.05); border-radius:8px;
                        height:8px; overflow:hidden; margin-bottom:0.4rem;">
                <div style="width:{confidence*100:.1f}%; height:100%;
                             background:linear-gradient(90deg,{class_color},{class_color}88);
                             border-radius:8px; transition:width 0.5s ease;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability chart
        st.markdown("**All Class Probabilities**")
        fig = make_probability_chart(probs)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Raw values in expander
        with st.expander("📋 Raw probability values"):
            for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                emoji = get_class_emoji(cls)
                color = get_class_color(cls)
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;"
                    f"border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    f"<span>{emoji} {cls}</span>"
                    f"<b style='color:{color};'>{prob*100:.2f}%</b></div>",
                    unsafe_allow_html=True,
                )

# ════════════════════════════════════════════════════════════
#  GRAD-CAM
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔥 Grad-CAM Explainability")

if not clf_status["exists"]:
    st.info("ℹ️ Grad-CAM requires the classifier model to be trained first.")
elif classification_result is None:
    st.warning("⚠️ Classification did not complete — Grad-CAM unavailable.")
elif not is_gradcam_available():
    st.warning("⚠️ `grad-cam` package not installed. Run: `pip install grad-cam`")
else:
    with st.spinner("🔥 Generating Grad-CAM heatmap..."):
        try:
            import torch
            # Need fresh model for Grad-CAM (must be in eval mode with grads)
            from src.models.classifier import build_efficientnet_b0, DEFAULT_WEIGHTS_PATH
            model_for_cam = build_efficientnet_b0()
            state = torch.load(DEFAULT_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model_for_cam.load_state_dict(state)
            model_for_cam.eval()

            cam_gen = GradCAMGenerator(model_for_cam, torch.device("cpu"))
            display_for_cam = resize_for_display(prep.display_clahe, 224)
            heatmap, overlay = cam_gen.generate(
                prep.classifier_tensor,
                display_for_cam,
                class_idx=classification_result["class_index"],
            )

            col_cam1, col_cam2 = st.columns(2)
            with col_cam1:
                st.image(
                    numpy_to_pil(heatmap),
                    caption="Grad-CAM Heatmap (attention regions)",
                    use_container_width=True,
                )
            with col_cam2:
                st.image(
                    numpy_to_pil(overlay),
                    caption="MRI + Grad-CAM Overlay",
                    use_container_width=True,
                )

            st.markdown("""
            <div style="font-size:0.8rem; color:#64748B; margin-top:0.5rem;">
                🔥 <b>Grad-CAM</b> highlights regions the model focused on for this prediction.
                Warmer colours (red/yellow) indicate higher attention.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"⚠️ Grad-CAM generation failed: {e}")

# ════════════════════════════════════════════════════════════
#  SEGMENTATION
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🎯 Tumor Segmentation — U-Net")

seg_status = segmenter_model_status()

if not seg_status["exists"]:
    st.markdown("""
    <div class="nv-not-trained">
        <div class="nv-not-trained-icon">⚠️</div>
        <div class="nv-not-trained-title">SEGMENTATION MODEL NOT TRAINED</div>
        <div style="color:#FCA5A5; font-size:0.9rem; margin-bottom:1rem;">
            The file <code>models/segmenter_unet.pth</code> was not found.
        </div>
        <div style="color:#94A3B8; font-size:0.85rem; text-align:left; max-width:600px; margin:0 auto;">
            <b>To train the segmenter:</b><br>
            1. Prepare segmentation dataset in <code>segmentation_dataset/{images,masks}/</code><br>
            2. Run: <code>py -3 train_segmenter.py</code><br>
            3. Trained weights will be saved to <code>models/segmenter_unet.pth</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.spinner("🎯 Running U-Net segmentation..."):
        try:
            segmenter = BrainTumorSegmenter()
            mask = segmenter.segment(prep.segmenter_tensor)

            # Resize display image to 256×256 for overlay
            display_256 = np.array(
                numpy_to_pil(prep.display_clahe).resize((256, 256))
            )
            seg_overlay = overlay_mask_on_image(display_256, mask)
            mask_rgb    = mask_to_rgb(mask, color=(0, 255, 128))

            col_seg1, col_seg2, col_seg3 = st.columns(3)
            with col_seg1:
                st.image(
                    numpy_to_pil(display_256),
                    caption="Input MRI (256×256)",
                    use_container_width=True,
                )
            with col_seg2:
                st.image(
                    numpy_to_pil(mask_rgb),
                    caption="Predicted Tumor Mask",
                    use_container_width=True,
                )
            with col_seg3:
                st.image(
                    numpy_to_pil(seg_overlay),
                    caption="MRI + Segmentation Overlay",
                    use_container_width=True,
                )

            # Mask statistics
            tumor_pixels = int(np.sum(mask > 127))
            total_pixels = mask.size
            tumor_pct    = (tumor_pixels / total_pixels) * 100

            st.markdown(f"""
            <div class="nv-result-card" style="margin-top:0.5rem;">
                <div style="display:flex; gap:2rem; flex-wrap:wrap;">
                    <div>
                        <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Tumor Pixels</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#10B981;">{tumor_pixels:,}</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Coverage</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#10B981;">{tumor_pct:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">Mask Size</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#00D4FF;">256×256</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Segmentation error: {e}")

# ── Footer disclaimer ────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="nv-disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> All outputs shown above are generated by an untrained/experimental
    AI model for academic demonstration purposes only. They do <b>NOT</b> constitute medical advice,
    diagnosis, or clinical assessment. Do not use this tool for any medical decision-making.
</div>
""", unsafe_allow_html=True)
