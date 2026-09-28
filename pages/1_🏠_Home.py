"""
pages/1_🏠_Home.py — NeuroVision AI Overview Page
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils import inject_css, classifier_model_status, segmenter_model_status

st.set_page_config(
    page_title="Home — NeuroVision AI",
    page_icon="🏠",
    layout="wide",
)

css_path = PROJECT_ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Sidebar disclaimer ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-weight:700; font-size:1.1rem;
            background:linear-gradient(135deg,#00D4FF,#7C3AED);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            NeuroVision AI
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="nv-disclaimer" style="font-size:0.72rem;">
        ⚠️ <b>Academic prototype only.</b><br>
        Not for medical diagnosis or clinical use.
    </div>
    """, unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="nv-hero">
    <div class="nv-title">🧠 NeuroVision AI</div>
    <div class="nv-subtitle">AI-Based Brain Tumor Detection, Segmentation &amp; Classification</div>
    <div style="max-width:680px; margin:0 auto; color:#64748B; font-size:0.9rem; line-height:1.7;">
        A complete deep learning pipeline that combines <b style="color:#00D4FF">EfficientNet-B0</b>
        transfer learning for classification, <b style="color:#7C3AED">U-Net</b> for tumor segmentation,
        and <b style="color:#10B981">Grad-CAM</b> for visual explainability — all in a single interactive
        Streamlit interface.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model status indicators ─────────────────────────────────
clf_status = classifier_model_status()
seg_status = segmenter_model_status()

col_s1, col_s2 = st.columns(2)
with col_s1:
    if clf_status["exists"]:
        st.success(f"✅ **Classifier Model** loaded ({clf_status['size_mb']} MB)")
    else:
        st.warning("⚠️ **Classifier Model** not trained — see Model Info page")
with col_s2:
    if seg_status["exists"]:
        st.success(f"✅ **Segmentation Model** loaded ({seg_status['size_mb']} MB)")
    else:
        st.warning("⚠️ **Segmentation Model** not trained — see Model Info page")

st.markdown("---")

# ── Feature cards ───────────────────────────────────────────
st.markdown("### 🔑 Key Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">🧠</span>
        <div class="nv-feature-title">4-Class Classification</div>
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            EfficientNet-B0 trained on MRI scans detects:<br>
            <span style="color:#EF4444">● Glioma</span><br>
            <span style="color:#F97316">● Meningioma</span><br>
            <span style="color:#A855F7">● Pituitary Tumor</span><br>
            <span style="color:#22C55E">● Healthy Brain</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">🎯</span>
        <div class="nv-feature-title">Tumor Segmentation</div>
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            Custom U-Net encoder-decoder with skip connections
            produces pixel-level tumor masks, enabling precise
            localization of tumor boundaries in MRI scans.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">🔥</span>
        <div class="nv-feature-title">Explainable AI (XAI)</div>
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
            Grad-CAM generates visual attention heatmaps showing
            <em>which regions</em> of the MRI influenced the model's
            classification decision — making AI decisions interpretable.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Architecture diagram ────────────────────────────────────
st.markdown("### 🏗️ System Architecture")

tab1, tab2 = st.tabs(["📋 Classification Pipeline", "🎯 Segmentation Pipeline"])

with tab1:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; gap:0.4rem; padding:1.5rem;">
        <div class="nv-flow-node" style="background:rgba(0,212,255,0.1); min-width:180px;">📁 MRI Image (PNG/JPG)</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="min-width:180px;">🔧 Preprocessing + CLAHE</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="background:rgba(124,58,237,0.15); border-color:#7C3AED; color:#A78BFA; min-width:180px;">
            🤖 EfficientNet-B0 (224×224 RGB)
        </div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="min-width:180px;">📊 Softmax → Class Probabilities</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div style="display:flex; gap:1rem; align-items:flex-start;">
            <div style="display:flex; flex-direction:column; align-items:center; gap:0.4rem;">
                <div class="nv-flow-node" style="background:rgba(239,68,68,0.1); border-color:#EF4444; color:#F87171;">
                    🔴 Predicted Class
                </div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; gap:0.4rem;">
                <div class="nv-flow-node" style="background:rgba(16,185,129,0.1); border-color:#10B981; color:#34D399;">
                    🔥 Grad-CAM Heatmap
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; gap:0.4rem; padding:1.5rem;">
        <div class="nv-flow-node" style="background:rgba(0,212,255,0.1); min-width:180px;">📁 MRI Image (PNG/JPG)</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="min-width:180px;">🔧 Grayscale + CLAHE (256×256)</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="background:rgba(124,58,237,0.15); border-color:#7C3AED; color:#A78BFA; min-width:240px;">
            🏗️ U-Net (Encoder → Bottleneck → Decoder)
        </div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="min-width:180px;">🎯 Sigmoid → Binary Mask</div>
        <div style="color:#64748B; font-size:1.2rem;">↓</div>
        <div class="nv-flow-node" style="background:rgba(16,185,129,0.1); border-color:#10B981; color:#34D399; min-width:180px;">
            🖼️ MRI + Mask Overlay
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Technology stack ────────────────────────────────────────
st.markdown("### ⚙️ Technology Stack")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    st.markdown("""
    <div class="nv-result-card" style="text-align:center;">
        <div style="font-size:1.6rem;">🔥</div>
        <div style="font-weight:600; color:#00D4FF; font-size:0.9rem;">PyTorch 2.14</div>
        <div style="font-size:0.75rem; color:#64748B;">Deep Learning Framework</div>
    </div>""", unsafe_allow_html=True)
with col_t2:
    st.markdown("""
    <div class="nv-result-card" style="text-align:center;">
        <div style="font-size:1.6rem;">🤖</div>
        <div style="font-weight:600; color:#7C3AED; font-size:0.9rem;">EfficientNet-B0</div>
        <div style="font-size:0.75rem; color:#64748B;">Transfer Learning</div>
    </div>""", unsafe_allow_html=True)
with col_t3:
    st.markdown("""
    <div class="nv-result-card" style="text-align:center;">
        <div style="font-size:1.6rem;">🎯</div>
        <div style="font-weight:600; color:#10B981; font-size:0.9rem;">U-Net</div>
        <div style="font-size:0.75rem; color:#64748B;">Segmentation</div>
    </div>""", unsafe_allow_html=True)
with col_t4:
    st.markdown("""
    <div class="nv-result-card" style="text-align:center;">
        <div style="font-size:1.6rem;">📊</div>
        <div style="font-weight:600; color:#F97316; font-size:0.9rem;">Streamlit</div>
        <div style="font-size:0.75rem; color:#64748B;">Interactive UI</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="nv-disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This system is an academic/research prototype and is
    <b>NOT intended for medical diagnosis</b>, treatment, or clinical decision-making.
    Always consult a qualified medical professional for health-related decisions.
</div>
""", unsafe_allow_html=True)
