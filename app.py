"""
app.py — NeuroVision AI Main Entry Point
Streamlit multi-page application for Brain Tumor Detection, Segmentation & Classification
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is always on sys.path ──────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page configuration (must be FIRST Streamlit call) ──────
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "NeuroVision AI — Academic Brain Tumor Detection System",
    },
)

# ── Inject custom CSS ───────────────────────────────────────
css_path = PROJECT_ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.8rem;">🧠</div>
        <div style="font-size:1.2rem; font-weight:700;
                    background:linear-gradient(135deg,#00D4FF,#7C3AED);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">
            NeuroVision AI
        </div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:0.2rem; letter-spacing:0.05em;">
            BRAIN TUMOR DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748B; padding:0 0.5rem;">
        <div style="margin-bottom:0.4rem; font-weight:600; color:#94A3B8;">NAVIGATION</div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("app.py", label="🏠 Home", icon=None)
    st.page_link("pages/1_🏠_Home.py", label="🏠 Overview", icon=None)
    st.page_link("pages/2_🔬_Analysis.py", label="🔬 MRI Analysis", icon=None)
    st.page_link("pages/3_📊_Model_Info.py", label="📊 Model Info", icon=None)
    st.page_link("pages/4_ℹ️_About.py", label="ℹ️ About", icon=None)

    st.markdown("---")
    st.markdown("""
    <div class="nv-disclaimer" style="font-size:0.72rem;">
        ⚠️ <b>Academic prototype only.</b><br>
        Not for medical diagnosis or clinical use.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; text-align:center; padding-top:0.5rem;">
        Velalar College of Engineering<br>
        AI & ML Mini Project • 2024–25
    </div>
    """, unsafe_allow_html=True)

# ── Main landing content ────────────────────────────────────
st.markdown("""
<div class="nv-hero">
    <div class="nv-title">🧠 NeuroVision AI</div>
    <div class="nv-subtitle">
        AI-Based Brain Tumor Detection, Segmentation &amp; Classification
    </div>
    <div style="font-size:0.85rem; color:#475569; max-width:600px; margin:0 auto;">
        An academic deep learning system using EfficientNet-B0 transfer learning
        and U-Net segmentation for automated MRI brain tumor analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick links ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">🔬</span>
        <div class="nv-feature-title">Start Analysis</div>
        <div style="font-size:0.82rem; color:#94A3B8;">
            Upload an MRI and run the full AI pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_🔬_Analysis.py", label="→ Open Analysis Page")

with col2:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">📊</span>
        <div class="nv-feature-title">Model Information</div>
        <div style="font-size:0.82rem; color:#94A3B8;">
            Architecture details and training guide
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_📊_Model_Info.py", label="→ Open Model Info")

with col3:
    st.markdown("""
    <div class="nv-feature-card">
        <span class="nv-feature-icon">ℹ️</span>
        <div class="nv-feature-title">About the Project</div>
        <div style="font-size:0.82rem; color:#94A3B8;">
            Team, college, and academic context
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/4_ℹ️_About.py", label="→ Open About")

st.markdown("---")
st.markdown("""
<div class="nv-disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This system is an academic/research prototype.
    It is <b>not intended for medical diagnosis</b>, treatment, or clinical decision-making.
    Always consult a qualified medical professional for any health concerns.
</div>
""", unsafe_allow_html=True)
