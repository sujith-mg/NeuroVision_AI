"""
pages/4_ℹ️_About.py — NeuroVision AI About Page
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="About — NeuroVision AI",
    page_icon="ℹ️",
    layout="wide",
)

css_path = PROJECT_ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="nv-disclaimer" style="font-size:0.72rem;">
        ⚠️ <b>Academic prototype only.</b><br>
        Not for medical diagnosis or clinical use.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<h1 style="font-size:2rem;font-weight:700;
    background:linear-gradient(135deg,#00D4FF,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    ℹ️ About NeuroVision AI
</h1>
<p style="color:#94A3B8; margin-top:-0.5rem;">
    Academic mini-project details, team, and references.
</p>
""", unsafe_allow_html=True)

# ── Project info ─────────────────────────────────────────────
st.markdown("""
<div class="nv-card">
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;">
        <div style="font-size:3rem;">🧠</div>
        <div>
            <div style="font-size:1.3rem; font-weight:700; color:#00D4FF;">NeuroVision AI</div>
            <div style="color:#94A3B8; font-size:0.88rem;">
                AI-Based Brain Tumor Detection, Segmentation &amp; Classification
            </div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-size:0.88rem;">
        <div>
            <span style="color:#64748B;">🏫 Institution</span><br>
            <b>Velalar College of Engineering and Technology</b><br>
            <span style="color:#64748B; font-size:0.8rem;">Erode, Tamil Nadu</span>
        </div>
        <div>
            <span style="color:#64748B;">📚 Programme</span><br>
            <b>B.E. Computer Science & Engineering</b><br>
            <span style="color:#64748B; font-size:0.8rem;">AI & ML Mini Project — 2024–25</span>
        </div>
        <div>
            <span style="color:#64748B;">📋 Project Type</span><br>
            <b>Academic Mini-Project</b><br>
            <span style="color:#64748B; font-size:0.8rem;">Not for clinical/commercial use</span>
        </div>
        <div>
            <span style="color:#64748B;">🔬 Domain</span><br>
            <b>Medical AI / Computer Vision</b><br>
            <span style="color:#64748B; font-size:0.8rem;">Deep Learning for Healthcare</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Team members ─────────────────────────────────────────────
st.markdown("### 👥 Project Team")

team_members = [
    {"name": "Santhosh S",    "role": "Team Lead / Classification Model",   "emoji": "🧑‍💻"},
    {"name": "Sujith M",      "role": "Segmentation Model / U-Net",          "emoji": "🧑‍🔬"},
    {"name": "Thilak VR",     "role": "Grad-CAM / Explainability",           "emoji": "🧑‍💡"},
    {"name": "Meiyarasan S",  "role": "UI Development / Streamlit",           "emoji": "🧑‍🎨"},
]

cols = st.columns(4)
for i, member in enumerate(team_members):
    with cols[i]:
        st.markdown(f"""
        <div class="nv-feature-card">
            <div class="nv-feature-icon">{member['emoji']}</div>
            <div class="nv-feature-title">{member['name']}</div>
            <div style="font-size:0.78rem; color:#64748B; line-height:1.5;">{member['role']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Tech stack ────────────────────────────────────────────────
st.markdown("### ⚙️ Technology Stack")

tech_cols = st.columns(3)
tech_items = [
    ("🔥 PyTorch 2.14", "Deep Learning Framework", "#EF4444"),
    ("📷 torchvision", "EfficientNet-B0 Transfer Learning", "#F97316"),
    ("🎯 U-Net (Custom)", "Tumor Segmentation", "#A855F7"),
    ("🔥 Grad-CAM", "Visual Explainability (pytorch-grad-cam)", "#10B981"),
    ("🌊 Streamlit", "Interactive Web Interface", "#00D4FF"),
    ("📊 Plotly", "Confidence Visualizations", "#6366F1"),
    ("🖼️ OpenCV", "Image Processing / CLAHE", "#F59E0B"),
    ("🔢 NumPy / SciPy", "Numerical Operations", "#06B6D4"),
    ("🐍 Python 3.14", "Runtime Environment (Windows)", "#3B82F6"),
]

for i, (name, desc, color) in enumerate(tech_items):
    with tech_cols[i % 3]:
        st.markdown(f"""
        <div class="nv-result-card" style="margin-bottom:0.6rem;">
            <div style="font-weight:600; color:{color}; font-size:0.88rem;">{name}</div>
            <div style="font-size:0.75rem; color:#64748B;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── References ────────────────────────────────────────────────
st.markdown("### 📖 References")

with st.expander("📚 Key Papers & Resources"):
    st.markdown("""
    1. **EfficientNet**: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

    2. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

    3. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

    4. **CLAHE**: Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, Academic Press.

    5. **Brain Tumor MRI Dataset**: Masoud Nickparvar. (2021). *Brain Tumor MRI Dataset*. Kaggle. [Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

    6. **Transfer Learning**: Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning*. IEEE TKDE, 22(10), 1345–1359.
    """)

# ── Disclaimer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="nv-disclaimer">
    <b>⚠️ Medical & Academic Disclaimer</b><br><br>
    This system — NeuroVision AI — is an <b>academic research prototype</b> developed as a
    mini-project submission for educational purposes at Velalar College of Engineering and Technology.
    <br><br>
    It is <b>NOT intended for</b> and must <b>NOT be used for</b>:
    <ul style="margin:0.5rem 0; padding-left:1.5rem; color:#FCD34D;">
        <li>Medical diagnosis of any condition</li>
        <li>Clinical decision-making</li>
        <li>Treatment planning</li>
        <li>Patient screening</li>
        <li>Any healthcare application</li>
    </ul>
    Always consult a qualified radiologist or medical professional for brain MRI interpretation.
    The authors accept no liability for any use of this software outside its intended academic scope.
</div>
""", unsafe_allow_html=True)
