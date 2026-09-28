"""
pages/3_📊_Model_Info.py — NeuroVision AI Model Information Page
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils import classifier_model_status, segmenter_model_status

st.set_page_config(
    page_title="Model Info — NeuroVision AI",
    page_icon="📊",
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
    clf_status = classifier_model_status()
    seg_status = segmenter_model_status()
    st.markdown("**Model Status**")
    if clf_status["exists"]:
        st.success(f"✅ Classifier ready ({clf_status['size_mb']} MB)")
    else:
        st.error("❌ Classifier: Not Trained")
    if seg_status["exists"]:
        st.success(f"✅ Segmenter ready ({seg_status['size_mb']} MB)")
    else:
        st.error("❌ Segmenter: Not Trained")

st.markdown("""
<h1 style="font-size:2rem;font-weight:700;
    background:linear-gradient(135deg,#00D4FF,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    📊 Model Architecture & Training
</h1>
<p style="color:#94A3B8; margin-top:-0.5rem;">
    Technical details, training instructions, and evaluation framework.
</p>
""", unsafe_allow_html=True)

tab_arch, tab_train, tab_metrics, tab_data = st.tabs([
    "🏗️ Architecture", "🏋️ Training Guide", "📈 Evaluation Metrics", "📁 Dataset Format"
])

# ── ARCHITECTURE TAB ────────────────────────────────────────
with tab_arch:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="nv-card">
            <div class="nv-section-header">🤖 EfficientNet-B0 Classifier</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.8;">
                <b style="color:#00D4FF;">EfficientNet-B0</b> is a compound-scaled CNN architecture
                that uniformly scales depth, width, and resolution for optimal efficiency.
                <br><br>
                <b>Transfer Learning:</b> The backbone is pretrained on ImageNet (~1.2M images,
                1000 classes). The final classifier head is replaced with a 4-class output layer
                to adapt for brain tumor classification.
                <br><br>
                <b>Custom Head:</b><br>
                &nbsp; → Dropout(0.3)<br>
                &nbsp; → Linear(1280 → 4)<br>
                &nbsp; → Softmax
                <br><br>
                <b>Input:</b> 224×224 RGB, ImageNet normalized<br>
                <b>Output:</b> 4-class softmax probabilities<br>
                <b>Parameters:</b> ~5.3M total<br>
                <b>Framework:</b> PyTorch / torchvision
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nv-card">
            <div class="nv-section-header">🔥 Grad-CAM Explainability</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.8;">
                <b style="color:#10B981;">Gradient-weighted Class Activation Mapping</b> (Grad-CAM)
                generates visual explanations for CNN decisions by using gradients flowing into
                the final convolutional layer.
                <br><br>
                <b>Target layer:</b> <code>features[-1]</code> (last MBConv block of EfficientNet-B0)<br>
                <b>Library:</b> pytorch-grad-cam<br>
                <b>Output:</b> Per-pixel attention heatmap, JET colormap overlay<br>
                <b>Purpose:</b> Identify which MRI regions influenced the classification
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="nv-card">
            <div class="nv-section-header">🏗️ U-Net Segmenter</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.8;">
                <b style="color:#7C3AED;">U-Net</b> is a fully convolutional encoder-decoder
                architecture designed for biomedical image segmentation.
                <br><br>
                <b>Encoder (4 stages):</b><br>
                &nbsp; Conv 1→64 → MaxPool<br>
                &nbsp; Conv 64→128 → MaxPool<br>
                &nbsp; Conv 128→256 → MaxPool<br>
                &nbsp; Conv 256→512 → MaxPool<br><br>
                <b>Bottleneck:</b> Conv 512→1024<br><br>
                <b>Decoder (4 stages):</b><br>
                &nbsp; Upsample + Skip + Conv 1024→512<br>
                &nbsp; Upsample + Skip + Conv 512→256<br>
                &nbsp; Upsample + Skip + Conv 256→128<br>
                &nbsp; Upsample + Skip + Conv 128→64<br><br>
                <b>Head:</b> Conv 64→1 + Sigmoid<br>
                <b>Input:</b> 1×256×256 (grayscale)<br>
                <b>Output:</b> 1×256×256 (binary mask)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nv-card">
            <div class="nv-section-header">🔧 Preprocessing Pipeline</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.8;">
                <b>Step 1:</b> Load image (PIL) → Convert to RGB<br>
                <b>Step 2:</b> Apply CLAHE in LAB colour space<br>
                &nbsp; clip_limit=2.0, tileGridSize=(8,8)<br>
                <b>Step 3:</b> Resize to 224×224 for classifier<br>
                <b>Step 4:</b> Normalize (ImageNet stats):<br>
                &nbsp; mean=[0.485, 0.456, 0.406]<br>
                &nbsp; std =[0.229, 0.224, 0.225]<br>
                <b>Step 5:</b> Convert to grayscale 256×256 for U-Net<br>
                <b>Step 6:</b> Normalize to [0, 1]
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── TRAINING TAB ────────────────────────────────────────────
with tab_train:
    st.markdown("""
    <div class="nv-card">
        <div class="nv-section-header">🏋️ Training the Classifier (EfficientNet-B0)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **1. Prepare your dataset** following the required folder structure (see Dataset tab).

    **2. Run the training script:**
    """)
    st.code("py -3 train_classifier.py", language="bash")

    st.markdown("**Training configuration (editable in `train_classifier.py`):**")
    st.code("""
# Training hyperparameters
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
IMAGE_SIZE    = (224, 224)

# Transfer learning strategy
# Phase 1: Freeze backbone, train head only (5 epochs)
# Phase 2: Unfreeze all layers, fine-tune (remaining epochs)
    """, language="python")

    st.markdown("**Weights saved to:**")
    st.code("models/classifier_efficientnet.pth", language="text")

    st.markdown("---")

    st.markdown("""
    <div class="nv-card">
        <div class="nv-section-header">🏋️ Training the Segmenter (U-Net)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Run the segmentation training script:**")
    st.code("py -3 train_segmenter.py", language="bash")

    st.code("""
# Segmentation training hyperparameters
EPOCHS        = 50
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE    = (256, 256)
LOSS          = BCEWithLogitsLoss + Dice Loss
    """, language="python")

    st.markdown("**Weights saved to:**")
    st.code("models/segmenter_unet.pth", language="text")

# ── METRICS TAB ────────────────────────────────────────────
with tab_metrics:
    st.markdown("""
    <div class="nv-not-trained" style="margin-bottom:1.5rem;">
        <div style="font-size:1rem; color:#FCD34D; font-weight:600; margin-bottom:0.5rem;">
            📋 Metrics will appear here after training
        </div>
        <div style="font-size:0.85rem; color:#94A3B8;">
            The placeholders below are for reference only. No values are hard-coded or invented.
            Train the models and evaluate to populate these metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### 🤖 Classifier Metrics")
        import pandas as pd
        clf_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)", "AUC-ROC"],
            "Value": ["— (not evaluated)", "— (not evaluated)", "— (not evaluated)", "— (not evaluated)", "— (not evaluated)"],
        })
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("#### 🎯 Segmenter Metrics")
        seg_df = pd.DataFrame({
            "Metric": ["Dice Coefficient", "IoU (Jaccard)", "Pixel Accuracy", "Sensitivity", "Specificity"],
            "Value": ["— (not evaluated)", "— (not evaluated)", "— (not evaluated)", "— (not evaluated)", "— (not evaluated)"],
        })
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

    st.info("""
    💡 **To evaluate the models after training:**

    ```python
    # Classifier evaluation
    py -3 train_classifier.py --evaluate

    # Segmenter evaluation
    py -3 train_segmenter.py --evaluate
    ```
    """)

# ── DATASET TAB ────────────────────────────────────────────
with tab_data:
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 📁 Classifier Dataset Structure")
        st.code("""
dataset/
├── Training/
│   ├── glioma/          # Glioma MRI scans
│   │   ├── img001.jpg
│   │   └── ...
│   ├── meningioma/      # Meningioma MRI scans
│   ├── pituitary/       # Pituitary tumor scans
│   └── notumor/         # Healthy brain scans
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
        """, language="text")

        st.markdown("""
        **Recommended dataset:**
        - [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
        - [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
        """)

    with col_d2:
        st.markdown("#### 📁 Segmenter Dataset Structure")
        st.code("""
segmentation_dataset/
├── images/          # Input MRI scans
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── masks/           # Ground truth tumor masks
    ├── img001.png   # Binary masks (0=background,
    ├── img002.png   #               255=tumor)
    └── ...
        """, language="text")

        st.markdown("""
        **Recommended dataset:**
        - [BraTS Challenge](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
        - [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

        **Note:** Mask filenames must match image filenames exactly.
        """)
