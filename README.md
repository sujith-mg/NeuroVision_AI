# NeuroVision AI — Brain Tumor Detection, Segmentation & Classification

> **Academic Mini-Project** — Velalar College of Engineering and Technology  
> ⚠️ NOT for medical diagnosis or clinical use.

---

## Overview

NeuroVision AI is a complete deep learning pipeline for brain MRI analysis, combining:
- **EfficientNet-B0** (transfer learning) for 4-class tumor classification
- **U-Net** (custom encoder-decoder) for pixel-level tumor segmentation
- **Grad-CAM** for visual explainability
- **Streamlit** interactive web interface

### Four Classes
| Class | Description |
|-------|-------------|
| 🔴 Glioma | Malignant brain tumor arising from glial cells |
| 🟠 Meningioma | Tumor arising from the meninges |
| 🟣 Pituitary Tumor | Tumor of the pituitary gland |
| 🟢 Healthy Brain | No tumor present |

---

## Quick Start

### 1. Clone / Setup

```bash
cd NeuroVision_AI
```

### 2. Install Dependencies

```bash
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
py -3 -m pip install grad-cam opencv-python-headless pillow numpy scipy scikit-learn pandas plotly streamlit tqdm
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Training the Models

### Classifier (EfficientNet-B0)

1. Prepare dataset:
```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

2. Train:
```bash
py -3 train_classifier.py
py -3 train_classifier.py --epochs 50 --batch-size 16
```

3. Weights saved to: `models/classifier_efficientnet.pth`

---

### Segmenter (U-Net)

1. Prepare dataset:
```
segmentation_dataset/
├── images/      # MRI scans
└── masks/       # Binary tumor masks (same filenames)
```

2. Train:
```bash
py -3 train_segmenter.py
py -3 train_segmenter.py --epochs 100 --batch-size 8
```

3. Weights saved to: `models/segmenter_unet.pth`

---

## Project Structure

```
NeuroVision_AI/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Dependencies
├── train_classifier.py         # EfficientNet-B0 training
├── train_segmenter.py          # U-Net training
│
├── pages/
│   ├── 1_🏠_Home.py            # Overview page
│   ├── 2_🔬_Analysis.py        # Main analysis page
│   ├── 3_📊_Model_Info.py      # Architecture & training guide
│   └── 4_ℹ️_About.py           # Team & references
│
├── src/
│   ├── preprocessing.py        # CLAHE + normalisation pipeline
│   ├── visualization.py        # Plotly charts, overlays
│   ├── utils.py                # Shared helpers
│   └── models/
│       ├── classifier.py       # BrainTumorClassifier (EfficientNet-B0)
│       ├── segmenter.py        # BrainTumorSegmenter (U-Net)
│       └── gradcam.py          # GradCAMGenerator
│
├── assets/
│   └── styles.css              # Dark medical-AI theme
│
└── models/                     # Trained .pth weights (add here)
    └── .gitkeep
```

---

## Unit Tests

```bash
py -3 -m pytest src/tests/ -v
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.14 |
| Classification | EfficientNet-B0 (torchvision) |
| Segmentation | Custom U-Net (PyTorch) |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Image Processing | OpenCV, Pillow, CLAHE |
| UI Framework | Streamlit |
| Visualization | Plotly |
| Language | Python 3.14 |
| Platform | Windows CPU |

---

## Team

| Name | Role |
|------|------|
| Santhosh S | Team Lead / Classification Model |
| Sujith M | Segmentation Model (U-Net) |
| Thilak VR | Grad-CAM / Explainability |
| Meiyarasan S | UI Development / Streamlit |

**Institution:** Velalar College of Engineering and Technology, Erode, Tamil Nadu  
**Programme:** B.E. Computer Science & Engineering  
**Academic Year:** 2024–25

---

## Medical Disclaimer

> ⚠️ **This system is an academic/research prototype and is NOT intended for medical diagnosis, treatment, or clinical decision-making.** Always consult a qualified medical professional for any health concerns.

---

## References

1. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019.
2. Ronneberger et al. (2015). *U-Net: CNNs for Biomedical Image Segmentation*. MICCAI 2015.
3. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
4. Zuiderveld (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.
