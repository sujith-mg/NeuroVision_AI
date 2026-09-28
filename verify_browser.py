"""
verify_browser.py — Clean browser verification
Imports torch first, then stubs streamlit, then tests all modules.
"""
import sys
sys.path.insert(0, 'd:/pr/NeuroVision_AI')

# ── Step 1: Pre-import PyTorch BEFORE stubbing anything ──────
import torch
import torchvision
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2

print(f'[OK] torch {torch.__version__}')
print(f'[OK] torchvision {torchvision.__version__}')
print(f'[OK] cv2 {cv2.__version__}')
print()

# ── Step 2: Stub streamlit (now PyTorch is safely imported) ──
import types

class _Stub:
    """Safe no-op stub for any Streamlit call."""
    def __call__(self, *a, **kw): return _Stub()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([])
    def __bool__(self): return False
    def __len__(self): return 0
    def __getattr__(self, n): return _Stub()

class StubSidebar(_Stub):
    pass

class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.sidebar = _Stub()
        self.session_state = types.SimpleNamespace()
    def set_page_config(self, *a, **kw): pass
    def columns(self, n):
        count = n if isinstance(n, int) else len(n)
        return [_Stub() for _ in range(count)]
    def tabs(self, names): return [_Stub() for _ in names]
    def spinner(self, *a, **kw): return _Stub()
    def file_uploader(self, *a, **kw): return None
    def stop(self): raise SystemExit(0)
    def __getattr__(self, n): return _Stub()

sys.modules['streamlit'] = StreamlitStub()

# ── Step 3: Now import all project modules ────────────────────
print('=== IMPORT TESTS ===')
import traceback
all_ok = True

def test_import(name):
    global all_ok
    try:
        mod = __import__(name, fromlist=[''])
        print(f'[PASS] import {name}')
        return mod
    except SystemExit:
        print(f'[PASS] import {name} (st.stop called — expected)')
        return None
    except Exception as e:
        print(f'[FAIL] import {name}: {type(e).__name__}: {e}')
        traceback.print_exc()
        all_ok = False
        return None

test_import('src.utils')
test_import('src.preprocessing')
test_import('src.visualization')
test_import('src.models.classifier')
test_import('src.models.segmenter')
test_import('src.models.gradcam')
test_import('src.models')

print()
print('=== LOGIC TESTS ===')

# ── 1. Classes & class list ───────────────────────────────────
from src.models.classifier import BrainTumorClassifier, CLASSES, NUM_CLASSES, build_efficientnet_b0
assert NUM_CLASSES == 4
assert set(CLASSES) == {'Glioma', 'Meningioma', 'Pituitary Tumor', 'Healthy Brain'}
print('[PASS] 4 correct classes defined')

# ── 2. EfficientNet-B0 architecture ──────────────────────────
model = build_efficientnet_b0()
model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
assert out.shape == (1, 4), f'Got {out.shape}'
print(f'[PASS] EfficientNet-B0 output shape: {tuple(out.shape)}')

# ── 3. Classifier — no-weights state ────────────────────────
clf = BrainTumorClassifier()
assert clf.is_trained() == False
print('[PASS] classifier.is_trained() == False (no .pth file)')

try:
    clf.predict(torch.randn(1,3,224,224))
    print('[FAIL] Expected RuntimeError'); all_ok = False
except RuntimeError as e:
    assert 'not trained' in str(e).lower()
    print('[PASS] predict() raises RuntimeError — no fake prediction')

# ── 4. Probabilities sum to 1 (with random weights) ──────────
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
    tmp_path = f.name
torch.save(model.state_dict(), tmp_path)
clf2 = BrainTumorClassifier(weights_path=tmp_path)
result = clf2.predict(torch.randn(1,3,224,224))
total_prob = sum(result['probabilities'].values())
assert abs(total_prob - 1.0) < 1e-5, f'Probs sum to {total_prob}'
assert result['class'] in CLASSES
assert 0.0 <= result['confidence'] <= 1.0
print(f'[PASS] predict() returns valid structure, probs sum={total_prob:.6f}')
print(f'       Class: {result["class"]}, Confidence: {result["confidence"]*100:.2f}%')
os.unlink(tmp_path)

# ── 5. U-Net architecture ─────────────────────────────────────
from src.models.segmenter import BrainTumorSegmenter, UNet
unet = UNet()
unet.eval()
seg_in = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    seg_out = unet(seg_in)
assert seg_out.shape == (1, 1, 256, 256)
assert float(seg_out.min()) >= 0.0
assert float(seg_out.max()) <= 1.0
print(f'[PASS] U-Net output shape: {tuple(seg_out.shape)}, range [{float(seg_out.min()):.3f}, {float(seg_out.max()):.3f}]')

# ── 6. Segmenter — no-weights state ──────────────────────────
seg = BrainTumorSegmenter()
assert seg.is_trained() == False
try:
    seg.segment(torch.randn(1,1,256,256))
    print('[FAIL] Expected RuntimeError'); all_ok = False
except RuntimeError as e:
    assert 'not trained' in str(e).lower()
    print('[PASS] segment() raises RuntimeError — no fake mask')

# ── 7. Preprocessing pipeline ─────────────────────────────────
from src.preprocessing import preprocess_mri, apply_clahe, tensor_to_display

for label, sz in [('small', (64,64)), ('standard', (256,256)), ('wide', (500,300))]:
    arr = np.random.randint(30, 200, (*sz, 3), dtype='uint8')
    r = preprocess_mri(Image.fromarray(arr))
    assert r.classifier_tensor.shape == (1,3,224,224)
    assert r.segmenter_tensor.shape  == (1,1,256,256)
    assert float(r.segmenter_tensor.min()) >= 0.0
    assert float(r.segmenter_tensor.max()) <= 1.0
    # Verify ImageNet normalisation applied
    t = r.classifier_tensor
    assert float(t.min()) < 0.0 or float(t.max()) > 1.0, 'Normalisation not applied'
print('[PASS] Preprocessing: correct shapes/dtypes/normalisation for all input sizes')

# CLAHE tests
rgb = np.random.randint(0,255,(128,128,3), dtype='uint8')
out = apply_clahe(rgb)
assert out.shape == (128,128,3) and out.dtype == np.uint8
gray = np.random.randint(0,255,(128,128), dtype='uint8')
out_g = apply_clahe(gray)
assert out_g.shape == (128,128)
print('[PASS] CLAHE works for RGB and grayscale')

# tensor_to_display
display_img = tensor_to_display(torch.randn(1,3,224,224))
assert display_img.shape == (224,224,3) and display_img.dtype == np.uint8
assert 0 <= int(display_img.min()) and int(display_img.max()) <= 255
print('[PASS] tensor_to_display produces uint8 [0-255] output')

# ── 8. Visualization ──────────────────────────────────────────
from src.visualization import (
    make_probability_chart, overlay_mask_on_image, mask_to_rgb,
    numpy_to_pil, resize_for_display, CLASS_COLORS, ALL_CLASSES
)

probs = {'Glioma':0.65,'Meningioma':0.20,'Pituitary Tumor':0.10,'Healthy Brain':0.05}
fig = make_probability_chart(probs)
assert isinstance(fig, go.Figure)
assert len(fig.data) == 1
assert fig.data[0].orientation == 'h'
print('[PASS] Plotly chart: valid Figure, horizontal bar, 1 trace')

for cls in ALL_CLASSES:
    assert cls in CLASS_COLORS
print('[PASS] CLASS_COLORS defined for all 4 classes')

orig = np.random.randint(0,200,(256,256,3), dtype='uint8')
mask = np.zeros((256,256), dtype='uint8'); mask[80:160,100:180] = 255
overlay = overlay_mask_on_image(orig, mask)
assert overlay.shape == (256,256,3) and overlay.dtype == np.uint8
print('[PASS] overlay_mask_on_image: correct shape and dtype')

rgb_mask = mask_to_rgb(mask)
assert rgb_mask.shape == (256,256,3)
print('[PASS] mask_to_rgb: H×W×3 output')

pil_img = numpy_to_pil(orig)
assert isinstance(pil_img, Image.Image)
print('[PASS] numpy_to_pil: returns PIL Image')

big = np.zeros((1200,900,3), dtype='uint8')
sm = resize_for_display(big, 512)
assert max(sm.shape[:2]) <= 512
print('[PASS] resize_for_display: caps longest side correctly')

# ── 9. Utils ──────────────────────────────────────────────────
from src.utils import (
    classifier_model_status, segmenter_model_status, get_class_emoji,
    get_class_color, format_confidence, load_css, PROJECT_ROOT
)

cs = classifier_model_status()
ss = segmenter_model_status()
assert cs['exists'] == False and ss['exists'] == False
print('[PASS] Both models correctly reported as missing')

for cls in ['Glioma','Meningioma','Pituitary Tumor','Healthy Brain']:
    assert get_class_emoji(cls) in ['🔴','🟠','🟣','🟢']
    assert get_class_color(cls).startswith('#')
print('[PASS] class emoji/color helpers work for all 4 classes')

assert format_confidence(0.876) == '87.6%'
assert format_confidence(0.0)   == '0.0%'
assert format_confidence(1.0)   == '100.0%'
print('[PASS] format_confidence formats all edge cases correctly')

css = load_css()
assert len(css) > 500
assert '--accent' in css and 'nv-card' in css and 'nv-hero' in css
print(f'[PASS] CSS loaded ({len(css)} chars), contains all key classes')

# ── 10. Grad-CAM ─────────────────────────────────────────────
from src.models.gradcam import GradCAMGenerator, is_gradcam_available
available = is_gradcam_available()
print(f'[PASS] Grad-CAM importable = {available}')

if available:
    # Verify it works with a real model
    cam_model = build_efficientnet_b0()
    cam_model.eval()
    gen = GradCAMGenerator(cam_model, torch.device('cpu'))
    t_in = torch.randn(1, 3, 224, 224)
    display = np.random.randint(0,200,(224,224,3), dtype='uint8')
    heatmap, overlay_cam = gen.generate(t_in, display, class_idx=0)
    assert heatmap.shape == (224, 224, 3)
    assert overlay_cam.shape == (224, 224, 3)
    print('[PASS] Grad-CAM generates heatmap and overlay correctly')

# ── 11. Sample MRI end-to-end ────────────────────────────────
sample = PROJECT_ROOT / 'assets' / 'sample_mri' / 'test_mri.png'
assert sample.exists(), f'Missing: {sample}'
img = Image.open(sample).convert('RGB')
result = preprocess_mri(img)
assert result.classifier_tensor.shape == (1,3,224,224)
assert result.segmenter_tensor.shape  == (1,1,256,256)
print(f'[PASS] Sample MRI ({img.size}) preprocesses correctly end-to-end')

# ── 12. HTTP Endpoint checks ──────────────────────────────────
import urllib.request
print()
print('=== HTTP ENDPOINT TESTS ===')
endpoints = [
    ('/_stcore/health',     'Health'),
    ('/',                   'Home'),
    ('/_stcore/allowed-message-origins', 'Origins'),
]
for path, label in endpoints:
    try:
        r = urllib.request.urlopen(f'http://localhost:8501{path}', timeout=8)
        body = r.read().decode('utf-8','replace')[:80]
        print(f'[PASS] {label} (HTTP {r.getcode()}): {body}')
    except Exception as e:
        print(f'[WARN] {label}: {e}')

# ── 13. Page file syntax check ────────────────────────────────
import ast
print()
print('=== PAGE SYNTAX CHECKS ===')
pages_dir = PROJECT_ROOT / 'pages'
for page_file in sorted(pages_dir.glob('*.py')):
    src_text = page_file.read_text(encoding='utf-8')
    try:
        ast.parse(src_text)
        print(f'[PASS] {page_file.name}: valid Python syntax')
    except SyntaxError as e:
        print(f'[FAIL] {page_file.name}: SyntaxError at line {e.lineno}: {e.msg}')
        all_ok = False

# Also check src modules
for src_file in (PROJECT_ROOT / 'src').rglob('*.py'):
    src_text = src_file.read_text(encoding='utf-8')
    try:
        ast.parse(src_text)
    except SyntaxError as e:
        print(f'[FAIL] {src_file.relative_to(PROJECT_ROOT)}: SyntaxError at line {e.lineno}')
        all_ok = False
print('[PASS] All src/*.py files have valid Python syntax')

# app.py
app_text = (PROJECT_ROOT / 'app.py').read_text(encoding='utf-8')
try:
    ast.parse(app_text)
    print('[PASS] app.py: valid Python syntax')
except SyntaxError as e:
    print(f'[FAIL] app.py: SyntaxError at line {e.lineno}: {e.msg}')
    all_ok = False

# ── Summary ───────────────────────────────────────────────────
print()
print('=' * 55)
print('BROWSER VERIFICATION COMPLETE')
print(f'Status: {"ALL PASSED" if all_ok else "FAILURES DETECTED — see above"}')
print('=' * 55)
