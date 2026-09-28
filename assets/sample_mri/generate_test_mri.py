"""
Generate a synthetic test MRI image for browser testing.
Saves to assets/sample_mri/test_mri.png
"""
import sys
from pathlib import Path
import numpy as np

# Try to use PIL
try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def generate_synthetic_mri(width=256, height=256):
    """Generate a brain-like synthetic MRI image."""
    img_arr = np.zeros((height, width), dtype=np.uint8)

    cx, cy = width // 2, height // 2
    brain_rx, brain_ry = width // 2 - 20, height // 2 - 20

    # Draw brain oval
    for y in range(height):
        for x in range(width):
            nx = (x - cx) / brain_rx
            ny = (y - cy) / brain_ry
            if nx**2 + ny**2 <= 1.0:
                # Base brain intensity with noise
                dist = (nx**2 + ny**2) ** 0.5
                intensity = int(160 - 60 * dist + np.random.randint(-10, 10))
                img_arr[y, x] = max(0, min(255, intensity))

    # Draw a simulated tumor region
    t_cx, t_cy = cx + 30, cy - 20
    t_r = 25
    for y in range(height):
        for x in range(width):
            d = ((x - t_cx)**2 + (y - t_cy)**2) ** 0.5
            if d <= t_r:
                intensity = int(220 - 50 * (d / t_r))
                if img_arr[y, x] > 0:  # only inside brain
                    img_arr[y, x] = max(0, min(255, intensity))

    return img_arr

output_dir = Path(__file__).parent.parent / "assets" / "sample_mri"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "test_mri.png"

arr = generate_synthetic_mri(256, 256)

if HAS_PIL:
    img = Image.fromarray(arr, mode='L').convert('RGB')
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(str(output_path))
    print(f"[OK] Saved synthetic MRI to {output_path}")
else:
    # Fallback: write raw bytes
    print("PIL not available, trying numpy save...")
    try:
        import struct, zlib
        # Create a minimal PNG manually (too complex, skip)
        print("❌ Cannot save without PIL")
    except:
        pass
