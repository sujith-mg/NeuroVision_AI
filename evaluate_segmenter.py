from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from src.models.segmenter import UNet

# Paths
IMAGE_DIR = Path("Segmentation/images")
MASK_DIR = Path("Segmentation/masks")
WEIGHTS = Path("models/segmenter_unet_10epoch_best.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

# Check required files
if not WEIGHTS.exists():
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Loading trained U-Net...")

model = UNet(in_channels=1, out_channels=1)
state = torch.load(WEIGHTS, map_location=device, weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

print("Checkpoint loaded successfully!")
print("Starting evaluation on", len(image_paths), "test images...")

results = []
total_intersection = 0
total_predicted = 0
total_actual = 0
total_loss = 0.0

visual_rows = []

for index, image_path in enumerate(image_paths, start=1):

    # Find matching mask
    mask_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = MASK_DIR / (image_path.stem + ext)
        if candidate.exists():
            mask_path = candidate
            break

    if mask_path is None:
        raise FileNotFoundError(f"Mask missing for {image_path.name}")

    # Read grayscale MRI and mask
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise RuntimeError(f"Could not read {image_path.name} or its mask")

    # Same resizing and normalization as training
    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)

    image_tensor = torch.from_numpy(
        image.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)

    true_mask = (mask > 127).astype(np.uint8)

    mask_tensor = torch.from_numpy(
        true_mask.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.inference_mode():
        prediction = model(image_tensor)
        loss = torch.nn.functional.binary_cross_entropy(
            prediction, mask_tensor
        ).item()

    predicted_mask = (
        prediction.squeeze().cpu().numpy() > THRESHOLD
    ).astype(np.uint8)

    # Calculate Dice and IoU
    intersection = int(np.logical_and(
        predicted_mask, true_mask
    ).sum())

    predicted_pixels = int(predicted_mask.sum())
    actual_pixels = int(true_mask.sum())

    dice_denominator = predicted_pixels + actual_pixels
    iou_denominator = predicted_pixels + actual_pixels - intersection

    dice = (
        2 * intersection / dice_denominator
        if dice_denominator > 0 else 1.0
    )

    iou = (
        intersection / iou_denominator
        if iou_denominator > 0 else 1.0
    )

    total_intersection += intersection
    total_predicted += predicted_pixels
    total_actual += actual_pixels
    total_loss += loss

    results.append({
        "image": image_path.name,
        "dice": dice,
        "iou": iou,
        "loss": loss,
        "actual_tumor_pixels": actual_pixels,
        "predicted_tumor_pixels": predicted_pixels
    })

    # Save first 8 visual comparisons
    if index <= 8:
        original = Image.fromarray(image).convert("RGB")
        ground_truth = Image.fromarray(true_mask * 255).convert("RGB")
        predicted = Image.fromarray(predicted_mask * 255).convert("RGB")

        base = image.astype(np.float32) * 0.5
        overlay = np.stack([base, base, base], axis=-1)
        overlay[:, :, 1] += true_mask * 127.5
        overlay[:, :, 0] += predicted_mask * 127.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        visual_rows.append([
            original,
            ground_truth,
            predicted,
            Image.fromarray(overlay)
        ])

    if index % 50 == 0 or index == len(image_paths):
        print(f"Processed {index}/{len(image_paths)} images")

# Calculate final metrics
count = len(results)

mean_dice = np.mean([r["dice"] for r in results])
mean_iou = np.mean([r["iou"] for r in results])
mean_loss = total_loss / count

global_dice = (
    2 * total_intersection / (total_predicted + total_actual)
    if total_predicted + total_actual > 0 else 1.0
)

global_iou = (
    total_intersection /
    (total_predicted + total_actual - total_intersection)
    if total_predicted + total_actual - total_intersection > 0
    else 1.0
)

# Save per-image metrics
csv_path = OUTPUT_DIR / "unet_test_metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Save visual comparison
if visual_rows:
    labels = ["MRI Image", "Ground Truth", "Prediction", "Overlay"]
    cell_width = 256
    cell_height = 286

    canvas = Image.new(
        "RGB", (cell_width * 4, cell_height * len(visual_rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(visual_rows):
        for col_index, picture in enumerate(row):
            x = col_index * cell_width
            y = row_index * cell_height
            draw.text((x + 5, y + 5), labels[col_index], fill="black")
            canvas.paste(picture, (x, y + 25))

    visual_path = OUTPUT_DIR / "unet_test_comparison.png"
    canvas.save(visual_path)

# Print results
print("\n" + "=" * 50)
print("       U-NET OFFICIAL TEST RESULTS")
print("=" * 50)
print("Test images:", count)
print(f"Mean Dice:   {mean_dice:.4f} ({mean_dice * 100:.2f}%)")
print(f"Mean IoU:    {mean_iou:.4f} ({mean_iou * 100:.2f}%)")
print(f"Mean Loss:   {mean_loss:.4f}")
print(f"Global Dice: {global_dice:.4f}")
print(f"Global IoU:  {global_iou:.4f}")
print("=" * 50)

print("\nSaved files:")
print("Metrics:", csv_path)
print("Visual comparison:", OUTPUT_DIR / "unet_test_comparison.png")
print("\nEvaluation completed!")
