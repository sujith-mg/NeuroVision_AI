"""
train_segmenter.py
U-Net Brain Tumor Segmenter — Training Script

Usage:
    py -3 train_segmenter.py
    py -3 train_segmenter.py --epochs 50 --batch-size 8

Dataset structure required:
    segmentation_dataset/
    ├── images/        # MRI scans (PNG/JPG)
    └── masks/         # Binary tumor masks (0=background, 255=tumor)
                       # Filename must match the corresponding image file

Output:
    models/segmenter_unet.pth
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from PIL import Image
from tqdm import tqdm

from src.models.segmenter import UNet

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_DIR  = PROJECT_ROOT / "segmentation_dataset"
MODELS_DIR   = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "segmenter_unet.pth"

DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 1e-4
IMAGE_SIZE         = (256, 256)
VAL_SPLIT          = 0.2


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class SegmentationDataset(Dataset):
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, images_dir: Path, masks_dir: Path, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.augment    = augment

        # Find all image files
        self.image_paths = sorted([
            p for p in images_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        # Match masks
        self.mask_paths = []
        for img_path in self.image_paths:
            # Try same name with .png, .jpg
            for ext in [".png", ".jpg", ".jpeg"]:
                mask_path = masks_dir / (img_path.stem + ext)
                if mask_path.exists():
                    self.mask_paths.append(mask_path)
                    break
            else:
                raise FileNotFoundError(
                    f"No matching mask found for {img_path.name} in {masks_dir}"
                )

        print(f"  Found {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (grayscale)
        img_np = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            img_np = np.array(Image.open(self.image_paths[idx]).convert("L"))
        img_np = cv2.resize(img_np, IMAGE_SIZE)

        # Load mask
        mask_np = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask_np is None:
            mask_np = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask_np = cv2.resize(mask_np, IMAGE_SIZE)

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            img_np  = cv2.flip(img_np,  1)
            mask_np = cv2.flip(mask_np, 1)
        if self.augment and np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
            img_np  = cv2.warpAffine(img_np,  M, IMAGE_SIZE)
            mask_np = cv2.warpAffine(mask_np, M, IMAGE_SIZE)

        # Normalize
        img_tensor  = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)  # [1,256,256]
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)  # [1,256,256]

        return img_tensor, mask_tensor


# ─────────────────────────────────────────────
# Dice loss
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred)
        flat_p = pred.view(-1)
        flat_t = target.view(-1)
        intersection = (flat_p * flat_t).sum()
        return 1 - (2. * intersection + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + self.dice_w * self.dice(pred, target)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum()
    union = pred_bin.sum() + true_mask.sum()
    if union == 0:
        return 1.0
    return (2 * intersection / union).item()


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum()
    union = pred_bin.sum() + true_mask.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ─────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total}] Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # Model outputs sigmoid already; use BCELoss instead
        loss = nn.functional.binary_cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()

        # Dice using direct output (already sigmoid)
        pred_bin = (outputs > 0.5).float()
        intersection = (pred_bin * masks).sum()
        union = pred_bin.sum() + masks.sum()
        dice = (2 * intersection / (union + 1e-6)).item() if union > 0 else 1.0

        total_loss += loss.item()
        total_dice += dice
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = nn.functional.binary_cross_entropy(outputs, masks)
            total_loss += loss.item()

            pred_bin = (outputs > 0.5).float()
            intersection = (pred_bin * masks).sum()
            union_d = pred_bin.sum() + masks.sum()
            union_i = union_d - intersection
            dice = (2 * intersection / (union_d + 1e-6)).item()
            iou  = (intersection / (union_i + 1e-6)).item()
            total_dice += dice
            total_iou  += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def train(args):
    print("\n" + "="*60)
    print("  NeuroVision AI — U-Net Segmenter Training")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    images_dir = DATASET_DIR / "images"
    masks_dir  = DATASET_DIR / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"\n❌ Dataset not found at {DATASET_DIR}")
        print("   Create: segmentation_dataset/images/ and segmentation_dataset/masks/")
        sys.exit(1)

    print("\n📁 Loading segmentation dataset...")
    full_dataset = SegmentationDataset(images_dir, masks_dir, augment=True)

    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Disable augmentation for val
    val_ds.dataset.augment = False

    print(f"  Training samples:   {train_size}")
    print(f"  Validation samples: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_dice = 0.0
    print(f"\n🏋️ Training U-Net for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})")
    print("-"*60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = train_one_epoch(model, train_loader, None, optimizer, device, epoch, args.epochs)
        val_loss, val_dice, val_iou = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        saved_str = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), WEIGHTS_PATH)
            saved_str = " ✅ (saved)"

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}"
              f"{saved_str} | {elapsed:.1f}s")

    print("\n" + "="*60)
    print(f"  ✅ Training complete!")
    print(f"  Best validation Dice: {best_dice:.4f}")
    print(f"  Weights saved to: {WEIGHTS_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Brain Tumor Segmenter")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--dataset-dir",type=str,   default=str(DATASET_DIR))
    args = parser.parse_args()
    DATASET_DIR = Path(args.dataset_dir)
    train(args)
