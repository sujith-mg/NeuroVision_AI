"""
train_classifier.py
EfficientNet-B0 Brain Tumor Classifier — Training Script

Usage:
    py -3 train_classifier.py
    py -3 train_classifier.py --epochs 50 --batch-size 16
    py -3 train_classifier.py --evaluate   (evaluate saved model)

Dataset structure required:
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

Output:
    models/classifier_efficientnet.pth
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.classifier import build_efficientnet_b0, CLASSES

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_DIR   = PROJECT_ROOT / "dataset"
MODELS_DIR    = PROJECT_ROOT / "models"
WEIGHTS_PATH  = MODELS_DIR / "classifier_efficientnet.pth"

DEFAULT_EPOCHS     = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
IMAGE_SIZE         = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Data transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────
def load_datasets(dataset_dir: Path):
    train_dir = dataset_dir / "Training"
    test_dir  = dataset_dir / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Please prepare dataset at: {dataset_dir}/Training/{{glioma,meningioma,pituitary,notumor}}/"
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    print(f"  Classes found: {train_dataset.classes}")
    print(f"  Training samples: {len(train_dataset)}")

    if test_dir.exists():
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        print(f"  Testing samples:  {len(test_dataset)}")
    else:
        print(f"  ⚠️  Testing directory not found. Skipping evaluation.")
        test_dataset = None

    return train_dataset, test_dataset


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}] Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total*100:.1f}%")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(args):
    print("\n" + "="*60)
    print("  NeuroVision AI — EfficientNet-B0 Classifier Training")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n📁 Loading datasets...")
    try:
        train_dataset, test_dataset = load_datasets(DATASET_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False) if test_dataset else None

    # Build model
    print("\n🤖 Building EfficientNet-B0...")
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = build_efficientnet_b0(num_classes=4)

    # Load ImageNet pretrained backbone (download if needed)
    pretrained = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Copy backbone weights, skip classifier
    pretrained_state = {k: v for k, v in pretrained.state_dict().items() if "classifier" not in k}
    model.load_state_dict(pretrained_state, strict=False)
    print("  ✅ ImageNet backbone weights loaded")

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc = 0.0
    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🏋️ Training for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})")
    print("-"*60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_str = ""
        if test_loader:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            val_str = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), WEIGHTS_PATH)
                val_str += " ✅ (saved)"
        else:
            torch.save(model.state_dict(), WEIGHTS_PATH)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
              f"{val_str} | {elapsed:.1f}s")

    print("\n" + "="*60)
    print(f"  ✅ Training complete!")
    print(f"  Best validation accuracy: {best_acc*100:.2f}%")
    print(f"  Weights saved to: {WEIGHTS_PATH}")
    print("="*60 + "\n")


def evaluate_only(args):
    """Load saved weights and evaluate on test set."""
    if not WEIGHTS_PATH.exists():
        print(f"❌ No weights found at {WEIGHTS_PATH}. Train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset = load_datasets(DATASET_DIR)
    if test_dataset is None:
        print("❌ No test dataset found.")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    model = build_efficientnet_b0()
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, test_loader, criterion, device)
    print(f"\n📊 Test Results:")
    print(f"   Loss:     {loss:.4f}")
    print(f"   Accuracy: {acc*100:.2f}%")

    # Per-class accuracy
    model.eval()
    class_correct = [0] * 4
    class_total   = [0] * 4
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                lbl = labels[i].item()
                class_correct[lbl] += (preds[i] == labels[i]).item()
                class_total[lbl]   += 1

    print("\n   Per-class accuracy:")
    class_names = test_dataset.classes
    for i, cls in enumerate(class_names):
        if class_total[i] > 0:
            print(f"   {cls:20s}: {class_correct[i]/class_total[i]*100:.2f}%")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Brain Tumor Classifier")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay",  type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--evaluate",      action="store_true", help="Evaluate saved model only")
    parser.add_argument("--dataset-dir",   type=str,   default=str(DATASET_DIR))
    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir)

    if args.evaluate:
        evaluate_only(args)
    else:
        train(args)
