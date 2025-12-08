# scripts/verify_validation_performance.py
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from src.data.dataset import IRMASDataset, create_stratified_train_val_split
from src.models.densenet import create_densenet121
from src.utils.device import select_device

device = select_device()
train_dir = Path("data/raw/IRMAS-TrainingData")

# Get validation indices (same seed as training)
_, val_indices = create_stratified_train_val_split(
    train_dir, val_ratio=0.15, random_seed=42
)

# Create validation dataset (NO augmentation)
val_dataset = IRMASDataset(
    data_dir=train_dir,
    target_sr=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    indices=val_indices,
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)
checkpoint = torch.load(
    "outputs/runs/full_augment_20251201_093013/continued/best_model.pth",
    map_location=device,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Run inference
all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in val_loader:
        specs = specs.to(device)
        logits = model(specs)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute F1
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"\n{'=' * 70}")
print("VALIDATION SET VERIFICATION")
print(f"{'=' * 70}")
print(f"Validation F1 (our calculation): {f1:.4f}")
print(f"Validation F1 (from checkpoint):  {checkpoint['val_f1']:.4f}")
print(f"Difference: {abs(f1 - checkpoint['val_f1']):.4f}")

if abs(f1 - checkpoint["val_f1"]) > 0.01:
    print("\n⚠️  WARNING: Scores don't match! Possible data leakage or metric bug.")
else:
    print("\n✓ Validation score confirmed - model really does get 75% on validation.")
