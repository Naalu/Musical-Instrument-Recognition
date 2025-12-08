#!/usr/bin/env python3
"""Quick ablation training for individual augmentations.

Trains a model for 10 epochs with a single augmentation type.
Use this ONLY if you need ablation checkpoints and don't have them.

Runtime: ~20-30 minutes per experiment on M1 Pro

Usage:
    python scripts/train_quick_ablation.py --augment pitch
    python scripts/train_quick_ablation.py --augment stretch
    python scripts/train_quick_ablation.py --augment specaug
    python scripts/train_quick_ablation.py --augment none
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.paths import get_data_dir
from src.data.augmented_dataset import AugmentedIRMASDataset
from src.data.dataset import IRMASDataset, get_class_weights
from src.data.splitter import create_stratified_train_val_split
from src.models.densenet import count_parameters, create_densenet121
from src.utils.device import select_device
from src.utils.seed import set_seed


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for specs, labels in pbar:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1


def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Val", leave=False):
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Quick ablation training")
    parser.add_argument(
        "--augment",
        type=str,
        required=True,
        choices=["none", "pitch", "stretch", "specaug"],
        help="Augmentation type",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = select_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Experiment naming
    aug_names = {
        "none": "baseline_no_aug",
        "pitch": "pitch_only",
        "stretch": "stretch_only",
        "specaug": "specaug_only",
    }
    aug_name = aug_names[args.augment]

    output_dir = Path(f"outputs/runs/{aug_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"QUICK ABLATION TRAINING: {aug_name}")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print()

    # Load config
    config = load_config("configs/baseline.yml")

    # Data directories
    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    # Create train/val split
    train_idx, val_idx = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=args.seed,
    )

    # Augmentation settings based on argument
    augment_pitch = args.augment == "pitch"
    augment_stretch = args.augment == "stretch"
    augment_specaug = args.augment == "specaug"

    print("Augmentation config:")
    print(f"  Pitch shift: {augment_pitch}")
    print(f"  Time stretch: {augment_stretch}")
    print(f"  SpecAugment: {augment_specaug}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = AugmentedIRMASDataset(
        data_dir=train_dir,
        augment_pitch=augment_pitch,
        augment_stretch=augment_stretch,
        augment_specaug=augment_specaug,
        indices=train_idx,
    )

    val_dataset = IRMASDataset(
        data_dir=train_dir,
        indices=val_idx,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # Create model
    print("\nCreating model...")
    model = create_densenet121(num_classes=11, pretrained=True, dropout_rate=0.5)
    model = model.to(device)

    # Loss function with class weights
    class_weights = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0

    # Calculate stage epochs
    stage1_epochs = min(3, args.epochs // 2)
    stage2_epochs = args.epochs - stage1_epochs

    # ==========================================================================
    # STAGE 1: Train classifier head only
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"STAGE 1: Training classifier head ({stage1_epochs} epochs)")
    print("=" * 70)

    model.freeze_features()
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    for epoch in range(stage1_epochs):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{stage1_epochs}: "
            f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_f1": val_f1,
                    "history": history,
                },
                output_dir / "best_model.pth",
            )
            print("  → New best! Saved checkpoint.")

    # ==========================================================================
    # STAGE 2: Fine-tune last layers
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"STAGE 2: Fine-tuning last layers ({stage2_epochs} epochs)")
    print("=" * 70)

    model.unfreeze_last_layers(num_blocks=2)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    for epoch in range(stage2_epochs):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        total_epoch = stage1_epochs + epoch + 1
        print(
            f"Epoch {total_epoch}/{args.epochs}: "
            f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": total_epoch,
                    "val_f1": val_f1,
                    "history": history,
                },
                output_dir / "best_model.pth",
            )
            print("  → New best! Saved checkpoint.")

    # ==========================================================================
    # SAVE SUMMARY
    # ==========================================================================
    summary = {
        "experiment": aug_name,
        "augmentation": {
            "augment_pitch": augment_pitch,
            "augment_stretch": augment_stretch,
            "augment_specaug": augment_specaug,
        },
        "training": {
            "total_epochs": args.epochs,
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "best_metrics": {
            "val_f1": best_val_f1,
            "train_f1": history["train_f1"][-1],
            "epoch": len(history["val_f1"]),
        },
        "history": history,
    }

    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val F1: {best_val_f1:.4f} ({best_val_f1 * 100:.2f}%)")
    print(f"Checkpoint: {output_dir / 'best_model.pth'}")
    print(f"Summary: {output_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
