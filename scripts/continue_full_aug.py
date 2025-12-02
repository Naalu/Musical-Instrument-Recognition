#!/usr/bin/env python3
"""Continue training from a checkpoint with additional epochs.

Usage:
    python scripts/continue_full_aug.py \
        --checkpoint outputs/runs/full_augment_20251201_093013/best_model.pth \
        --additional-epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import load_config
from src.core.paths import get_data_dir
from src.data.augmented_dataset import AugmentedIRMASDataset
from src.data.dataset import (
    IRMASDataset,
    create_stratified_train_val_split,
    get_class_weights,
)
from src.models.densenet import create_densenet121
from src.train.trainer import Trainer
from src.utils.device import select_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., outputs/runs/full_augment_*/best_model.pth)",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=50,
        help="Number of additional epochs to train (default: 50)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (should match original training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Get the experiment directory
    exp_dir = checkpoint_path.parent

    print("=" * 70)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment dir: {exp_dir}")
    print(f"Additional epochs: {args.additional_epochs}")
    print()

    # Load config and setup
    config = load_config(args.config)
    set_seed(args.seed)
    device = select_device()

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print()

    # Load original experiment summary to get augmentation settings
    summary_file = exp_dir / "experiment_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)

        augment_config = summary.get("augmentation", {})
        augment_pitch = augment_config.get("augment_pitch", True)
        augment_stretch = augment_config.get("augment_stretch", True)
        augment_specaug = augment_config.get("augment_specaug", True)

        print("Original augmentation settings:")
        print(f"  Pitch shift: {augment_pitch}")
        print(f"  Time stretch: {augment_stretch}")
        print(f"  SpecAugment: {augment_specaug}")
        print()
    else:
        print("⚠️  No experiment summary found, assuming full augmentation")
        augment_pitch = augment_stretch = augment_specaug = True

    # =========================================================================
    # DATA LOADING (Same as original)
    # =========================================================================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"
    train_indices, val_indices = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=args.seed,
    )

    # Training dataset WITH augmentation (same as original)
    train_dataset = AugmentedIRMASDataset(
        data_dir=train_dir,
        augment_pitch=augment_pitch,
        augment_stretch=augment_stretch,
        augment_specaug=augment_specaug,
        target_sr=config["audio"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        indices=train_indices,
    )

    # Validation dataset WITHOUT augmentation
    val_dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=config["audio"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        indices=val_indices,
    )

    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print()

    # Create data loaders
    batch_size = config["train"]["batch_size"]
    use_pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=use_pin_memory,
    )

    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    print("=" * 70)
    print("MODEL SETUP")
    print("=" * 70)

    model = create_densenet121(
        num_classes=11,
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout"],
    )
    model = model.to(device)

    # Get class weights
    class_weights = get_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # =========================================================================
    # LOAD CHECKPOINT
    # =========================================================================
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Model state loaded")

    # Get checkpoint info
    start_epoch = checkpoint.get("epoch", 0)
    best_val_f1 = checkpoint.get("val_f1", 0.0)
    print(f"✓ Checkpoint from epoch: {start_epoch}")
    print(f"✓ Best val F1 so far: {best_val_f1:.4f}")

    # Load training history if available
    history = checkpoint.get(
        "history",
        {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        },
    )
    print(f"✓ Loaded {len(history.get('val_f1', []))} epochs of history")
    print()

    # =========================================================================
    # CONTINUE TRAINING
    # =========================================================================
    print("=" * 70)
    print("CONTINUE TRAINING")
    print("=" * 70)

    # Create optimizer (will be updated from checkpoint if available)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,  # Stage 2 LR (full aug was in stage 2)
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )

    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✓ Optimizer state loaded")

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # Load scheduler state if available
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✓ Scheduler state loaded")

    print()
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Training for {args.additional_epochs} more epochs...")
    print()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(exp_dir / "continued"),  # Save to subdirectory
        patience=config["train"].get("patience", 10),
    )

    # Initialize trainer with checkpoint state
    trainer.best_val_f1 = best_val_f1
    trainer.history = history

    # Train for additional epochs
    trainer.train(num_epochs=args.additional_epochs)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Starting val F1: {best_val_f1:.4f}")
    print(f"Final val F1: {trainer.best_val_f1:.4f}")
    print(f"Improvement: {(trainer.best_val_f1 - best_val_f1):.4f}")
    print()

    if trainer.best_val_f1 >= 0.67:
        print("✅ Exceeded individual augmentation performance!")
    elif trainer.best_val_f1 > best_val_f1:
        print("✅ Improvement achieved!")
    else:
        print("ℹ️  Model may have converged at previous checkpoint")

    print(f"\nContinued checkpoints saved to: {exp_dir / 'continued'}")

    # Update experiment summary
    summary["training"]["total_epochs"] = len(trainer.history["val_f1"])
    summary["training"]["continued_from_epoch"] = start_epoch
    summary["best_metrics"]["val_f1"] = trainer.best_val_f1
    summary["best_metrics"]["epoch"] = (
        trainer.history["val_f1"].index(trainer.best_val_f1) + 1
    )

    with open(exp_dir / "experiment_summary_continued.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Updated summary: {exp_dir / 'experiment_summary_continued.json'}")


if __name__ == "__main__":
    main()
