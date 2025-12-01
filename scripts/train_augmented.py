#!/usr/bin/env python3
"""Training script with configurable data augmentation for ablation studies.

This script supports running ablation experiments to measure the impact of
each augmentation technique independently and in combination.

Ablation Experiments:
    python scripts/train_augmented.py --no-augment      # Baseline (no aug)
    python scripts/train_augmented.py --pitch-only      # Pitch shift only
    python scripts/train_augmented.py --stretch-only    # Time stretch only
    python scripts/train_augmented.py --specaug-only    # SpecAugment only
    python scripts/train_augmented.py --full-augment    # All augmentations

Two-Stage Training (recommended):
    python scripts/train_augmented.py --full-augment --two-stage

Example:
    # Full augmentation with two-stage training
    python scripts/train_augmented.py --full-augment --two-stage

    # Quick test (2 epochs)
    python scripts/train_augmented.py --specaug-only --epochs 2
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
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
from src.models.densenet import count_parameters, create_densenet121
from src.train.trainer import Trainer
from src.utils.device import select_device
from src.utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train IRMAS classifier with configurable augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Study Examples:
  %(prog)s --no-augment        # Baseline (no augmentation)
  %(prog)s --pitch-only        # Only pitch shifting
  %(prog)s --stretch-only      # Only time stretching
  %(prog)s --specaug-only      # Only SpecAugment
  %(prog)s --full-augment      # All augmentations combined

Two-Stage Training (recommended for best results):
  %(prog)s --full-augment --two-stage
        """,
    )

    # Augmentation presets (mutually exclusive)
    aug_group = parser.add_mutually_exclusive_group(required=True)
    aug_group.add_argument(
        "--no-augment",
        action="store_true",
        help="No augmentation (baseline)",
    )
    aug_group.add_argument(
        "--pitch-only",
        action="store_true",
        help="Only pitch shifting augmentation",
    )
    aug_group.add_argument(
        "--stretch-only",
        action="store_true",
        help="Only time stretching augmentation",
    )
    aug_group.add_argument(
        "--specaug-only",
        action="store_true",
        help="Only SpecAugment",
    )
    aug_group.add_argument(
        "--full-augment",
        action="store_true",
        help="All augmentations combined",
    )

    # Optional: Two-stage training
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training (freeze then unfreeze)",
    )

    # Training parameters
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name",
    )

    return parser.parse_args()


def get_augmentation_settings(args):
    """Get augmentation flags from command line arguments.

    Returns:
        Tuple of (augment_pitch, augment_stretch, augment_specaug, experiment_name)
    """
    if args.no_augment:
        return False, False, False, "baseline_no_aug"
    elif args.pitch_only:
        return True, False, False, "ablation_pitch_only"
    elif args.stretch_only:
        return False, True, False, "ablation_stretch_only"
    elif args.specaug_only:
        return False, False, True, "ablation_specaug_only"
    elif args.full_augment:
        return True, True, True, "full_augment"
    else:
        raise ValueError("No augmentation preset specified")


def create_experiment_dir(experiment_name: str) -> Path:
    """Create directory for experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("outputs/runs") / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def main():
    """Main training function."""
    args = parse_args()

    # Get augmentation settings
    augment_pitch, augment_stretch, augment_specaug, auto_name = (
        get_augmentation_settings(args)
    )
    experiment_name = args.experiment_name or auto_name

    # Print header
    print("=" * 70)
    print("AUGMENTED TRAINING - ABLATION STUDY")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(
        f"Augmentation: pitch={augment_pitch}, stretch={augment_stretch}, specaug={augment_specaug}"
    )
    if args.two_stage:
        print("Training: Two-stage (freeze then unfreeze)")
    print()

    # Load config
    config = load_config(args.config)

    # Set seed for reproducibility
    seed = args.seed
    set_seed(seed)

    # Select device
    device = select_device()
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()

    # Create experiment directory
    exp_dir = create_experiment_dir(experiment_name)
    print(f"Output directory: {exp_dir}")
    print()

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    # Create train/val split
    train_indices, val_indices = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=seed,
    )

    # Feature extraction parameters
    target_sr = config["audio"]["sample_rate"]
    n_mels = config["features"]["n_mels"]
    n_fft = config["features"]["n_fft"]
    hop_length = config["features"]["hop_length"]

    # Training dataset WITH augmentation
    print("\nCreating training dataset...")
    train_dataset = AugmentedIRMASDataset(
        data_dir=train_dir,
        augment_pitch=augment_pitch,
        augment_stretch=augment_stretch,
        augment_specaug=augment_specaug,
        target_sr=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        indices=train_indices,
    )

    # Validation dataset WITHOUT augmentation (fair evaluation)
    print("\nCreating validation dataset...")
    val_dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        indices=val_indices,
    )

    # Create data loaders
    batch_size = args.batch_size or config["train"]["batch_size"]
    num_workers = config.get("compute", {}).get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get class weights for handling imbalance
    # Need to use non-augmented dataset for weight calculation
    temp_dataset = IRMASDataset(data_dir=train_dir, indices=train_indices)
    class_weights = get_class_weights(temp_dataset).to(device)

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    model = create_densenet121(
        num_classes=config["data"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout"],
    )

    trainable_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    epochs = args.epochs or config["train"]["num_epochs"]
    lr = config["train"]["learning_rate"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(exp_dir),
        patience=config["train"].get("patience", 10),
    )

    print(f"Training for {epochs} epochs")
    print(f"Learning rate: {lr}")
    print()

    # Train
    history = trainer.train(num_epochs=epochs)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoint saved to: {exp_dir / 'best_model.pth'}")
    print()

    if trainer.best_val_f1 >= 0.60:
        print("✅ Target F1-score of 0.60 achieved!")
    else:
        print(f"⚠️  Target not quite reached. Best: {trainer.best_val_f1:.4f}")

    # Save experiment summary
    summary_path = exp_dir / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(
            f"Augmentation: pitch={augment_pitch}, stretch={augment_stretch}, specaug={augment_specaug}\n"
        )
        f.write(f"Best validation F1: {trainer.best_val_f1:.4f}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
