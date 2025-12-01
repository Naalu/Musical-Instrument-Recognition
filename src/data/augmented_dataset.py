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
        help="Only SpecAugment (frequency/time masking)",
    )
    aug_group.add_argument(
        "--full-augment",
        action="store_true",
        help="All augmentations (pitch + stretch + SpecAugment)",
    )
    aug_group.add_argument(
        "--custom",
        action="store_true",
        help="Custom augmentation (use --pitch, --stretch, --specaug flags)",
    )

    # Custom augmentation flags (used with --custom)
    parser.add_argument(
        "--pitch", action="store_true", help="Enable pitch shift (with --custom)"
    )
    parser.add_argument(
        "--stretch", action="store_true", help="Enable time stretch (with --custom)"
    )
    parser.add_argument(
        "--specaug", action="store_true", help="Enable SpecAugment (with --custom)"
    )

    # Training configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to base config file (default: configs/baseline.yml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training (freeze then fine-tune)",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=5,
        help="Epochs for stage 1 - frozen features (default: 5)",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=45,
        help="Epochs for stage 2 - fine-tuning (default: 45)",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=1e-5,
        help="Learning rate for stage 2 (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (auto-generated if not provided)",
    )

    return parser.parse_args()


def get_augmentation_settings(args):
    """Determine augmentation settings based on command line args.

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
    elif args.custom:
        name_parts = ["custom"]
        if args.pitch:
            name_parts.append("pitch")
        if args.stretch:
            name_parts.append("stretch")
        if args.specaug:
            name_parts.append("specaug")
        return args.pitch, args.stretch, args.specaug, "_".join(name_parts)
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
        print(
            f"Training: Two-stage (Stage 1: {args.stage1_epochs} epochs, Stage 2: {args.stage2_epochs} epochs)"
        )
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
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
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

    if args.two_stage:
        # Two-stage training
        history = train_two_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            exp_dir=exp_dir,
            config=config,
            args=args,
        )
    else:
        # Single-stage training
        history = train_single_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            exp_dir=exp_dir,
            config=config,
            args=args,
        )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Best validation F1: {history['best_val_f1']:.4f}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Checkpoint saved to: {exp_dir / 'best_model.pth'}")
    print()

    # Save experiment summary
    save_experiment_summary(
        exp_dir,
        experiment_name,
        args,
        history,
        augment_pitch,
        augment_stretch,
        augment_specaug,
    )

    return history


def train_single_stage(
    model, train_loader, val_loader, criterion, device, exp_dir, config, args
):
    """Single-stage training (all parameters trainable)."""
    lr = args.lr or config["train"]["learning_rate"]
    epochs = args.epochs or config["train"]["num_epochs"]
    patience = config["train"]["early_stopping"]["patience"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
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
        patience=patience,
    )

    print(f"Training for {epochs} epochs (patience={patience})")
    print(f"Learning rate: {lr}")
    print()

    history = trainer.train(num_epochs=epochs)

    return history


def train_two_stage(
    model, train_loader, val_loader, criterion, device, exp_dir, config, args
):
    """Two-stage training with feature freezing."""
    stage1_epochs = args.stage1_epochs
    stage2_epochs = args.stage2_epochs
    stage1_lr = args.lr or config["train"]["learning_rate"]
    stage2_lr = args.stage2_lr
    patience = config["train"]["early_stopping"]["patience"]

    # -------------------------------------------------------------------------
    # Stage 1: Frozen backbone, train classifier only
    # -------------------------------------------------------------------------
    print("-" * 50)
    print(f"STAGE 1: Train classifier head ({stage1_epochs} epochs)")
    print("-" * 50)

    model.freeze_features()
    _, trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")

    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
        weight_decay=1e-4,
    )

    trainer1 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer1,
        device=device,
        checkpoint_dir=str(exp_dir / "stage1"),
        patience=stage1_epochs + 1,  # Don't early stop in stage 1
    )

    history1 = trainer1.train(num_epochs=stage1_epochs)

    print(f"\nStage 1 complete. Val F1: {history1['best_val_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Stage 2: Unfreeze and fine-tune
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"STAGE 2: Fine-tune all layers ({stage2_epochs} epochs)")
    print("-" * 50)

    model.unfreeze_features()
    _, trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Learning rate: {stage2_lr}")

    optimizer2 = torch.optim.Adam(model.parameters(), lr=stage2_lr, weight_decay=1e-4)

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="max", factor=0.5, patience=5, verbose=True
    )

    trainer2 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device=device,
        checkpoint_dir=str(exp_dir),
        patience=patience,
    )

    history2 = trainer2.train(num_epochs=stage2_epochs)

    # Combine histories
    combined_history = {
        "best_val_f1": max(history1["best_val_f1"], history2["best_val_f1"]),
        "best_epoch": history2["best_epoch"] + stage1_epochs,
        "stage1_val_f1": history1["best_val_f1"],
        "stage2_val_f1": history2["best_val_f1"],
    }

    return combined_history


def save_experiment_summary(
    exp_dir,
    experiment_name,
    args,
    history,
    augment_pitch,
    augment_stretch,
    augment_specaug,
):
    """Save experiment summary to file."""
    import json

    summary = {
        "experiment_name": experiment_name,
        "augmentation": {
            "pitch_shift": augment_pitch,
            "time_stretch": augment_stretch,
            "specaugment": augment_specaug,
        },
        "training": {
            "two_stage": args.two_stage,
            "seed": args.seed,
        },
        "results": {
            "best_val_f1": float(history["best_val_f1"]),
            "best_epoch": int(history["best_epoch"]),
        },
    }

    summary_path = exp_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
