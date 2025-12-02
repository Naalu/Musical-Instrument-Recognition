#!/usr/bin/env python3
"""Training script with configurable data augmentation AND proper two-stage training.

FIXED VERSION: Actually implements two-stage training when --two-stage flag is used.

Stage 1: Train classifier head only (5 epochs, frozen backbone, LR=1e-4)
Stage 2: Fine-tune last 30% of model (45 epochs, unfrozen layers, LR=1e-5)

Usage:
    python scripts/train_augmented.py --no-augment --two-stage --seed 42
    python scripts/train_augmented.py --full-augment --two-stage --seed 42
"""

import argparse
import json
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
    )

    # Augmentation presets
    aug_group = parser.add_mutually_exclusive_group(required=True)
    aug_group.add_argument("--no-augment", action="store_true")
    aug_group.add_argument("--pitch-only", action="store_true")
    aug_group.add_argument("--stretch-only", action="store_true")
    aug_group.add_argument("--specaug-only", action="store_true")
    aug_group.add_argument("--full-augment", action="store_true")

    # Training options
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training (freeze then unfreeze)",
    )
    parser.add_argument("--config", type=str, default="configs/baseline.yml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default=None)

    # Two-stage specific parameters
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=5,
        help="Epochs for stage 1 (frozen backbone)",
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=45, help="Epochs for stage 2 (unfrozen)"
    )
    parser.add_argument(
        "--stage2-lr", type=float, default=1e-5, help="Learning rate for stage 2"
    )

    return parser.parse_args()


def get_augmentation_settings(args):
    """Get augmentation flags."""
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


def create_experiment_dir(experiment_name: str) -> Path:
    """Create directory for experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("outputs/runs") / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_summary(exp_dir: Path, args, config, trainer, augment_config):
    """Save comprehensive experiment summary."""
    summary = {
        "experiment_name": exp_dir.name,
        "timestamp": datetime.now().isoformat(),
        "augmentation": {
            "augment_pitch": augment_config[0],
            "augment_stretch": augment_config[1],
            "augment_specaug": augment_config[2],
        },
        "training": {
            "two_stage": args.two_stage,
            "total_epochs": len(trainer.history["val_f1"]),
            "seed": args.seed,
        },
        "best_metrics": {
            "epoch": trainer.history["val_f1"].index(trainer.best_val_f1) + 1,
            "val_f1": trainer.best_val_f1,
            "val_loss": min(trainer.history["val_loss"]),
            "train_f1": max(trainer.history["train_f1"]),
        },
        "config_path": args.config,
    }

    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save training history as CSV
    import pandas as pd

    history_df = pd.DataFrame(
        {
            "epoch": range(1, len(trainer.history["val_f1"]) + 1),
            "train_loss": trainer.history["train_loss"],
            "train_f1": trainer.history["train_f1"],
            "val_loss": trainer.history["val_loss"],
            "val_f1": trainer.history["val_f1"],
            "lr": trainer.history["lr"],
        }
    )
    history_df.to_csv(exp_dir / "training_history.csv", index=False)


def main():
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
        f"Augmentation: pitch={augment_pitch}, stretch={augment_stretch}, "
        f"specaug={augment_specaug}"
    )
    if args.two_stage:
        print(
            f"Training: Two-stage (Stage1: {args.stage1_epochs}ep frozen, "
            f"Stage2: {args.stage2_epochs}ep unfrozen @ LR={args.stage2_lr})"
        )
    else:
        print("Training: Single-stage (full model)")
    print()

    # Load config and setup
    config = load_config(args.config)
    set_seed(args.seed)
    device = select_device()

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
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
    train_indices, val_indices = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=args.seed,
    )

    # Training dataset WITH augmentation
    print("\nCreating training dataset...")
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
    print("Creating validation dataset...")
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
    batch_size = args.batch_size or config["train"]["batch_size"]
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

    print("Model: DenseNet121")
    print(f"Pretrained: {config['model']['pretrained']}")
    print()

    # =========================================================================
    # TRAINING
    # =========================================================================

    if args.two_stage:
        # =====================================================================
        # STAGE 1: Train classifier head only
        # =====================================================================
        print("=" * 70)
        print("STAGE 1: TRAINING CLASSIFIER HEAD ONLY")
        print("=" * 70)

        model.freeze_features()
        trainable_params = count_parameters(model)
        print(f"Frozen backbone, trainable parameters: {trainable_params:,}")
        print()

        optimizer_stage1 = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"].get("weight_decay", 1e-4),
        )

        scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage1,
            mode="max",
            factor=0.5,
            patience=3,
        )

        trainer_stage1 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_stage1,
            scheduler=scheduler_stage1,
            device=device,
            checkpoint_dir=str(exp_dir / "stage1"),
            patience=5,
        )

        print(f"Training for {args.stage1_epochs} epochs...")
        history_stage1 = trainer_stage1.train(num_epochs=args.stage1_epochs)

        print(f"\nStage 1 complete! Best val F1: {trainer_stage1.best_val_f1:.4f}")
        print()

        # =====================================================================
        # STAGE 2: Fine-tune last layers
        # =====================================================================
        print("=" * 70)
        print("STAGE 2: FINE-TUNING LAST LAYERS")
        print("=" * 70)

        model.unfreeze_last_layers(num_blocks=2)  # Unfreeze last 30%
        trainable_params = count_parameters(model)
        print(f"Unfroze last 2 blocks, trainable parameters: {trainable_params:,}")
        print()

        optimizer_stage2 = torch.optim.Adam(
            model.parameters(),
            lr=args.stage2_lr,  # Lower LR for fine-tuning
            weight_decay=config["train"].get("weight_decay", 1e-4),
        )

        scheduler_stage2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage2,
            mode="max",
            factor=0.5,
            patience=5,
        )

        trainer_stage2 = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_stage2,
            scheduler=scheduler_stage2,
            device=device,
            checkpoint_dir=str(exp_dir),
            patience=config["train"].get("patience", 10),
        )

        # Initialize with stage 1 best score
        trainer_stage2.best_val_f1 = trainer_stage1.best_val_f1
        trainer_stage2.history = trainer_stage1.history  # Continue history

        print(f"Training for {args.stage2_epochs} epochs...")
        history_stage2 = trainer_stage2.train(num_epochs=args.stage2_epochs)

        # Use stage 2 trainer for final results
        final_trainer = trainer_stage2

    else:
        # =====================================================================
        # SINGLE-STAGE: Train full model
        # =====================================================================
        print("=" * 70)
        print("SINGLE-STAGE TRAINING")
        print("=" * 70)

        trainable_params = count_parameters(model)
        print(f"Trainable parameters: {trainable_params:,}")
        print()

        epochs = args.epochs or config["train"]["num_epochs"]

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["learning_rate"],
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

        print(f"Training for {epochs} epochs...")
        history = trainer.train(num_epochs=epochs)

        final_trainer = trainer

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Best validation F1: {final_trainer.best_val_f1:.4f}")
    print(f"Checkpoint saved to: {exp_dir / 'best_model.pth'}")
    print()

    if final_trainer.best_val_f1 >= 0.60:
        print("✅ Target F1-score of 0.60 achieved!")
    else:
        print(f"⚠️  Target not quite reached. Best: {final_trainer.best_val_f1:.4f}")

    # Save experiment summary
    save_experiment_summary(
        exp_dir,
        args,
        config,
        final_trainer,
        (augment_pitch, augment_stretch, augment_specaug),
    )

    print(f"\nExperiment summary saved to: {exp_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
