"""Two-stage training script with feature freezing.

Stage 1: Train classifier head only (frozen features)
Stage 2: Fine-tune entire network with lower learning rate

Example:
    python scripts/train_two_stage.py
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import load_config
from src.core.paths import get_data_dir
from src.data.dataset import (
    IRMASDataset,
    create_stratified_train_val_split,
    get_class_weights,
)
from src.models.densenet import (
    count_parameters,
    create_densenet121,
    print_model_summary,
)
from src.train.trainer import Trainer
from src.utils.device import select_device
from src.utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Two-stage training for IRMAS")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yml", help="Path to config file"
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=5,
        help="Number of epochs for stage 1 (frozen features)",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=45,
        help="Number of epochs for stage 2 (full fine-tuning)",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=1e-5,
        help="Learning rate for stage 2 (default: 1e-5)",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    print("=" * 70)
    print("TWO-STAGE TRAINING")
    print("=" * 70)
    print(f"Stage 1: {args.stage1_epochs} epochs (frozen features)")
    print(f"Stage 2: {args.stage2_epochs} epochs (fine-tune all, LR={args.stage2_lr})")
    print()

    config = load_config(args.config)
    set_seed(config["train"]["random_seed"])
    device = select_device()

    print(f"Device: {device}")
    print()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    train_indices, val_indices = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=config["train"]["random_seed"],
    )

    train_dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=config["audio"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        indices=train_indices,
    )

    val_dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=config["audio"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True if device != "cpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True if device != "cpu" else False,
    )

    print()

    # Create model
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_densenet121(
        num_classes=11,
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout"],
    )

    print_model_summary(model)
    print()

    # Loss function with class weights
    print("Computing class weights...")
    class_weights = get_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print()

    # ========================================================================
    # STAGE 1: Train classifier head only
    # ========================================================================
    print("=" * 70)
    print("STAGE 1: TRAINING CLASSIFIER HEAD ONLY")
    print("=" * 70)
    print("Freezing feature extractor...")

    model.freeze_features()
    print(f"Trainable parameters: {count_parameters(model):,}")
    print()

    # Optimizer for stage 1
    optimizer_stage1 = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
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
        checkpoint_dir="checkpoints/stage1",
        patience=5,
    )

    print("\nTraining classifier head...")
    history_stage1 = trainer_stage1.train(
        num_epochs=args.stage1_epochs,
        save_every=2,
    )

    print(f"\nStage 1 complete! Best val F1: {trainer_stage1.best_val_f1:.4f}")
    print()

    # ========================================================================
    # STAGE 2: Fine-tune last layers only (not entire network)
    # ========================================================================
    print("=" * 70)
    print("STAGE 2: FINE-TUNING LAST LAYERS")
    print("=" * 70)
    print("Unfreezing last 2 dense blocks (~30% of parameters)...")

    # Instead of: model.unfreeze_features()
    # Use selective unfreezing:
    model.unfreeze_last_layers(num_blocks=2)
    print()

    # Optimizer for stage 2 with LOWER learning rate
    optimizer_stage2 = torch.optim.Adam(
        model.parameters(),
        lr=args.stage2_lr,  # Much lower!
        weight_decay=config["train"]["weight_decay"],
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
        checkpoint_dir="checkpoints/stage2",
        patience=10,
    )

    # Initialize with stage 1 best score
    trainer_stage2.best_val_f1 = trainer_stage1.best_val_f1

    print("\nFine-tuning last layers...")
    history_stage2 = trainer_stage2.train(
        num_epochs=args.stage2_epochs,
        save_every=5,
    )

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TWO-STAGE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Stage 1 best F1: {trainer_stage1.best_val_f1:.4f}")
    print(f"Stage 2 best F1: {trainer_stage2.best_val_f1:.4f}")
    print()

    if trainer_stage2.best_val_f1 >= 0.60:
        print("✅ Target F1-score of 0.60 achieved!")
    else:
        print(f"⚠️  Target not quite reached. Best: {trainer_stage2.best_val_f1:.4f}")
        print("Consider:")
        print("  - Adding data augmentation")
        print("  - Training for more epochs")
        print("  - Adjusting learning rates")

    print("\nBest model saved to: checkpoints/stage2/best_model.pth")


if __name__ == "__main__":
    main()
