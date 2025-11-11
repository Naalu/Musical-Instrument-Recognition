"""Training script for IRMAS instrument classification.

Example:
    # Train with default configuration
    python scripts/train.py

    # Train with custom config
    python scripts/train.py --config configs/baseline.yml

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import load_config
from src.core.device import get_device
from src.core.paths import get_data_dir
from src.core.seed import set_seed
from src.data.dataset import (
    IRMASDataset,
    create_stratified_train_val_split,
    get_class_weights,
)
from src.models.densenet import create_densenet121, print_model_summary
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IRMAS instrument classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to config file (default: configs/baseline.yml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda/mps). If not specified, auto-detect.",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load config
    print("=" * 70)
    print("LOADING CONFIGURATION")
    print("=" * 70)
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print()

    # Set random seed
    set_seed(config["training"]["random_seed"])
    print(f"Random seed: {config['training']['random_seed']}")
    print()

    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")
    print()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    # Create train/val split
    train_indices, val_indices = create_stratified_train_val_split(
        train_dir,
        val_ratio=config["data"]["val_split"],
        random_seed=config["training"]["random_seed"],
    )

    # Create datasets
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
        n_fft=config["features"]["hop_length"],
        hop_length=config["features"]["hop_length"],
        indices=val_indices,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if device != "cpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if device != "cpu" else False,
    )

    print("\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
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

    # Create loss function
    if config["training"]["use_class_weights"]:
        print("Computing class weights...")
        class_weights = get_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    print()

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    print("Optimizer: Adam")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    print()

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    print("Learning rate scheduler: ReduceLROnPlateau")
    print("  Factor: 0.5")
    print("  Patience: 5 epochs")
    print()

    # Create trainer
    print("=" * 70)
    print("CREATING TRAINER")
    print("=" * 70)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        patience=config["training"]["early_stopping_patience"],
    )

    print()

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()

    # Train
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_every=config["training"]["save_every"],
    )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"\nBest validation F1-score: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    print()


if __name__ == "__main__":
    main()
