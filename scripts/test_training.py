"""Quick test of training pipeline (1 epoch)."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Subset

from src.core.paths import get_data_dir
from src.data.dataset import IRMASDataset
from src.models.densenet import create_densenet121
from src.train.trainer import Trainer
from src.utils.device import select_device
from src.utils.seed import set_seed


def main():
    print("=" * 70)
    print("QUICK TRAINING TEST (1 EPOCH)")
    print("=" * 70)

    # Setup
    set_seed(42)
    device = select_device()

    # Small subset for testing (100 samples)
    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"
    dataset = IRMASDataset(train_dir, target_sr=22050, n_mels=128)

    # Take only first 100 samples
    subset_indices = list(range(100))
    train_subset = Subset(dataset, subset_indices[:80])
    val_subset = Subset(dataset, subset_indices[80:100])

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    print(f"Using {len(train_subset)} train + {len(val_subset)} val samples")
    print(f"Device: {device}")
    print()

    # Model
    model = create_densenet121(num_classes=11, pretrained=False)

    # Training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="checkpoints/test",
        patience=5,
    )

    # Train for 1 epoch
    print("\nTraining for 1 epoch...\n")
    history = trainer.train(num_epochs=1)

    print("\n" + "=" * 70)
    print("✅ TRAINING TEST PASSED!")
    print("=" * 70)
    print("All components working correctly:")
    print("  ✓ Data loading")
    print("  ✓ Model forward/backward pass")
    print("  ✓ Optimizer step")
    print("  ✓ Validation")
    print("  ✓ Metrics computation")
    print("  ✓ Checkpointing")
    print("\nReady for full training!")


if __name__ == "__main__":
    main()
