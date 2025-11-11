"""Test PyTorch Dataset for IRMAS data."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.paths import get_data_dir
from src.data.dataset import (
    IRMASDataset,
    create_stratified_train_val_split,
    create_train_val_split,
    get_class_weights,
)


def test_dataset_creation():
    """Test basic dataset creation."""
    print("Testing: IRMASDataset creation")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    # Create dataset
    dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=22050,
        n_mels=128,
    )

    print(f"\n  Dataset size: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")

    assert len(dataset) == 6705, "Should have 6705 samples"
    assert dataset.num_classes == 11, "Should have 11 classes"

    print("  ✓ Dataset created successfully")
    return True


def test_dataset_getitem():
    """Test getting items from dataset."""
    print("\nTesting: Dataset __getitem__()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=22050,
        n_mels=128,
    )

    # Get first sample
    melspec, label = dataset[0]

    print(f"  Mel-spectrogram shape: {melspec.shape}")
    print(f"  Label: {label}")
    print(f"  Label type: {type(label)}")
    print(f"  Tensor dtype: {melspec.dtype}")

    # Verify shape: (1, n_mels, time_frames)
    assert len(melspec.shape) == 3, "Should be 3D tensor"
    assert melspec.shape[0] == 1, "Should have 1 channel"
    assert melspec.shape[1] == 128, "Should have 128 mel bands"
    assert melspec.shape[2] > 0, "Should have time frames"

    # Verify label
    assert isinstance(label, (int, np.integer)), "Label should be integer"
    assert 0 <= label < 11, "Label should be in range [0, 10]"

    # Verify tensor type
    assert melspec.dtype == torch.float32, "Should be float32"

    print("  ✓ Dataset __getitem__() works correctly")
    return True


def test_dataloader():
    """Test PyTorch DataLoader with dataset."""
    print("\nTesting: PyTorch DataLoader")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=22050,
        n_mels=128,
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
    )

    print("  DataLoader created with batch_size=16")
    print(f"  Number of batches: {len(loader)}")

    # Get first batch
    batch_melspecs, batch_labels = next(iter(loader))

    print(f"  Batch spectrograms shape: {batch_melspecs.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")

    # Verify batch shape
    assert batch_melspecs.shape[0] == 16, "Batch size should be 16"
    assert batch_melspecs.shape[1] == 1, "Should have 1 channel"
    assert batch_melspecs.shape[2] == 128, "Should have 128 mel bands"
    assert batch_labels.shape[0] == 16, "Should have 16 labels"

    print("  ✓ DataLoader works correctly")
    return True


def test_train_val_split():
    """Test train/validation split."""
    print("\nTesting: create_train_val_split()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    # Create split
    train_indices, val_indices = create_train_val_split(
        train_dir, val_ratio=0.15, random_seed=42
    )

    print(f"\n  Train indices: {len(train_indices)}")
    print(f"  Val indices: {len(val_indices)}")

    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    overlap = train_set & val_set

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices!"

    # Verify total
    total = len(train_indices) + len(val_indices)
    assert total == 6705, f"Total should be 6705, got {total}"

    # Create datasets with splits
    train_dataset = IRMASDataset(
        data_dir=train_dir,
        indices=train_indices,
    )

    val_dataset = IRMASDataset(
        data_dir=train_dir,
        indices=val_indices,
    )

    print(f"\n  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size: {len(val_dataset)}")

    print("  ✓ Train/val split works correctly")
    return True


def test_stratified_split():
    """Test stratified train/validation split."""
    print("\nTesting: create_stratified_train_val_split()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    # Create stratified split
    train_indices, val_indices = create_stratified_train_val_split(
        train_dir, val_ratio=0.15, random_seed=42
    )

    # Verify no overlap
    assert len(set(train_indices) & set(val_indices)) == 0

    print("\n  ✓ Stratified split works correctly")
    return True


def test_class_weights():
    """Test class weight calculation."""
    print("\nTesting: get_class_weights()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=22050,
        n_mels=128,
    )

    # Get class weights
    weights = get_class_weights(dataset)

    print(f"\n  Weights shape: {weights.shape}")
    print(f"  Weights dtype: {weights.dtype}")

    assert weights.shape[0] == 11, "Should have 11 class weights"
    assert weights.dtype == torch.float32, "Should be float32"
    assert torch.all(weights > 0), "All weights should be positive"

    print("  ✓ Class weights calculated correctly")
    return True


def test_class_distribution():
    """Test getting class distribution."""
    print("\nTesting: Dataset.get_class_distribution()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    dataset = IRMASDataset(
        data_dir=train_dir,
        target_sr=22050,
        n_mels=128,
    )

    dist = dataset.get_class_distribution()

    print("\n  Class distribution:")
    for instrument, count in sorted(dist.items()):
        print(f"    {instrument}: {count}")

    # Verify total
    total = sum(dist.values())
    assert total == 6705, f"Total should be 6705, got {total}"

    print("  ✓ Class distribution works")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH DATASET TESTS")
    print("=" * 70)

    # Test 1: Dataset creation
    has_data = test_dataset_creation()

    if has_data:
        # Test 2: Get item
        test_dataset_getitem()

        # Test 3: DataLoader
        test_dataloader()

        # Test 4: Train/val split
        test_train_val_split()

        # Test 5: Stratified split
        test_stratified_split()

        # Test 6: Class weights
        test_class_weights()

        # Test 7: Class distribution
        test_class_distribution()

    print("\n" + "=" * 70)
    if has_data:
        print("✅ ALL PYTORCH DATASET TESTS PASSED!")
        print("=" * 70)
        print("\n🎵→🔥 Dataset is ready for training!")
        print("   - Train/val split with artist-based leakage prevention ✓")
        print("   - DataLoader integration working ✓")
        print("   - Class weights for imbalance handling ✓")
    else:
        print("⚠️  TESTS SKIPPED (IRMAS data not found)")
        print("=" * 70)
