#!/usr/bin/env python3
"""Comprehensive pre-flight test for augmentation pipeline.

Tests all components before running full training:
1. Individual augmentation functions (pitch, stretch, specaugment)
2. AugmentedIRMASDataset with each augmentation combination
3. DataLoader iteration
4. Model forward pass with augmented data
5. Single training step

Run this BEFORE full training to catch errors early!

Usage:
    python scripts/test_augmentation_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Track test results
TESTS_PASSED = 0
TESTS_FAILED = 0


def test_passed(name):
    global TESTS_PASSED
    TESTS_PASSED += 1
    print(f"  ✓ {name}")


def test_failed(name, error):
    global TESTS_FAILED
    TESTS_FAILED += 1
    print(f"  ✗ {name}: {error}")


def section(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print("=" * 70)


# =============================================================================
# TEST 1: Individual Augmentation Functions
# =============================================================================
def test_augmentation_functions():
    section("TEST 1: Individual Augmentation Functions")

    # Create dummy audio (3 seconds at 22050 Hz)
    sr = 22050
    duration = 3.0
    dummy_audio = np.random.randn(int(sr * duration)).astype(np.float32)

    # Create dummy spectrogram (128 mel bands x 130 frames)
    dummy_spec = np.random.randn(128, 130).astype(np.float32)

    # Test 1.1: Pitch shift
    try:
        from src.augment.pitch import pitch_shift, random_pitch_shift

        # Test direct pitch shift
        shifted = pitch_shift(dummy_audio, sr, n_steps=2.0)
        assert shifted.shape == dummy_audio.shape, "Shape mismatch after pitch shift"
        test_passed("pitch_shift()")

        # Test random pitch shift
        aug_audio = random_pitch_shift(
            dummy_audio, sr, max_steps=2.0, probability=1.0, seed=42
        )
        assert aug_audio.shape == dummy_audio.shape, (
            "Shape mismatch after random pitch shift"
        )
        test_passed("random_pitch_shift()")

    except Exception as e:
        test_failed("Pitch shift", str(e))

    # Test 1.2: Time stretch
    try:
        from src.augment.stretch import random_time_stretch, time_stretch

        # Test direct time stretch
        stretched = time_stretch(dummy_audio, rate=1.1)
        assert len(stretched) > 0, "Time stretch returned empty array"
        test_passed("time_stretch()")

        # Test random time stretch
        aug_audio = random_time_stretch(
            dummy_audio, rate_range=(0.9, 1.1), probability=1.0, seed=42
        )
        assert len(aug_audio) > 0, "Random time stretch returned empty array"
        test_passed("random_time_stretch()")

    except Exception as e:
        test_failed("Time stretch", str(e))

    # Test 1.3: SpecAugment
    try:
        from src.augment.specaugment import (
            frequency_mask,
            random_spec_augment,
            spec_augment,
            time_mask,
        )

        # Test frequency mask
        masked = frequency_mask(dummy_spec.copy(), max_mask_size=20, num_masks=1)
        assert masked.shape == dummy_spec.shape, "Shape mismatch after freq mask"
        test_passed("frequency_mask()")

        # Test time mask
        masked = time_mask(dummy_spec.copy(), max_mask_size=20, num_masks=1)
        assert masked.shape == dummy_spec.shape, "Shape mismatch after time mask"
        test_passed("time_mask()")

        # Test combined spec_augment
        augmented = spec_augment(dummy_spec.copy())
        assert augmented.shape == dummy_spec.shape, "Shape mismatch after spec_augment"
        test_passed("spec_augment()")

        # Test random spec_augment
        augmented = random_spec_augment(dummy_spec.copy(), probability=1.0, seed=42)
        assert augmented.shape == dummy_spec.shape, (
            "Shape mismatch after random_spec_augment"
        )
        test_passed("random_spec_augment()")

    except Exception as e:
        test_failed("SpecAugment", str(e))


# =============================================================================
# TEST 2: AugmentedIRMASDataset
# =============================================================================
def test_augmented_dataset():
    section("TEST 2: AugmentedIRMASDataset")

    try:
        from src.core.paths import get_data_dir
        from src.data.augmented_dataset import AugmentedIRMASDataset

        train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

        if not train_dir.exists():
            print(f"  ⚠️  IRMAS data not found at {train_dir}")
            print("  Skipping dataset tests")
            return False

        # Use small subset for testing
        indices = list(range(20))

        # Test 2.1: No augmentation
        try:
            dataset = AugmentedIRMASDataset(
                data_dir=train_dir,
                augment_pitch=False,
                augment_stretch=False,
                augment_specaug=False,
                indices=indices,
            )
            spec, label = dataset[0]
            assert spec.shape[0] == 1, "Should have 1 channel"
            assert spec.shape[1] == 128, "Should have 128 mel bands"
            test_passed("AugmentedIRMASDataset (no augmentation)")
        except Exception as e:
            test_failed("AugmentedIRMASDataset (no augmentation)", str(e))

        # Test 2.2: Pitch only
        try:
            dataset = AugmentedIRMASDataset(
                data_dir=train_dir,
                augment_pitch=True,
                augment_stretch=False,
                augment_specaug=False,
                indices=indices,
            )
            spec, label = dataset[0]
            assert spec.shape[0] == 1, "Should have 1 channel"
            test_passed("AugmentedIRMASDataset (pitch only)")
        except Exception as e:
            test_failed("AugmentedIRMASDataset (pitch only)", str(e))

        # Test 2.3: SpecAugment only (fastest)
        try:
            dataset = AugmentedIRMASDataset(
                data_dir=train_dir,
                augment_pitch=False,
                augment_stretch=False,
                augment_specaug=True,
                indices=indices,
            )
            spec, label = dataset[0]
            assert spec.shape[0] == 1, "Should have 1 channel"
            test_passed("AugmentedIRMASDataset (specaug only)")
        except Exception as e:
            test_failed("AugmentedIRMASDataset (specaug only)", str(e))

        # Test 2.4: Full augmentation
        try:
            dataset = AugmentedIRMASDataset(
                data_dir=train_dir,
                augment_pitch=True,
                augment_stretch=True,
                augment_specaug=True,
                indices=indices,
            )
            spec, label = dataset[0]
            assert spec.shape[0] == 1, "Should have 1 channel"
            test_passed("AugmentedIRMASDataset (full augmentation)")
        except Exception as e:
            test_failed("AugmentedIRMASDataset (full augmentation)", str(e))

        return True

    except Exception as e:
        test_failed("AugmentedIRMASDataset import", str(e))
        return False


# =============================================================================
# TEST 3: DataLoader Integration
# =============================================================================
def test_dataloader():
    section("TEST 3: DataLoader Integration")

    try:
        from src.core.paths import get_data_dir
        from src.data.augmented_dataset import AugmentedIRMASDataset

        train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

        if not train_dir.exists():
            print("  ⚠️  Skipping (no data)")
            return False

        # Create dataset with specaugment (fastest)
        dataset = AugmentedIRMASDataset(
            data_dir=train_dir,
            augment_pitch=False,
            augment_stretch=False,
            augment_specaug=True,
            indices=list(range(32)),  # Small subset
        )

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
        )

        # Iterate through one batch
        for batch_idx, (specs, labels) in enumerate(loader):
            assert specs.shape[0] <= 8, "Batch size should be <= 8"
            assert specs.shape[1] == 1, "Should have 1 channel"
            assert specs.shape[2] == 128, "Should have 128 mel bands"
            assert len(labels) == specs.shape[0], "Labels should match batch size"

            if batch_idx == 0:
                print(f"  Batch shape: {specs.shape}")
                print(f"  Labels: {labels.tolist()}")

            if batch_idx >= 2:  # Test 3 batches
                break

        test_passed("DataLoader iteration")
        return True

    except Exception as e:
        test_failed("DataLoader", str(e))
        return False


# =============================================================================
# TEST 4: Model Forward Pass with Augmented Data
# =============================================================================
def test_model_forward():
    section("TEST 4: Model Forward Pass")

    try:
        from src.models.densenet import create_densenet121
        from src.utils.device import select_device

        device = select_device()
        print(f"  Device: {device}")

        # Create model (no pretrained weights for speed)
        model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)
        model = model.to(device)
        model.eval()

        # Create dummy batch (simulating augmented spectrograms)
        batch = torch.randn(4, 1, 128, 130).to(device)

        with torch.no_grad():
            logits = model(batch)

        assert logits.shape == (4, 11), f"Expected (4, 11), got {logits.shape}"
        test_passed(f"Forward pass on {device}")

        return True

    except Exception as e:
        test_failed("Model forward pass", str(e))
        return False


# =============================================================================
# TEST 5: Single Training Step
# =============================================================================
def test_training_step():
    section("TEST 5: Single Training Step")

    try:
        from src.core.paths import get_data_dir
        from src.data.augmented_dataset import AugmentedIRMASDataset
        from src.models.densenet import create_densenet121
        from src.utils.device import select_device

        device = select_device()
        train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

        if not train_dir.exists():
            print("  ⚠️  Skipping (no data)")
            return False

        # Small dataset
        dataset = AugmentedIRMASDataset(
            data_dir=train_dir,
            augment_pitch=False,
            augment_stretch=False,
            augment_specaug=True,
            indices=list(range(16)),
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

        # Model
        model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)
        model = model.to(device)
        model.train()

        # Training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Single training step
        specs, labels = next(iter(loader))
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}")
        test_passed("Single training step")

        return True

    except Exception as e:
        test_failed("Training step", str(e))
        return False


# =============================================================================
# TEST 6: Config and Script Compatibility
# =============================================================================
def test_config_compatibility():
    section("TEST 6: Config Compatibility")

    try:
        from src.core.config import load_config

        config = load_config("configs/baseline.yml")

        # Check required keys exist
        required_train_keys = ["learning_rate", "batch_size", "num_epochs"]
        for key in required_train_keys:
            if key not in config["train"]:
                test_failed(f"Config missing train.{key}", "Key not found")
                return False
        test_passed("Required train keys present")

        # Check patience (the key that caused the error)
        patience = config["train"].get(
            "patience", config["train"].get("early_stopping_patience", 10)
        )
        print(f"  Early stopping patience: {patience}")
        test_passed("Patience config accessible")

        # Check model keys
        required_model_keys = ["pretrained", "dropout", "num_classes"]
        for key in required_model_keys:
            if key not in config["model"]:
                test_failed(f"Config missing model.{key}", "Key not found")
                return False
        test_passed("Required model keys present")

        return True

    except Exception as e:
        test_failed("Config loading", str(e))
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(" AUGMENTATION PIPELINE PRE-FLIGHT TEST")
    print("=" * 70)
    print("\nThis test validates all components before full training.\n")

    # Run all tests
    test_augmentation_functions()
    has_data = test_augmented_dataset()
    if has_data:
        test_dataloader()
    test_model_forward()
    if has_data:
        test_training_step()
    test_config_compatibility()

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  Tests passed: {TESTS_PASSED}")
    print(f"  Tests failed: {TESTS_FAILED}")

    if TESTS_FAILED == 0:
        print("\n" + "=" * 70)
        print(" ✅ ALL TESTS PASSED - Ready for full training!")
        print("=" * 70)
        print("\nYou can now run:")
        print("  python scripts/train_augmented.py --specaug-only --two-stage")
        print("  python scripts/train_augmented.py --full-augment --two-stage")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f" ❌ {TESTS_FAILED} TEST(S) FAILED - Fix errors before training!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
