"""Test random seed utilities for reproducibility."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random

import numpy as np
import torch

from src.utils.seed import (
    get_rng_state,
    seed_everything,
    set_rng_state,
    set_seed,
    worker_init_fn,
)


def test_set_seed():
    """Test that setting seed produces reproducible results."""
    print("Testing: set_seed()")

    # Set seed and generate random numbers
    set_seed(42)
    py_rand_1 = random.random()
    np_rand_1 = np.random.rand()
    torch_rand_1 = torch.rand(1).item()

    # Reset seed and generate again
    set_seed(42)
    py_rand_2 = random.random()
    np_rand_2 = np.random.rand()
    torch_rand_2 = torch.rand(1).item()

    # Should be identical
    assert py_rand_1 == py_rand_2, "Python random should be reproducible"
    assert np_rand_1 == np_rand_2, "NumPy random should be reproducible"
    assert torch_rand_1 == torch_rand_2, "PyTorch random should be reproducible"

    print(f"  ✓ Python random: {py_rand_1:.6f} (reproducible)")
    print(f"  ✓ NumPy random: {np_rand_1:.6f} (reproducible)")
    print(f"  ✓ PyTorch random: {torch_rand_1:.6f} (reproducible)")
    print("  ✓ All random generators are reproducible with same seed")


def test_different_seeds():
    """Test that different seeds produce different results."""
    print("\nTesting: Different seeds produce different results")

    set_seed(42)
    result_1 = torch.rand(5)

    set_seed(123)
    result_2 = torch.rand(5)

    # Should be different
    assert not torch.allclose(result_1, result_2), (
        "Different seeds should give different results"
    )

    print(f"  ✓ Seed 42:  {result_1.numpy()}")
    print(f"  ✓ Seed 123: {result_2.numpy()}")
    print("  ✓ Different seeds produce different results")


def test_rng_state_save_restore():
    """Test saving and restoring RNG state."""
    print("\nTesting: get_rng_state() and set_rng_state()")

    set_seed(42)

    # Generate some random numbers
    before_1 = torch.rand(3)

    # Save state
    saved_state = get_rng_state()
    print("  ✓ RNG state saved")

    # Generate more random numbers
    after_save = torch.rand(3)

    # Restore state
    set_rng_state(saved_state)
    print("  ✓ RNG state restored")

    # Should generate the same numbers as after_save
    after_restore = torch.rand(3)

    assert torch.allclose(after_save, after_restore), (
        "Restored state should reproduce same values"
    )
    print(f"  ✓ After save:    {after_save.numpy()}")
    print(f"  ✓ After restore: {after_restore.numpy()}")
    print("  ✓ RNG state successfully restored")


def test_worker_init_fn():
    """Test DataLoader worker initialization function."""
    print("\nTesting: worker_init_fn()")

    # Simulate worker initialization
    set_seed(42)

    # Initialize workers with different IDs and collect values
    worker_init_fn(0)
    worker_0_value = np.random.rand()

    worker_init_fn(1)
    worker_1_value = np.random.rand()

    worker_init_fn(2)
    worker_2_value = np.random.rand()

    print(f"  ✓ Worker 0: {worker_0_value:.6f}")
    print(f"  ✓ Worker 1: {worker_1_value:.6f}")
    print(f"  ✓ Worker 2: {worker_2_value:.6f}")

    # CRITICAL: Verify they're actually different!
    assert worker_0_value != worker_1_value, (
        "Workers 0 and 1 should have different seeds"
    )
    assert worker_1_value != worker_2_value, (
        "Workers 1 and 2 should have different seeds"
    )
    assert worker_0_value != worker_2_value, (
        "Workers 0 and 2 should have different seeds"
    )

    print("  ✓ Each worker gets different random sequence (verified!)")


def test_seed_everything():
    """Test the convenience function."""
    print("\nTesting: seed_everything()")

    # Use convenience function
    seed_everything(42, deterministic=False)

    values = torch.rand(5)
    print(f"  ✓ Generated values: {values.numpy()}")

    # Should be reproducible
    seed_everything(42, deterministic=False)
    values_2 = torch.rand(5)

    assert torch.allclose(values, values_2), "seed_everything should be reproducible"
    print("  ✓ seed_everything() works correctly")


def test_mps_availability():
    """Test MPS (Apple Silicon GPU) availability."""
    print("\nTesting: MPS (Apple Silicon GPU) availability")

    if torch.backends.mps.is_available():
        print("  ✓ MPS is available!")
        print("  ✓ Your M1 Pro GPU can be used for training 🚀")

        # Test that we can create tensors on MPS
        device = torch.device("mps")
        x = torch.rand(3, 3, device=device)
        print(f"  ✓ Created tensor on MPS: shape {x.shape}")
    else:
        print("  ⚠️  MPS not available (this is unexpected on M1 Pro)")
        print("  ℹ️  You may need to update PyTorch")


if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM SEED UTILITIES TESTS")
    print("=" * 70)

    # Test 1: Basic seeding
    test_set_seed()

    # Test 2: Different seeds
    test_different_seeds()

    # Test 3: State save/restore
    test_rng_state_save_restore()

    # Test 4: Worker initialization
    test_worker_init_fn()

    # Test 5: Convenience function
    test_seed_everything()

    # Test 6: MPS availability
    test_mps_availability()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n💡 Tip: Use seed_everything(42) at the start of your training scripts")
    print("   for reproducible experiments!")
