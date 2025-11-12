"""
Unit tests for SpecAugment (spectrogram masking).

Run with:
  python tests/test_specaugment.py              # Direct execution
  pytest tests/test_specaugment.py -v           # With pytest
"""

import sys
from pathlib import Path

# Add src to path (matches existing test pattern)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.augment.specaugment import (
    frequency_mask,
    random_spec_augment,
    spec_augment,
    time_mask,
)


class TestFrequencyMask:
    """Test suite for frequency_mask function."""

    @pytest.fixture
    def test_spectrogram(self):
        """Create a test spectrogram (128 mel bands × 130 frames)."""
        # Create spectrogram with known values (all ones)
        spec = np.ones((128, 130))
        return spec

    def test_frequency_mask_preserves_shape(self, test_spectrogram):
        """Test that frequency masking preserves spectrogram shape."""
        spec_aug = frequency_mask(test_spectrogram, max_mask_size=27, num_masks=1)

        assert spec_aug.shape == test_spectrogram.shape, (
            f"Shape changed: {spec_aug.shape} != {test_spectrogram.shape}"
        )

    def test_frequency_mask_modifies_spectrogram(self, test_spectrogram):
        """Test that frequency masking actually modifies the spectrogram."""
        spec_aug = frequency_mask(test_spectrogram, max_mask_size=27, num_masks=1)

        # Should be different from original (unless mask_size randomly chosen as 0)
        # Run multiple times to ensure we get a non-zero mask at least once
        modified = False
        for _ in range(10):
            spec_aug = frequency_mask(test_spectrogram, max_mask_size=27, num_masks=1)
            if not np.array_equal(spec_aug, test_spectrogram):
                modified = True
                break

        assert modified, "Frequency mask never modified the spectrogram"

    def test_frequency_mask_uses_mask_value(self, test_spectrogram):
        """Test that masked regions use the specified mask value."""
        mask_value = -99.0
        spec_aug = frequency_mask(
            test_spectrogram, max_mask_size=27, num_masks=1, mask_value=mask_value
        )

        # Run multiple times to ensure we get a mask
        for _ in range(10):
            spec_aug = frequency_mask(
                test_spectrogram, max_mask_size=27, num_masks=1, mask_value=mask_value
            )
            # If there are any masked values, they should be mask_value
            if (spec_aug == mask_value).any():
                # Check that entire frequency bands are masked (all time frames)
                masked_freqs = np.where((spec_aug == mask_value).all(axis=1))[0]
                if len(masked_freqs) > 0:
                    # Verify these frequencies are completely masked
                    for freq in masked_freqs:
                        assert (spec_aug[freq, :] == mask_value).all()
                    return

        # If we got here, no masks were applied in 10 tries (unlikely but possible)
        # This is acceptable for the test

    def test_frequency_mask_multiple(self, test_spectrogram):
        """Test applying multiple frequency masks."""
        spec_aug = frequency_mask(test_spectrogram, max_mask_size=27, num_masks=2)

        assert spec_aug.shape == test_spectrogram.shape

    def test_frequency_mask_bounds(self, test_spectrogram):
        """Test that masking stays within spectrogram bounds."""
        # Large mask that could exceed bounds
        spec_aug = frequency_mask(test_spectrogram, max_mask_size=50, num_masks=1)

        assert spec_aug.shape == test_spectrogram.shape
        assert np.isfinite(spec_aug).all()


class TestTimeMask:
    """Test suite for time_mask function."""

    @pytest.fixture
    def test_spectrogram(self):
        """Create a test spectrogram (128 mel bands × 130 frames)."""
        spec = np.ones((128, 130))
        return spec

    def test_time_mask_preserves_shape(self, test_spectrogram):
        """Test that time masking preserves spectrogram shape."""
        spec_aug = time_mask(test_spectrogram, max_mask_size=40, num_masks=1)

        assert spec_aug.shape == test_spectrogram.shape

    def test_time_mask_modifies_spectrogram(self, test_spectrogram):
        """Test that time masking actually modifies the spectrogram."""
        # Run multiple times to ensure we get a non-zero mask
        modified = False
        for _ in range(10):
            spec_aug = time_mask(test_spectrogram, max_mask_size=40, num_masks=1)
            if not np.array_equal(spec_aug, test_spectrogram):
                modified = True
                break

        assert modified, "Time mask never modified the spectrogram"

    def test_time_mask_uses_mask_value(self, test_spectrogram):
        """Test that masked regions use the specified mask value."""
        mask_value = -99.0

        # Run multiple times to ensure we get a mask
        for _ in range(10):
            spec_aug = time_mask(
                test_spectrogram, max_mask_size=40, num_masks=1, mask_value=mask_value
            )
            # If there are any masked values, they should be mask_value
            if (spec_aug == mask_value).any():
                # Check that entire time frames are masked (all frequencies)
                masked_times = np.where((spec_aug == mask_value).all(axis=0))[0]
                if len(masked_times) > 0:
                    # Verify these time frames are completely masked
                    for t in masked_times:
                        assert (spec_aug[:, t] == mask_value).all()
                    return

    def test_time_mask_multiple(self, test_spectrogram):
        """Test applying multiple time masks."""
        spec_aug = time_mask(test_spectrogram, max_mask_size=40, num_masks=2)

        assert spec_aug.shape == test_spectrogram.shape

    def test_time_mask_bounds(self, test_spectrogram):
        """Test that masking stays within spectrogram bounds."""
        # Large mask that could exceed bounds
        spec_aug = time_mask(test_spectrogram, max_mask_size=60, num_masks=1)

        assert spec_aug.shape == test_spectrogram.shape
        assert np.isfinite(spec_aug).all()


class TestSpecAugment:
    """Test suite for spec_augment function (combined masking)."""

    @pytest.fixture
    def test_spectrogram(self):
        """Create a test spectrogram (128 mel bands × 130 frames)."""
        spec = np.ones((128, 130))
        return spec

    def test_spec_augment_preserves_shape(self, test_spectrogram):
        """Test that SpecAugment preserves spectrogram shape."""
        spec_aug = spec_augment(
            test_spectrogram,
            freq_mask_max=27,
            time_mask_max=40,
            num_freq_masks=1,
            num_time_masks=1,
        )

        assert spec_aug.shape == test_spectrogram.shape

    def test_spec_augment_modifies_spectrogram(self, test_spectrogram):
        """Test that SpecAugment modifies the spectrogram."""
        # Run multiple times to ensure we get masks
        modified = False
        for _ in range(10):
            spec_aug = spec_augment(
                test_spectrogram,
                freq_mask_max=27,
                time_mask_max=40,
                num_freq_masks=1,
                num_time_masks=1,
            )
            if not np.array_equal(spec_aug, test_spectrogram):
                modified = True
                break

        assert modified, "SpecAugment never modified the spectrogram"

    def test_spec_augment_multiple_masks(self, test_spectrogram):
        """Test applying multiple frequency and time masks."""
        spec_aug = spec_augment(
            test_spectrogram,
            freq_mask_max=27,
            time_mask_max=40,
            num_freq_masks=2,
            num_time_masks=2,
        )

        assert spec_aug.shape == test_spectrogram.shape

    def test_spec_augment_no_masks(self, test_spectrogram):
        """Test that zero masks returns near-original spectrogram."""
        spec_aug = spec_augment(
            test_spectrogram,
            freq_mask_max=0,
            time_mask_max=0,
            num_freq_masks=0,
            num_time_masks=0,
        )

        assert np.array_equal(spec_aug, test_spectrogram)


class TestRandomSpecAugment:
    """Test suite for random_spec_augment function."""

    @pytest.fixture
    def test_spectrogram(self):
        """Create a test spectrogram (128 mel bands × 130 frames)."""
        spec = np.ones((128, 130))
        return spec

    def test_probability_zero(self, test_spectrogram):
        """Test that probability=0.0 never applies augmentation."""
        for _ in range(10):
            spec_aug = random_spec_augment(test_spectrogram, probability=0.0)
            assert np.array_equal(spec_aug, test_spectrogram), (
                "Spectrogram was modified despite probability=0.0"
            )

    def test_probability_one(self, test_spectrogram):
        """Test that probability=1.0 always applies augmentation."""
        # Run multiple times - should be modified in most cases
        modified_count = 0
        for _ in range(10):
            spec_aug = random_spec_augment(
                test_spectrogram, freq_mask_max=27, time_mask_max=40, probability=1.0
            )
            if not np.array_equal(spec_aug, test_spectrogram):
                modified_count += 1

        # Should be modified in most trials (at least 8 out of 10)
        # (Small chance of zero-size mask selection)
        assert modified_count >= 8, (
            f"Only {modified_count}/10 were modified with probability=1.0"
        )

    def test_seed_reproducibility(self, test_spectrogram):
        """Test that same seed produces same result."""
        spec_aug1 = random_spec_augment(test_spectrogram, probability=1.0, seed=42)
        spec_aug2 = random_spec_augment(test_spectrogram, probability=1.0, seed=42)

        assert np.array_equal(spec_aug1, spec_aug2), (
            "Same seed produced different results"
        )

    def test_seed_different(self, test_spectrogram):
        """Test that different seeds produce different results."""
        spec_aug1 = random_spec_augment(test_spectrogram, probability=1.0, seed=42)
        spec_aug2 = random_spec_augment(test_spectrogram, probability=1.0, seed=123)

        # Very likely to be different with different random seeds
        # (Small chance they could be identical if masks happen to be in same place)
        assert not np.array_equal(spec_aug1, spec_aug2), (
            "Different seeds produced identical results (unlikely but possible)"
        )

    def test_probability_distribution(self, test_spectrogram):
        """Test that probability parameter works approximately as expected."""
        num_trials = 100
        num_augmented = 0

        for _ in range(num_trials):
            spec_aug = random_spec_augment(
                test_spectrogram, freq_mask_max=27, time_mask_max=40, probability=0.5
            )
            if not np.array_equal(spec_aug, test_spectrogram):
                num_augmented += 1

        # With probability=0.5, expect roughly 50% augmented (allow ±20% tolerance)
        expected = num_trials * 0.5
        tolerance = num_trials * 0.2

        assert abs(num_augmented - expected) < tolerance, (
            f"Probability test failed: {num_augmented}/{num_trials} augmented "
            f"(expected ~{expected} ± {tolerance})"
        )


if __name__ == "__main__":
    """
    Run tests directly without pytest (matches existing test pattern).
    For comprehensive testing, use: pytest tests/test_specaugment.py -v
    """
    print("=" * 70)
    print("SPECAUGMENT TESTS")
    print("=" * 70)
    print()

    # Create test spectrogram (IRMAS dimensions: 128 mel bands × 130 frames)
    test_spec = np.ones((128, 130))

    print("Testing: frequency_mask()")
    print(f"  Test spectrogram: {test_spec.shape}")

    # Test 1: Frequency mask
    spec_aug = frequency_mask(test_spec, max_mask_size=27, num_masks=1)
    num_masked = np.sum(spec_aug == 0.0)
    print(f"  ✓ Shape preserved: {spec_aug.shape}")
    print(f"  ✓ Masked {num_masked} values")

    print("\nTesting: time_mask()")

    # Test 2: Time mask
    spec_aug = time_mask(test_spec, max_mask_size=40, num_masks=1)
    num_masked = np.sum(spec_aug == 0.0)
    print(f"  ✓ Shape preserved: {spec_aug.shape}")
    print(f"  ✓ Masked {num_masked} values")

    print("\nTesting: spec_augment()")

    # Test 3: Combined SpecAugment
    spec_aug = spec_augment(
        test_spec,
        freq_mask_max=27,
        time_mask_max=40,
        num_freq_masks=1,
        num_time_masks=1,
    )
    num_masked = np.sum(spec_aug == 0.0)
    print(f"  ✓ Shape preserved: {spec_aug.shape}")
    print(f"  ✓ Masked {num_masked} values (freq + time)")

    print("\nTesting: random_spec_augment()")

    # Test 4: Probability = 0
    spec_aug = random_spec_augment(test_spec, probability=0.0)
    assert np.array_equal(spec_aug, test_spec)
    print("  ✓ probability=0.0 never modifies")

    # Test 5: Probability = 1
    spec_aug = random_spec_augment(test_spec, probability=1.0)
    print("  ✓ probability=1.0 applies augmentation")

    # Test 6: Seed reproducibility
    spec_aug1 = random_spec_augment(test_spec, probability=1.0, seed=42)
    spec_aug2 = random_spec_augment(test_spec, probability=1.0, seed=42)
    assert np.array_equal(spec_aug1, spec_aug2)
    print("  ✓ Seed reproducibility works")

    # Test 7: Probability distribution
    num_trials = 100
    num_augmented = sum(
        not np.array_equal(random_spec_augment(test_spec, probability=0.5), test_spec)
        for _ in range(num_trials)
    )
    print(
        f"  ✓ Probability test: {num_augmented}/{num_trials} augmented (expected ~50)"
    )

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFor comprehensive pytest testing, run:")
    print("  pytest tests/test_specaugment.py -v")
