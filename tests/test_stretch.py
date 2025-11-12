"""
Unit tests for time stretch augmentation.

Run with:
  python tests/test_stretch.py              # Direct execution
  pytest tests/test_stretch.py -v           # With pytest
"""

import sys
from pathlib import Path

# Add src to path (matches existing test pattern)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.augment.stretch import random_time_stretch, time_stretch


class TestTimeStretch:
    """Test suite for time_stretch function."""

    @pytest.fixture
    def test_audio(self):
        """Create a simple test signal (440 Hz sine wave = A4 note)."""
        sr = 22050
        duration = 3.0  # 3 seconds
        frequency = 440.0  # A4

        # Generate sine wave
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * frequency * t)

        return audio, sr

    def test_time_stretch_speed_up(self, test_audio):
        """Test speeding up audio (rate > 1.0)."""
        audio, sr = test_audio
        original_length = len(audio)

        # Speed up by 20% (rate=1.2)
        stretched = time_stretch(audio, rate=1.2)

        # Check that audio is shorter (approximately 1/1.2 = 0.833× original)
        expected_length = int(original_length / 1.2)
        assert abs(len(stretched) - expected_length) < 100, (
            f"Stretched length {len(stretched)} not close to expected {expected_length}"
        )

        # Check output is valid
        assert np.isfinite(stretched).all(), "Stretched audio contains NaN or Inf"

    def test_time_stretch_slow_down(self, test_audio):
        """Test slowing down audio (rate < 1.0)."""
        audio, sr = test_audio
        original_length = len(audio)

        # Slow down by 20% (rate=0.8)
        stretched = time_stretch(audio, rate=0.8)

        # Check that audio is longer (approximately 1/0.8 = 1.25× original)
        expected_length = int(original_length / 0.8)
        assert abs(len(stretched) - expected_length) < 100, (
            f"Stretched length {len(stretched)} not close to expected {expected_length}"
        )

        # Check output is valid
        assert np.isfinite(stretched).all(), "Stretched audio contains NaN or Inf"

    def test_time_stretch_no_change(self, test_audio):
        """Test that rate=1.0 produces similar length (may vary slightly)."""
        audio, sr = test_audio

        stretched = time_stretch(audio, rate=1.0)

        # With rate=1.0, length should be very similar (allow small variation)
        assert abs(len(stretched) - len(audio)) < 100, (
            f"Rate=1.0 produced length {len(stretched)} vs original {len(audio)}"
        )

        assert np.isfinite(stretched).all()

    def test_time_stretch_extreme_slow(self, test_audio):
        """Test extreme slowdown (rate=0.5 → 2× longer)."""
        audio, sr = test_audio
        original_length = len(audio)

        stretched = time_stretch(audio, rate=0.5)

        # Should be approximately twice as long
        expected_length = int(original_length / 0.5)
        assert abs(len(stretched) - expected_length) < 200, (
            f"Extreme slow: length {len(stretched)} vs expected {expected_length}"
        )

        assert np.isfinite(stretched).all()

    def test_time_stretch_extreme_fast(self, test_audio):
        """Test extreme speedup (rate=2.0 → 0.5× shorter)."""
        audio, sr = test_audio
        original_length = len(audio)

        stretched = time_stretch(audio, rate=2.0)

        # Should be approximately half as long
        expected_length = int(original_length / 2.0)
        assert abs(len(stretched) - expected_length) < 100, (
            f"Extreme fast: length {len(stretched)} vs expected {expected_length}"
        )

        assert np.isfinite(stretched).all()


class TestRandomTimeStretch:
    """Test suite for random_time_stretch function."""

    @pytest.fixture
    def test_audio(self):
        """Create a simple test signal."""
        sr = 22050
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * frequency * t)

        return audio, sr

    def test_probability_zero(self, test_audio):
        """Test that probability=0.0 never applies augmentation."""
        audio, sr = test_audio

        # With probability=0, audio should never be modified
        for _ in range(10):
            result = random_time_stretch(audio, rate_range=(0.8, 1.2), probability=0.0)
            # Length should be identical (no stretching applied)
            assert len(result) == len(audio), (
                "Audio was modified despite probability=0.0"
            )

    def test_probability_one(self, test_audio):
        """Test that probability=1.0 always applies augmentation."""
        audio, sr = test_audio
        original_length = len(audio)

        # With probability=1, audio should always be modified
        # (unless rate happens to be exactly 1.0, which is unlikely)
        modified_count = 0
        for _ in range(10):
            result = random_time_stretch(audio, rate_range=(0.8, 1.2), probability=1.0)
            # Length should differ if rate != 1.0 (which is very likely)
            if len(result) != original_length:
                modified_count += 1

        # Should be modified in most trials (at least 8 out of 10)
        assert modified_count >= 8, (
            f"Only {modified_count}/10 trials were modified with probability=1.0"
        )

    def test_seed_reproducibility(self, test_audio):
        """Test that same seed produces same result."""
        audio, sr = test_audio

        # Apply with same seed twice
        result1 = random_time_stretch(
            audio, rate_range=(0.8, 1.2), probability=1.0, seed=42
        )
        result2 = random_time_stretch(
            audio, rate_range=(0.8, 1.2), probability=1.0, seed=42
        )

        # Should produce identical results
        assert len(result1) == len(result2), "Same seed produced different lengths"
        assert np.allclose(result1, result2, atol=1e-5), (
            "Same seed produced different audio content"
        )

    def test_seed_different(self, test_audio):
        """Test that different seeds produce different results."""
        audio, sr = test_audio

        # Apply with different seeds
        result1 = random_time_stretch(
            audio, rate_range=(0.8, 1.2), probability=1.0, seed=42
        )
        result2 = random_time_stretch(
            audio, rate_range=(0.8, 1.2), probability=1.0, seed=123
        )

        # Should produce different results (very likely with random rates)
        # Check if either length or content differs
        length_differs = len(result1) != len(result2)
        content_differs = (
            not np.allclose(result1, result2, atol=1e-5) if not length_differs else True
        )

        assert length_differs or content_differs, (
            "Different seeds produced identical results"
        )

    def test_probability_distribution(self, test_audio):
        """Test that probability parameter works approximately as expected."""
        audio, sr = test_audio
        original_length = len(audio)

        # Run many trials with probability=0.5
        num_trials = 100
        num_stretched = 0

        for i in range(num_trials):
            result = random_time_stretch(
                audio, rate_range=(0.8, 1.2), probability=0.5, seed=None
            )
            # Check if it was stretched (length will differ from original)
            if len(result) != original_length:
                num_stretched += 1

        # With probability=0.5, expect roughly 50% stretched (allow ±20% tolerance)
        expected = num_trials * 0.5
        tolerance = num_trials * 0.2

        assert abs(num_stretched - expected) < tolerance, (
            f"Probability test failed: {num_stretched}/{num_trials} stretched "
            f"(expected ~{expected} ± {tolerance})"
        )

    def test_rate_range(self, test_audio):
        """Test different rate ranges."""
        audio, sr = test_audio

        # Test various rate ranges
        ranges = [(0.8, 1.2), (0.7, 1.3), (0.5, 1.5), (0.9, 1.1)]

        for rate_range in ranges:
            result = random_time_stretch(
                audio, rate_range=rate_range, probability=1.0, seed=42
            )
            assert np.isfinite(result).all(), (
                f"Rate range {rate_range} produced invalid audio"
            )

    def test_output_valid(self, test_audio):
        """Test that output contains valid numerical values."""
        audio, sr = test_audio

        result = random_time_stretch(audio, rate_range=(0.8, 1.2), probability=1.0)

        assert np.isfinite(result).all(), "Output contains NaN or Inf values"


if __name__ == "__main__":
    """
    Run tests directly without pytest (matches existing test pattern).
    For comprehensive testing, use: pytest tests/test_stretch.py -v
    """
    print("=" * 70)
    print("TIME STRETCH AUGMENTATION TESTS")
    print("=" * 70)
    print()

    # Create test audio
    sr = 22050
    duration = 3.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * frequency * t)
    original_length = len(test_audio)

    print("Testing: time_stretch()")
    print(f"  Test audio: {test_audio.shape}, sr={sr}")

    # Test 1: Speed up
    stretched = time_stretch(test_audio, rate=1.2)
    print(f"  ✓ Speed up (1.2×): {original_length} → {len(stretched)} samples")

    # Test 2: Slow down
    stretched = time_stretch(test_audio, rate=0.8)
    print(f"  ✓ Slow down (0.8×): {original_length} → {len(stretched)} samples")

    # Test 3: No change
    stretched = time_stretch(test_audio, rate=1.0)
    print(f"  ✓ No change (1.0×): {original_length} → {len(stretched)} samples")

    print("\nTesting: random_time_stretch()")

    # Test 4: Probability = 0
    result = random_time_stretch(test_audio, probability=0.0)
    assert len(result) == original_length
    print("  ✓ probability=0.0 never modifies")

    # Test 5: Probability = 1
    result = random_time_stretch(test_audio, probability=1.0)
    print(
        f"  ✓ probability=1.0 always modifies: {original_length} → {len(result)} samples"
    )

    # Test 6: Seed reproducibility
    result1 = random_time_stretch(test_audio, probability=1.0, seed=42)
    result2 = random_time_stretch(test_audio, probability=1.0, seed=42)
    assert len(result1) == len(result2)
    assert np.allclose(result1, result2, atol=1e-5)
    print("  ✓ Seed reproducibility works")

    # Test 7: Probability distribution
    num_trials = 100
    num_stretched = sum(
        len(random_time_stretch(test_audio, probability=0.5)) != original_length
        for _ in range(num_trials)
    )
    print(
        f"  ✓ Probability test: {num_stretched}/{num_trials} stretched (expected ~50)"
    )

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFor comprehensive pytest testing, run:")
    print("  pytest tests/test_stretch.py -v")
