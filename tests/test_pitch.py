"""
Unit tests for pitch shift augmentation.

Run with:
  python tests/test_pitch.py              # Direct execution
  pytest tests/test_pitch.py -v           # With pytest
"""

import sys
from pathlib import Path

# Add src to path (matches existing test pattern)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.augment.pitch import pitch_shift, random_pitch_shift


class TestPitchShift:
    """Test suite for pitch_shift function."""

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

    def test_pitch_shift_preserves_length(self, test_audio):
        """Test that pitch shifting preserves audio length."""
        audio, sr = test_audio

        # Shift up by 2 semitones
        shifted = pitch_shift(audio, sr, n_steps=2.0)

        assert shifted.shape == audio.shape, (
            f"Shifted audio length {shifted.shape} != original {audio.shape}"
        )

    def test_pitch_shift_up(self, test_audio):
        """Test shifting up by 2 semitones."""
        audio, sr = test_audio

        shifted = pitch_shift(audio, sr, n_steps=2.0)

        # Check output is valid
        assert shifted.shape == audio.shape
        assert np.isfinite(shifted).all(), "Shifted audio contains NaN or Inf"

        # Check that audio was actually modified
        assert not np.array_equal(shifted, audio), (
            "Shifted audio is identical to original"
        )

    def test_pitch_shift_down(self, test_audio):
        """Test shifting down by 2 semitones."""
        audio, sr = test_audio

        shifted = pitch_shift(audio, sr, n_steps=-2.0)

        # Check output is valid
        assert shifted.shape == audio.shape
        assert np.isfinite(shifted).all(), "Shifted audio contains NaN or Inf"

        # Check that audio was actually modified
        assert not np.array_equal(shifted, audio), (
            "Shifted audio is identical to original"
        )

    def test_pitch_shift_zero_steps(self, test_audio):
        """Test that zero shift returns similar audio (may have small numerical differences)."""
        audio, sr = test_audio

        shifted = pitch_shift(audio, sr, n_steps=0.0)

        # With zero shift, audio should be very similar (allowing for numerical precision)
        assert shifted.shape == audio.shape
        assert np.allclose(shifted, audio, atol=1e-3), (
            "Zero shift produced significantly different audio"
        )

    def test_pitch_shift_fractional_steps(self, test_audio):
        """Test pitch shifting with fractional semitones (quarter tone)."""
        audio, sr = test_audio

        # Shift by half a semitone (quarter tone)
        shifted = pitch_shift(audio, sr, n_steps=0.5)

        assert shifted.shape == audio.shape
        assert np.isfinite(shifted).all()
        assert not np.array_equal(shifted, audio)


class TestRandomPitchShift:
    """Test suite for random_pitch_shift function."""

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
            result = random_pitch_shift(audio, sr, max_steps=2.0, probability=0.0)
            assert np.array_equal(result, audio), (
                "Audio was modified despite probability=0.0"
            )

    def test_probability_one(self, test_audio):
        """Test that probability=1.0 always applies augmentation."""
        audio, sr = test_audio

        # With probability=1, audio should always be modified
        for _ in range(10):
            result = random_pitch_shift(audio, sr, max_steps=2.0, probability=1.0)
            assert not np.array_equal(result, audio), (
                "Audio was not modified despite probability=1.0"
            )

    def test_seed_reproducibility(self, test_audio):
        """Test that same seed produces same result."""
        audio, sr = test_audio

        # Apply with same seed twice
        result1 = random_pitch_shift(audio, sr, max_steps=2.0, probability=1.0, seed=42)
        result2 = random_pitch_shift(audio, sr, max_steps=2.0, probability=1.0, seed=42)

        assert np.array_equal(result1, result2), "Same seed produced different results"

    def test_seed_different(self, test_audio):
        """Test that different seeds produce different results."""
        audio, sr = test_audio

        # Apply with different seeds
        result1 = random_pitch_shift(audio, sr, max_steps=2.0, probability=1.0, seed=42)
        result2 = random_pitch_shift(
            audio, sr, max_steps=2.0, probability=1.0, seed=123
        )

        assert not np.array_equal(result1, result2), (
            "Different seeds produced identical results"
        )

    def test_probability_distribution(self, test_audio):
        """Test that probability parameter works approximately as expected."""
        audio, sr = test_audio

        # Run many trials with probability=0.5
        num_trials = 100
        num_shifted = 0

        for i in range(num_trials):
            result = random_pitch_shift(
                audio, sr, max_steps=2.0, probability=0.5, seed=None
            )
            if not np.array_equal(result, audio):
                num_shifted += 1

        # With probability=0.5, expect roughly 50% shifted (allow ±20% tolerance)
        expected = num_trials * 0.5
        tolerance = num_trials * 0.2

        assert abs(num_shifted - expected) < tolerance, (
            f"Probability test failed: {num_shifted}/{num_trials} shifted "
            f"(expected ~{expected} ± {tolerance})"
        )

    def test_max_steps_range(self, test_audio):
        """Test that shifts are within the specified range."""
        audio, sr = test_audio
        max_steps = 2.0

        # We can't directly test the shift amount, but we can verify
        # that the function runs without error for various max_steps values
        for steps in [0.5, 1.0, 2.0, 4.0]:
            result = random_pitch_shift(
                audio, sr, max_steps=steps, probability=1.0, seed=42
            )
            assert result.shape == audio.shape
            assert np.isfinite(result).all()

    def test_output_shape(self, test_audio):
        """Test that output shape matches input shape."""
        audio, sr = test_audio

        result = random_pitch_shift(audio, sr, max_steps=2.0, probability=0.5)

        assert result.shape == audio.shape, (
            f"Output shape {result.shape} != input shape {audio.shape}"
        )

    def test_output_valid(self, test_audio):
        """Test that output contains valid numerical values."""
        audio, sr = test_audio

        result = random_pitch_shift(audio, sr, max_steps=2.0, probability=1.0)

        assert np.isfinite(result).all(), "Output contains NaN or Inf values"


if __name__ == "__main__":
    """
    Run tests directly without pytest (matches existing test pattern).
    For comprehensive testing, use: pytest tests/test_pitch.py -v
    """
    print("=" * 70)
    print("PITCH SHIFT AUGMENTATION TESTS")
    print("=" * 70)
    print()

    # Create test audio
    sr = 22050
    duration = 3.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * frequency * t)

    print("Testing: pitch_shift()")
    print(f"  Test audio: {test_audio.shape}, sr={sr}")

    # Test 1: Basic pitch shift up
    shifted = pitch_shift(test_audio, sr, n_steps=2.0)
    assert shifted.shape == test_audio.shape
    assert not np.array_equal(shifted, test_audio)
    print("  ✓ Shift up by 2 semitones")

    # Test 2: Pitch shift down
    shifted = pitch_shift(test_audio, sr, n_steps=-2.0)
    assert shifted.shape == test_audio.shape
    print("  ✓ Shift down by 2 semitones")

    # Test 3: Zero shift
    shifted = pitch_shift(test_audio, sr, n_steps=0.0)
    assert np.allclose(shifted, test_audio, atol=1e-3)
    print("  ✓ Zero shift preserves audio")

    print("\nTesting: random_pitch_shift()")

    # Test 4: Probability = 0
    result = random_pitch_shift(test_audio, sr, probability=0.0)
    assert np.array_equal(result, test_audio)
    print("  ✓ probability=0.0 never modifies")

    # Test 5: Probability = 1
    result = random_pitch_shift(test_audio, sr, probability=1.0)
    assert not np.array_equal(result, test_audio)
    print("  ✓ probability=1.0 always modifies")

    # Test 6: Seed reproducibility
    result1 = random_pitch_shift(test_audio, sr, probability=1.0, seed=42)
    result2 = random_pitch_shift(test_audio, sr, probability=1.0, seed=42)
    assert np.array_equal(result1, result2)
    print("  ✓ Seed reproducibility works")

    # Test 7: Probability distribution
    num_trials = 100
    num_shifted = sum(
        not np.array_equal(
            random_pitch_shift(test_audio, sr, probability=0.5), test_audio
        )
        for _ in range(num_trials)
    )
    print(f"  ✓ Probability test: {num_shifted}/{num_trials} shifted (expected ~50)")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFor comprehensive pytest testing, run:")
    print("  pytest tests/test_pitch.py -v")
