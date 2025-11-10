"""Test mel-spectrogram feature extraction."""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.paths import get_data_dir
from src.features.extraction import (
    extract_melspectrogram,
    extract_melspectrogram_from_file,
    get_melspectrogram_shape,
    load_melspectrogram,
    normalize_melspectrogram,
    save_melspectrogram,
    visualize_melspectrogram,
)


def test_extract_melspectrogram():
    """Test mel-spectrogram extraction from waveform."""
    print("Testing: extract_melspectrogram()")

    # Create synthetic audio (1 second at 22050 Hz)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    waveform = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave (A4 note)

    # Extract mel-spectrogram
    melspec = extract_melspectrogram(waveform, sr, n_mels=128)

    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Mel-spectrogram shape: {melspec.shape}")
    print(f"  Value range: [{melspec.min():.2f}, {melspec.max():.2f}] dB")

    # Verify shape
    assert len(melspec.shape) == 2, "Should be 2D array"
    assert melspec.shape[0] == 128, "Should have 128 mel bands"
    assert melspec.shape[1] > 0, "Should have time frames"

    print("  ✓ Mel-spectrogram extraction works")


def test_extract_from_real_file():
    """Test extraction from actual IRMAS file."""
    print("\nTesting: extract_melspectrogram_from_file() with real data")

    # Find a real IRMAS file
    train_dir = get_data_dir("raw") / "IRMAS-TrainingData" / "pia"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    wav_files = list(train_dir.glob("*.wav"))
    if not wav_files:
        print("  ⚠️  No WAV files found")
        return False

    test_file = wav_files[0]
    print(f"  Testing with: {test_file.name}")

    # Extract mel-spectrogram
    melspec, sr = extract_melspectrogram_from_file(
        test_file, target_sr=22050, n_mels=128, hop_length=512
    )

    print(f"  Mel-spectrogram shape: {melspec.shape}")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Duration: ~{melspec.shape[1] * 512 / sr:.2f}s")
    print(f"  Value range: [{melspec.min():.2f}, {melspec.max():.2f}] dB")

    # Verify shape for 3-second audio
    assert melspec.shape[0] == 128, "Should have 128 mel bands"

    # For 3 seconds at 22050 Hz with hop_length=512:
    # ~130 frames expected
    expected_frames = int(np.ceil(3.0 * sr / 512))
    assert abs(melspec.shape[1] - expected_frames) < 5, (
        f"Expected ~{expected_frames} frames"
    )

    print("  ✓ Real file extraction works correctly")
    return True


def test_save_and_load():
    """Test saving and loading mel-spectrograms."""
    print("\nTesting: save_melspectrogram() and load_melspectrogram()")

    # Create test mel-spectrogram
    melspec_original = np.random.randn(128, 130)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_melspec.npz"

        # Save
        save_melspectrogram(melspec_original, save_path)
        print(f"  Saved to: {save_path}")

        # Verify file exists
        assert save_path.exists(), "File should exist"

        # Load
        melspec_loaded = load_melspectrogram(save_path)
        print(f"  Loaded shape: {melspec_loaded.shape}")

        # Verify they match
        assert np.allclose(melspec_original, melspec_loaded), "Arrays should match"

        print("  ✓ Save and load works correctly")


def test_expected_shape():
    """Test mel-spectrogram shape calculation."""
    print("\nTesting: get_melspectrogram_shape()")

    # Test for 3-second IRMAS samples
    shape = get_melspectrogram_shape(
        duration=3.0, sample_rate=22050, hop_length=512, n_mels=128
    )

    print(f"  Expected shape for 3s audio: {shape}")

    assert shape[0] == 128, "Should have 128 mel bands"
    assert 125 < shape[1] < 135, f"Expected ~130 frames, got {shape[1]}"

    print("  ✓ Shape calculation correct")


def test_normalization():
    """Test mel-spectrogram normalization."""
    print("\nTesting: normalize_melspectrogram()")

    # Create test spectrogram with known range
    melspec = np.random.randn(128, 130) * 10 + 5  # Mean ~5, range varies

    print(f"  Original - min: {melspec.min():.2f}, max: {melspec.max():.2f}")

    # Test min-max normalization
    melspec_minmax = normalize_melspectrogram(melspec, method="min_max")

    print(
        f"  Min-max - min: {melspec_minmax.min():.2f}, max: {melspec_minmax.max():.2f}"
    )

    assert abs(melspec_minmax.min() - 0.0) < 0.01, "Min should be 0"
    assert abs(melspec_minmax.max() - 1.0) < 0.01, "Max should be 1"

    print("  ✓ Min-max normalization works")

    # Test standard normalization
    melspec_standard = normalize_melspectrogram(melspec, method="standard")

    mean = melspec_standard.mean()
    std = melspec_standard.std()

    print(f"  Standard - mean: {mean:.2f}, std: {std:.2f}")

    assert abs(mean) < 0.01, "Mean should be ~0"
    assert abs(std - 1.0) < 0.1, "Std should be ~1"

    print("  ✓ Standard normalization works")


def test_visualization():
    """Test mel-spectrogram visualization."""
    print("\nTesting: visualize_melspectrogram()")

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData" / "pia"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    wav_files = list(train_dir.glob("*.wav"))
    if not wav_files:
        return False

    test_file = wav_files[0]

    # Extract mel-spectrogram
    melspec, sr = extract_melspectrogram_from_file(test_file)

    # Save visualization to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "melspec_plot.png"

        visualize_melspectrogram(
            melspec,
            sr,
            hop_length=512,
            title=f"Piano - {test_file.name}",
            save_path=save_path,
        )

        # Verify file was created
        assert save_path.exists(), "Plot should be saved"

        print(f"  ✓ Visualization saved to: {save_path}")

    return True


def test_batch_extraction():
    """Test batch extraction from multiple files."""
    print("\nTesting: extract_features_batch()")

    from src.features.extraction import extract_features_batch

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData" / "pia"

    if not train_dir.exists():
        print("  ⚠️  IRMAS training data not found, skipping")
        return False

    # Get first 5 files
    wav_files = list(train_dir.glob("*.wav"))[:5]

    if len(wav_files) < 5:
        print("  ⚠️  Not enough files found")
        return False

    print(f"  Extracting features from {len(wav_files)} files...")

    melspecs = extract_features_batch(
        wav_files, target_sr=22050, n_mels=128, show_progress=True
    )

    print(f"  ✓ Extracted {len(melspecs)} mel-spectrograms")

    # Verify all succeeded
    assert all(m is not None for m in melspecs), "All extractions should succeed"

    # Verify shapes
    for i, melspec in enumerate(melspecs):
        print(f"    File {i + 1}: shape {melspec.shape}")
        assert melspec.shape[0] == 128, "Should have 128 mel bands"

    print("  ✓ Batch extraction works")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE EXTRACTION TESTS")
    print("=" * 70)

    # Test 1: Basic extraction (synthetic audio)
    test_extract_melspectrogram()

    # Test 2: Extract from real IRMAS file
    has_data = test_extract_from_real_file()

    # Test 3: Save and load
    test_save_and_load()

    # Test 4: Shape calculation
    test_expected_shape()

    # Test 5: Normalization
    test_normalization()

    if has_data:
        # Test 6: Visualization
        test_visualization()

        # Test 7: Batch extraction
        test_batch_extraction()

    print("\n" + "=" * 70)
    print("✅ ALL FEATURE EXTRACTION TESTS PASSED!")
    print("=" * 70)

    if has_data:
        print("\n🎵→📊 Feature extraction module is ready!")
        print("   Successfully tested with real IRMAS files.")
        print("   Mel-spectrograms: 128 mel bands × ~130 time frames")
    else:
        print("\n⚠️  Note: Some tests skipped (IRMAS data not found)")
        print("   But core feature extraction works correctly!")
