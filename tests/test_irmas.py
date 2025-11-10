"""Test IRMAS dataset indexing."""

import sys
import tempfile
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.irmas import (
    IDX_TO_LABEL,
    IRMAS_CLASSES,
    LABEL_TO_IDX,
    get_class_distribution,
    index_test_data,
    index_training_data,
    parse_test_labels,
    parse_training_filename,
)


def test_constants():
    """Test IRMAS constants."""
    print("Testing: IRMAS constants")

    # Check we have 11 classes
    assert len(IRMAS_CLASSES) == 11, "Should have 11 instrument classes"
    print(f"  ✓ 11 instrument classes: {', '.join(IRMAS_CLASSES)}")

    # Check mappings are consistent
    assert len(LABEL_TO_IDX) == 11
    assert len(IDX_TO_LABEL) == 11
    print("  ✓ Label mappings consistent")

    # Check bidirectional mapping works
    for label, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == label
    print("  ✓ Bidirectional mapping works")


def test_parse_training_filename():
    """Test filename parsing logic."""
    print("\nTesting: parse_training_filename()")

    # Test case 1: Standard format
    path1 = Path("/fake/path/IRMAS-TrainingData/cla/010__[cla][cla][nod]2104__1.wav")
    meta1 = parse_training_filename(path1)

    assert meta1["instrument"] == "cla"
    assert meta1["artist_id"] == "010"
    assert meta1["track_id"] == "010__[cla][cla][nod]2104"
    print(f"  ✓ Parsed: {path1.name}")
    print(f"    Instrument: {meta1['instrument']}")
    print(f"    Artist: {meta1['artist_id']}")
    print(f"    Track: {meta1['track_id']}")

    # Test case 2: Different format
    path2 = Path("/fake/path/IRMAS-TrainingData/pia/123__piano_recording__5.wav")
    meta2 = parse_training_filename(path2)

    assert meta2["instrument"] == "pia"
    assert meta2["artist_id"] == "123"
    print("  ✓ Parsed different format successfully")


def test_parse_test_labels():
    """Test parsing test label files."""
    print("\nTesting: parse_test_labels()")

    # Create a temporary label file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("gel\n")
        f.write("pia\n")
        temp_file = Path(f.name)

    try:
        labels = parse_test_labels(temp_file)
        assert labels == ["gel", "pia"]
        print(f"  ✓ Parsed labels: {labels}")
    finally:
        temp_file.unlink()


def test_index_training_data_mock():
    """Test training data indexing with mock files."""
    print("\nTesting: index_training_data() with mock data")

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create instrument directories with mock files
        # Note: Only creating 3 of 11 classes for quick testing
        for instrument in ["cel", "cla", "flu"]:
            inst_dir = tmpdir / instrument
            inst_dir.mkdir()

            # Create 2 mock .wav files per instrument
            for i in range(2):
                wav_file = (
                    inst_dir / f"artist001__track_{instrument}_{i}__excerpt{i}.wav"
                )
                wav_file.touch()

        # Index the mock data (suppress expected warnings for missing directories)
        import contextlib
        import io

        # Capture warnings about missing directories (expected in mock test)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            df = index_training_data(tmpdir)

        # Count how many warnings (should be 8 missing directories)
        warnings_output = f.getvalue()
        warning_count = warnings_output.count("Warning:")
        print(
            f"  ✓ Suppressed {warning_count} expected warnings for missing directories"
        )

        print(f"  ✓ Indexed {len(df)} mock files")
        print(f"  ✓ Columns: {list(df.columns)}")

        # Verify structure
        assert len(df) == 6  # 3 instruments × 2 files
        assert "filepath" in df.columns
        assert "instrument" in df.columns
        assert "label_idx" in df.columns
        assert "artist_id" in df.columns
        assert "track_id" in df.columns

        print("  ✓ DataFrame structure correct")

        # Check instruments
        instruments = df["instrument"].unique()
        assert set(instruments) == {"cel", "cla", "flu"}
        print(f"  ✓ Found instruments: {sorted(instruments)}")

        # Check class distribution
        dist = get_class_distribution(df, test=False)
        print(f"  ✓ Class distribution:\n{dist}")


def test_index_test_data_mock():
    """Test test data indexing with mock files."""
    print("\nTesting: index_test_data() with mock data")

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock test files
        for i in range(3):
            # Create .wav file
            wav_file = tmpdir / f"{i + 1}.wav"
            wav_file.touch()

            # Create corresponding .txt label file
            txt_file = tmpdir / f"{i + 1}.txt"
            with open(txt_file, "w") as f:
                if i == 0:
                    f.write("gel\npia\n")  # Multi-label
                elif i == 1:
                    f.write("voi\n")  # Single label
                else:
                    f.write("cla\nsax\n")  # Multi-label

        # Index the mock data
        df = index_test_data(tmpdir)

        print(f"  ✓ Indexed {len(df)} mock test files")

        # Verify structure
        assert len(df) == 3
        assert "filepath" in df.columns
        assert "labels" in df.columns
        assert "label_indices" in df.columns

        print("  ✓ DataFrame structure correct")

        # Check multi-label
        assert len(df.iloc[0]["labels"]) == 2  # First file has 2 instruments
        assert len(df.iloc[1]["labels"]) == 1  # Second file has 1 instrument
        print("  ✓ Multi-label handling works")

        print(f"  Sample labels: {df.iloc[0]['labels']}")


if __name__ == "__main__":
    print("=" * 70)
    print("IRMAS INDEXER TESTS (Mock Data)")
    print("=" * 70)

    # Test 1: Constants
    test_constants()

    # Test 2: Filename parsing
    test_parse_training_filename()

    # Test 3: Test label parsing
    test_parse_test_labels()

    # Test 4: Training data indexing (mock)
    test_index_training_data_mock()

    # Test 5: Test data indexing (mock)
    test_index_test_data_mock()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED (Mock Data)!")
    print("=" * 70)
    print("\n💡 Run:")
    print("   python scripts/verify_dataset.py")
    print("   to test with real data!")
