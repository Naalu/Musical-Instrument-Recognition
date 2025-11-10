"""Verify IRMAS dataset is properly downloaded and structured."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.paths import get_data_dir
from src.data.irmas import (
    IRMAS_CLASSES,
    get_class_distribution,
    index_test_data,
    index_training_data,
)


def verify_training_data():
    """Verify training data structure and content."""
    print("=" * 70)
    print("VERIFYING TRAINING DATA")
    print("=" * 70)

    train_dir = get_data_dir("raw") / "IRMAS-TrainingData"

    if not train_dir.exists():
        print(f"❌ Training data not found at: {train_dir}")
        print("\nPlease download and extract IRMAS-TrainingData.zip to data/raw/")
        return False

    print(f"✓ Training directory exists: {train_dir}")

    # Check all instrument directories exist
    print("\nChecking instrument directories:")
    missing = []
    for instrument in IRMAS_CLASSES:
        inst_dir = train_dir / instrument
        if inst_dir.exists():
            wav_count = len(list(inst_dir.glob("*.wav")))
            print(f"  ✓ {instrument}/  ({wav_count} .wav files)")
        else:
            print(f"  ❌ {instrument}/  (missing)")
            missing.append(instrument)

    if missing:
        print(f"\n❌ Missing directories: {missing}")
        return False

    # Index the data
    print("\nIndexing training data...")
    df = index_training_data(train_dir)

    print(f"✓ Indexed {len(df)} training samples")

    # Show class distribution
    print("\nClass distribution:")
    dist = get_class_distribution(df, test=False)
    for instrument, count in dist.items():
        print(f"  {instrument}: {count:4d} samples")

    print(f"\nTotal: {dist.sum()} samples")

    # Expected total: ~6705
    if dist.sum() < 6000:
        print("⚠️  Warning: Expected ~6705 samples, got fewer. Check extraction.")
    else:
        print("✓ Sample count looks good!")

    return True


def verify_test_data():
    """Verify test data structure and content."""
    print("\n" + "=" * 70)
    print("VERIFYING TEST DATA")
    print("=" * 70)

    data_raw = get_data_dir("raw")

    # Check for all three parts
    parts_found = []
    total_wav = 0
    total_txt = 0

    for part_num in [1, 2, 3]:
        part_dir = data_raw / f"IRMAS-TestingData-Part{part_num}"

        if part_dir.exists():
            parts_found.append(part_num)
            print(f"✓ Part {part_num} exists: {part_dir}")

            # Look for nested directories with .wav files
            search_dir = None
            for subdir in part_dir.iterdir():
                if subdir.is_dir():
                    wav_count = len(list(subdir.glob("*.wav")))
                    if wav_count > 0:
                        search_dir = subdir
                        print(f"  (nested in {subdir.name}/ subdirectory)")
                        break

            # If no nested directory found, use parent
            if search_dir is None:
                search_dir = part_dir

            # Count files in the correct directory
            wav_files = list(search_dir.glob("*.wav"))
            txt_files = list(search_dir.glob("*.txt"))

            print(f"  .wav files: {len(wav_files)}")
            print(f"  .txt files: {len(txt_files)}")

            total_wav += len(wav_files)
            total_txt += len(txt_files)
        else:
            print(f"❌ Part {part_num} not found: {part_dir}")

    if not parts_found:
        print("\n❌ No test data parts found!")
        print("Please download IRMAS-TestingData-Part1/2/3.zip")
        return False

    print("\nTotal across all parts:")
    print(f"  .wav files: {total_wav}")
    print(f"  .txt files: {total_txt}")

    if total_wav != total_txt:
        print("⚠️  Warning: Mismatched .wav and .txt counts")

    # Index all test data
    print("\nIndexing all test data parts...")
    df = index_test_data(data_raw)

    print(f"✓ Indexed {len(df)} test samples")

    # Show distribution across parts
    print("\nSamples per part:")
    for part_num in sorted(df["part"].unique()):
        count = (df["part"] == part_num).sum()
        print(f"  Part {part_num}: {count} samples")

    # Show some examples
    print("\nSample test files (multi-label):")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        labels_str = ", ".join(row["labels"])
        print(f"  Part{row['part']} - {row['filename']:30s} → {labels_str}")

    # Expected total: ~2874
    if len(df) < 2800:
        print(f"\n⚠️  Warning: Expected ~2874 samples, got {len(df)}.")
        if len(parts_found) < 3:
            print(f"    Missing parts: {set([1, 2, 3]) - set(parts_found)}")
            print("    Download missing parts from https://www.upf.edu/web/mtg/irmas")
    else:
        print("\n✓ Sample count looks good!")

    return len(df) >= 2800  # Pass if we have most of the data


if __name__ == "__main__":
    print("\n" + "🎵" * 35)
    print("IRMAS DATASET VERIFICATION")
    print("🎵" * 35 + "\n")

    train_ok = verify_training_data()
    test_ok = verify_test_data()

    print("\n" + "=" * 70)
    if train_ok and test_ok:
        print("✅ DATASET VERIFICATION PASSED!")
        print("=" * 70)
        print("\n🚀 Your IRMAS dataset is ready for training!")
    else:
        print("❌ DATASET VERIFICATION FAILED")
        print("=" * 70)
        print("\nPlease check the errors above and ensure:")
        print("1. Files are downloaded from https://www.upf.edu/web/mtg/irmas")
        print("2. Zip files are extracted to data/raw/")
        print("3. Directory structure matches expected format")
    print()
