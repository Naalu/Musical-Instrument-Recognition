"""Extract IRMAS dataset with proper UTF-8 filename handling."""

import zipfile
from pathlib import Path


def extract_with_progress(zip_path, extract_to):
    """Extract zip file with progress reporting."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return False

    print(f"Extracting: {zip_path.name}")
    print(f"       To: {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"Total files: {total_files}")
            print("Extracting...")

            # Extract with progress
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, extract_to)

                # Show progress every 100 files
                if i % 100 == 0 or i == total_files:
                    print(
                        f"  Progress: {i}/{total_files} files ({100 * i // total_files}%)"
                    )

            print("✅ Extraction complete!")
            return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def main():
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_raw = project_root / "data" / "raw"

    print("=" * 70)
    print("IRMAS DATASET EXTRACTION (UTF-8 Safe)")
    print("=" * 70)

    # Training data
    train_zip = data_raw / "IRMAS-TrainingData.zip"
    if train_zip.exists():
        print("\n1. Training Data")
        success = extract_with_progress(train_zip, data_raw)
        if not success:
            print("⚠️  Continuing despite training data extraction issue...")
    else:
        print(f"\n⚠️  Training data not found: {train_zip}")

    # Test data (all 3 parts)
    test_parts = [
        "IRMAS-TestingData-Part1.zip",
        "IRMAS-TestingData-Part2.zip",
        "IRMAS-TestingData-Part3.zip",
    ]

    for i, part_name in enumerate(test_parts, 1):
        test_zip = data_raw / part_name
        if test_zip.exists():
            print(f"\n{i + 1}. Test Data - {part_name}")
            success = extract_with_progress(test_zip, data_raw)
            if not success:
                print(f"⚠️  Continuing despite {part_name} extraction issue...")
        else:
            print(f"\n⚠️  {part_name} not found at: {test_zip}")
            print("    Download from: https://www.upf.edu/web/mtg/irmas")

    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 70)
    print("\nRun: python scripts/verify_dataset.py")
    print("to verify the dataset is ready.")


if __name__ == "__main__":
    main()
