"""IRMAS dataset indexing and metadata extraction.

Scans the IRMAS directory structure, parses filenames, and creates
a pandas DataFrame index for efficient data access and splitting.

Example:
    >>> from src.data.irmas import index_training_data
    >>> df = index_training_data('data/raw/IRMAS-TrainingData')
    >>> print(df.shape)
    (6705, 5)  # 6705 samples with 5 metadata columns
"""

import re
from pathlib import Path
from typing import List

import pandas as pd

# IRMAS instrument classes (alphabetical order)
IRMAS_CLASSES = [
    "cel",  # Cello
    "cla",  # Clarinet
    "flu",  # Flute
    "gac",  # Acoustic Guitar
    "gel",  # Electric Guitar
    "org",  # Organ
    "pia",  # Piano
    "sax",  # Saxophone
    "tru",  # Trumpet
    "vio",  # Violin
    "voi",  # Voice
]

# Create label-to-index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(IRMAS_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


def parse_training_filename(filepath: Path) -> dict:
    """Parse IRMAS training filename to extract metadata.

    IRMAS training filenames contain metadata about the recording:
    Format: [instrument]-[details]-[number].wav
    Example: 010__[cla][cla][nod]2104__1.wav

    The filename encodes:
    - Artist/performer identifier (for track-level splitting)
    - Instrument class
    - Recording session information

    Args:
        filepath: Path to audio file.

    Returns:
        Dictionary with metadata: instrument, artist_id, track_id, filename.

    Example:
        >>> path = Path('data/raw/IRMAS-TrainingData/cla/010__[cla][cla][nod]2104__1.wav')
        >>> meta = parse_training_filename(path)
        >>> meta['instrument']
        'cla'
        >>> meta['track_id']
        '010__[cla][cla][nod]2104'
    """
    filename = filepath.stem  # Filename without extension

    # Extract instrument from parent directory
    instrument = filepath.parent.name

    # Track ID: everything before the final '__' (if it exists)
    # This groups excerpts from the same original recording
    if "__" in filename:
        parts = filename.rsplit("__", 1)
        track_id = parts[0]
    else:
        track_id = filename

    # Artist ID: extract leading numeric/alphanumeric prefix
    # This helps identify different artists/performers
    match = re.match(r"^(\d+)", filename)
    if match:
        artist_id = match.group(1)
    else:
        # Fallback: use first part before brackets
        match = re.match(r"^([^\[]+)", filename)
        artist_id = match.group(1).strip("_") if match else "unknown"

    return {
        "instrument": instrument,
        "artist_id": artist_id,
        "track_id": track_id,
        "filename": filepath.name,
    }


def index_training_data(data_dir: str | Path) -> pd.DataFrame:
    """Create an index of all IRMAS training data.

    Scans the IRMAS-TrainingData directory, extracts metadata from filenames,
    and returns a pandas DataFrame with all samples.

    Args:
        data_dir: Path to IRMAS-TrainingData directory.

    Returns:
        DataFrame with columns:
        - filepath: absolute path to audio file
        - instrument: instrument class label
        - label_idx: integer label (0-10)
        - artist_id: artist/performer identifier
        - track_id: track identifier (for grouping excerpts)
        - filename: original filename

    Raises:
        FileNotFoundError: If data_dir doesn't exist.
        ValueError: If no audio files found.

    Example:
        >>> df = index_training_data('data/raw/IRMAS-TrainingData')
        >>> df.groupby('instrument').size()
        cel    388
        cla    505
        flu    451
        ...
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")

    # Collect all .wav files from subdirectories
    records = []

    for instrument in IRMAS_CLASSES:
        instrument_dir = data_dir / instrument

        if not instrument_dir.exists():
            print(f"Warning: Directory not found for {instrument}: {instrument_dir}")
            continue

        # Find all .wav files
        wav_files = sorted(instrument_dir.glob("*.wav"))

        for wav_file in wav_files:
            # Parse filename metadata
            metadata = parse_training_filename(wav_file)

            # Create record
            record = {
                "filepath": str(wav_file.resolve()),
                "instrument": metadata["instrument"],
                "label_idx": LABEL_TO_IDX[metadata["instrument"]],
                "artist_id": metadata["artist_id"],
                "track_id": metadata["track_id"],
                "filename": metadata["filename"],
            }

            records.append(record)

    if not records:
        raise ValueError(f"No audio files found in {data_dir}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort by instrument and filename for consistency
    df = df.sort_values(["instrument", "filename"]).reset_index(drop=True)

    return df


def parse_test_labels(label_file: Path) -> List[str]:
    """Parse IRMAS test label file.

    Test label files contain one instrument per line (tab-separated if multiple).
    Example content:
        gel
        pia

    This indicates both electric guitar and piano are present.

    Args:
        label_file: Path to .txt label file.

    Returns:
        List of instrument labels present in the audio.

    Example:
        >>> labels = parse_test_labels(Path('data/raw/IRMAS-TestingData-Part1/1.txt'))
        >>> labels
        ['gel', 'pia']
    """
    with open(label_file, "r") as f:
        lines = f.read().strip().split("\n")

    # Each line may contain one or more tab-separated instruments
    instruments = []
    for line in lines:
        line = line.strip()
        if line:
            # Split by tab in case multiple instruments on one line
            instruments.extend(
                [inst.strip() for inst in line.split("\t") if inst.strip()]
            )

    # Remove duplicates while preserving order
    seen = set()
    unique_instruments = []
    for inst in instruments:
        if inst not in seen:
            seen.add(inst)
            unique_instruments.append(inst)

    return unique_instruments


def index_test_data(data_dir: str | Path) -> pd.DataFrame:
    """Create an index of all IRMAS test data.

    Scans the IRMAS-TestingData-Part1 directory and pairs .wav files
    with their corresponding .txt label files.

    Args:
        data_dir: Path to IRMAS-TestingData-Part1 directory.

    Returns:
        DataFrame with columns:
        - filepath: absolute path to audio file
        - labels: list of instrument labels present
        - label_indices: list of label indices
        - filename: original filename

    Raises:
        FileNotFoundError: If data_dir doesn't exist.
        ValueError: If no audio files found.

    Example:
        >>> df = index_test_data('data/raw/IRMAS-TestingData-Part1')
        >>> df.iloc[0]['labels']
        ['gel', 'pia']
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")

    # Find all .wav files
    wav_files = sorted(data_dir.glob("*.wav"))

    if not wav_files:
        raise ValueError(f"No audio files found in {data_dir}")

    records = []

    for wav_file in wav_files:
        # Corresponding label file
        label_file = wav_file.with_suffix(".txt")

        if not label_file.exists():
            print(f"Warning: Label file not found for {wav_file.name}")
            continue

        # Parse labels
        labels = parse_test_labels(label_file)

        # Convert to indices
        label_indices = [
            LABEL_TO_IDX[label] for label in labels if label in LABEL_TO_IDX
        ]

        # Create record
        record = {
            "filepath": str(wav_file.resolve()),
            "labels": labels,
            "label_indices": label_indices,
            "filename": wav_file.name,
        }

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    return df


def get_class_distribution(df: pd.DataFrame, test: bool = False) -> pd.Series:
    """Get the distribution of instrument classes in the dataset.

    Args:
        df: DataFrame from index_training_data() or index_test_data().
        test: If True, counts multi-label test data. If False, counts training data.

    Returns:
        Series with instrument counts.

    Example:
        >>> df = index_training_data('data/raw/IRMAS-TrainingData')
        >>> dist = get_class_distribution(df)
        >>> print(dist)
        cel    388
        cla    505
        ...
    """
    if test:
        # For test data, explode multi-label lists
        labels_flat = df["labels"].explode()
        return labels_flat.value_counts().sort_index()
    else:
        # For training data, simple count
        return df["instrument"].value_counts().sort_index()


def get_artist_distribution(df: pd.DataFrame) -> pd.Series:
    """Get the distribution of samples per artist in training data.

    Useful for understanding data leakage risk and creating balanced splits.

    Args:
        df: DataFrame from index_training_data().

    Returns:
        Series with sample counts per artist_id.

    Example:
        >>> df = index_training_data('data/raw/IRMAS-TrainingData')
        >>> artist_dist = get_artist_distribution(df)
        >>> print(artist_dist.describe())
    """
    return df["artist_id"].value_counts()


def get_track_distribution(df: pd.DataFrame) -> pd.Series:
    """Get the distribution of samples per track in training data.

    Multiple 3-second excerpts may come from the same original track.
    This helps identify tracks that should stay together during splitting.

    Args:
        df: DataFrame from index_training_data().

    Returns:
        Series with sample counts per track_id.

    Example:
        >>> df = index_training_data('data/raw/IRMAS-TrainingData')
        >>> track_dist = get_track_distribution(df)
        >>> print(f"Tracks with multiple excerpts: {(track_dist > 1).sum()}")
    """
    return df["track_id"].value_counts()
