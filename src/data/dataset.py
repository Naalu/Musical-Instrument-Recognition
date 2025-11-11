"""PyTorch Dataset for IRMAS musical instrument classification.

Provides efficient data loading with on-the-fly feature extraction.
Supports training, validation, and test splits.

Example:
    >>> from src.data.dataset import IRMASDataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create dataset
    >>> dataset = IRMASDataset(
    ...     data_dir='data/raw/IRMAS-TrainingData',
    ...     split='train',
    ...     target_sr=22050,
    ...     n_mels=128
    ... )
    >>>
    >>> # Create dataloader
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>>
    >>> # Iterate
    >>> for spectrograms, labels in loader:
    ...     print(spectrograms.shape, labels.shape)
    ...     break
    torch.Size([32, 1, 128, 130]) torch.Size([32])
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.irmas import IRMAS_CLASSES, index_training_data
from src.features.extraction import extract_melspectrogram_from_file


class IRMASDataset(Dataset):
    """PyTorch Dataset for IRMAS training data.

    Loads audio files and extracts mel-spectrograms on-the-fly.
    Supports train/val/test splits based on artist_id to prevent data leakage.

    Attributes:
        data_dir: Path to IRMAS-TrainingData directory.
        dataframe: Pandas DataFrame with file paths and labels.
        target_sr: Target sample rate for audio loading.
        n_mels: Number of mel frequency bands.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        transform: Optional transform to apply to spectrograms.
        num_classes: Number of instrument classes (11).

    Example:
        >>> dataset = IRMASDataset(
        ...     data_dir='data/raw/IRMAS-TrainingData',
        ...     target_sr=22050,
        ...     n_mels=128
        ... )
        >>> print(len(dataset))
        6705
        >>> melspec, label = dataset[0]
        >>> print(melspec.shape, label)
        torch.Size([1, 128, 130]) 0
    """

    def __init__(
        self,
        data_dir: str | Path,
        target_sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ):
        """Initialize IRMAS dataset.

        Args:
            data_dir: Path to IRMAS-TrainingData directory.
            target_sr: Target sample rate (default: 22050).
            n_mels: Number of mel bands (default: 128).
            n_fft: FFT window size (default: 2048).
            hop_length: Hop length (default: 512).
            transform: Optional transform (augmentation) to apply.
            indices: Optional list of indices to use (for train/val split).
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

        # Index the dataset
        self.dataframe = index_training_data(self.data_dir)

        # Filter by indices if provided (for train/val split)
        if indices is not None:
            self.dataframe = self.dataframe.iloc[indices].reset_index(drop=True)

        self.num_classes = len(IRMAS_CLASSES)

        print("IRMAS Dataset initialized:")
        print(f"  Samples: {len(self.dataframe)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Target SR: {self.target_sr}Hz")
        print(f"  Mel bands: {self.n_mels}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (mel_spectrogram, label):
            - mel_spectrogram: Tensor of shape (1, n_mels, time_frames)
            - label: Integer class label (0-10)
        """
        # Get file info
        row = self.dataframe.iloc[idx]
        filepath = row["filepath"]
        label = row["label_idx"]

        # Extract mel-spectrogram
        melspec, _ = extract_melspectrogram_from_file(
            filepath,
            target_sr=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Convert to tensor and add channel dimension
        # Shape: (n_mels, time) -> (1, n_mels, time)
        melspec_tensor = torch.from_numpy(melspec).float().unsqueeze(0)

        # Apply transform if provided (e.g., augmentation)
        if self.transform is not None:
            melspec_tensor = self.transform(melspec_tensor)

        return melspec_tensor, label

    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in this dataset.

        Returns:
            Dictionary mapping class names to counts.

        Example:
            >>> dataset = IRMASDataset('data/raw/IRMAS-TrainingData')
            >>> dist = dataset.get_class_distribution()
            >>> print(dist['pia'])
            721
        """
        counts = self.dataframe["instrument"].value_counts().to_dict()
        return counts

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sample metadata.

        Example:
            >>> info = dataset.get_sample_info(0)
            >>> print(info['instrument'], info['filename'])
        """
        row = self.dataframe.iloc[idx]
        return {
            "filepath": row["filepath"],
            "instrument": row["instrument"],
            "label_idx": row["label_idx"],
            "artist_id": row["artist_id"],
            "track_id": row["track_id"],
            "filename": row["filename"],
        }


def create_train_val_split(
    data_dir: str | Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Create train/validation split with proper leakage prevention.

    Splits by artist_id while trying to achieve target val_ratio in terms
    of sample count (not just artist count). This ensures validation set
    is large enough for reliable evaluation.

    Args:
        data_dir: Path to IRMAS-TrainingData directory.
        val_ratio: Target fraction of samples for validation (default: 0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices):
        - train_indices: List of indices for training set
        - val_indices: List of indices for validation set

    Example:
        >>> train_idx, val_idx = create_train_val_split(
        ...     'data/raw/IRMAS-TrainingData',
        ...     val_ratio=0.15,
        ...     random_seed=42
        ... )
        >>> print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        Train: 5699, Val: 1006
    """
    from src.utils.seed import set_seed

    # Set random seed
    set_seed(random_seed)

    # Index dataset
    df = index_training_data(data_dir)

    # Count samples per artist
    artist_counts = df.groupby("artist_id").size().to_dict()

    # Create list of (artist_id, sample_count) tuples
    artists_with_counts = list(artist_counts.items())

    # Shuffle artists
    np.random.shuffle(artists_with_counts)

    # Greedily assign artists to validation until we reach target ratio
    total_samples = len(df)
    target_val_samples = int(total_samples * val_ratio)

    val_artists = []
    val_sample_count = 0

    for artist_id, count in artists_with_counts:
        if val_sample_count < target_val_samples:
            val_artists.append(artist_id)
            val_sample_count += count
        else:
            # Stop once we've reached target
            break

    val_artists = set(val_artists)
    train_artists = set(artist_counts.keys()) - val_artists

    # Get indices for each split
    train_indices = df[df["artist_id"].isin(train_artists)].index.tolist()
    val_indices = df[df["artist_id"].isin(val_artists)].index.tolist()

    print("Train/Val Split:")
    print(f"  Total samples: {len(df)}")
    print(
        f"  Train samples: {len(train_indices)} ({len(train_indices) / len(df) * 100:.1f}%)"
    )
    print(
        f"  Val samples: {len(val_indices)} ({len(val_indices) / len(df) * 100:.1f}%)"
    )
    print(f"  Train artists: {len(train_artists)}")
    print(f"  Val artists: {len(val_artists)}")

    # Verify no overlap
    assert len(train_artists & val_artists) == 0, "Artist overlap detected!"

    # Warn if validation set is too far from target
    actual_val_ratio = len(val_indices) / len(df)
    if abs(actual_val_ratio - val_ratio) > 0.05:
        print(
            f"  ⚠️  Warning: Val ratio {actual_val_ratio:.1%} differs from target {val_ratio:.1%}"
        )
        print("      This is normal when splitting by artist (to prevent leakage).")

    return train_indices, val_indices


def create_stratified_train_val_split(
    data_dir: str | Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Create stratified train/val split maintaining class proportions.

    Similar to create_train_val_split but ensures each class has approximately
    the same train/val ratio. Uses a balanced assignment approach to handle
    cases where artist-level splitting makes perfect ratios impossible.

    Also prevents artist leakage by splitting at the artist level.

    Args:
        data_dir: Path to IRMAS-TrainingData directory.
        val_ratio: Target fraction for validation per class (default: 0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices).

    Example:
        >>> train_idx, val_idx = create_stratified_train_val_split(
        ...     'data/raw/IRMAS-TrainingData',
        ...     val_ratio=0.15
        ... )
    """
    from src.utils.seed import set_seed

    # Set random seed
    set_seed(random_seed)

    # Index dataset
    df = index_training_data(data_dir)

    train_indices = []
    val_indices = []

    # Split each class separately
    for instrument in IRMAS_CLASSES:
        # Get samples for this class
        class_df = df[df["instrument"] == instrument]

        # Count samples per artist for this class
        artist_counts = class_df.groupby("artist_id").size().to_dict()

        # Create list of (artist_id, sample_count) tuples and shuffle
        artists_with_counts = list(artist_counts.items())
        np.random.shuffle(artists_with_counts)

        # Use balanced assignment: for each artist, assign to train or val
        # based on which choice gets us closer to target ratio
        train_count = 0
        val_count = 0
        train_artists = []
        val_artists = []

        total_class_samples = len(class_df)
        target_val_count = total_class_samples * val_ratio

        for artist_id, count in artists_with_counts:
            # Calculate error if we add this artist to validation
            error_if_val = abs((val_count + count) - target_val_count)
            # Calculate error if we add this artist to training (val stays same)
            error_if_train = abs(val_count - target_val_count)

            # Choose whichever minimizes error from target
            if error_if_val < error_if_train:
                val_artists.append(artist_id)
                val_count += count
            else:
                train_artists.append(artist_id)
                train_count += count

        # Get indices for this class
        train_idx_class = class_df[
            class_df["artist_id"].isin(train_artists)
        ].index.tolist()
        val_idx_class = class_df[class_df["artist_id"].isin(val_artists)].index.tolist()

        train_indices.extend(train_idx_class)
        val_indices.extend(val_idx_class)

    print("Stratified Train/Val Split:")
    print(f"  Total samples: {len(df)}")
    print(
        f"  Train samples: {len(train_indices)} ({len(train_indices) / len(df) * 100:.1f}%)"
    )
    print(
        f"  Val samples: {len(val_indices)} ({len(val_indices) / len(df) * 100:.1f}%)"
    )

    # Show per-class distribution
    print("\n  Per-class distribution:")
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    for instrument in IRMAS_CLASSES:
        train_count = (train_df["instrument"] == instrument).sum()
        val_count = (val_df["instrument"] == instrument).sum()
        total_count = train_count + val_count
        val_pct = val_count / total_count * 100 if total_count > 0 else 0
        print(
            f"    {instrument}: train={train_count:3d}, val={val_count:3d} ({val_pct:.1f}% val)"
        )

    return train_indices, val_indices


def get_class_weights(dataset: IRMASDataset) -> torch.Tensor:
    """Calculate class weights for handling class imbalance.

    Computes inverse frequency weights that can be used with
    CrossEntropyLoss to give more importance to underrepresented classes.

    Args:
        dataset: IRMAS dataset.

    Returns:
        Tensor of shape (num_classes,) with class weights.

    Example:
        >>> dataset = IRMASDataset('data/raw/IRMAS-TrainingData')
        >>> weights = get_class_weights(dataset)
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """
    # Get class counts
    class_counts = np.zeros(len(IRMAS_CLASSES))

    for idx in range(len(dataset)):
        label = dataset.dataframe.iloc[idx]["label_idx"]
        class_counts[label] += 1

    # Calculate weights (inverse frequency)
    total = class_counts.sum()
    class_weights = total / (len(IRMAS_CLASSES) * class_counts)

    # Convert to tensor
    weights_tensor = torch.from_numpy(class_weights).float()

    print("Class weights (for handling imbalance):")
    for i, (instrument, weight) in enumerate(zip(IRMAS_CLASSES, class_weights)):
        print(f"  {instrument}: {weight:.3f} (count: {int(class_counts[i])})")

    return weights_tensor
