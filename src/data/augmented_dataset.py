"""Augmented PyTorch Dataset for IRMAS with configurable data augmentation.

This module provides a dataset class that supports three types of augmentation:
1. Pitch shifting (applied to raw audio)
2. Time stretching (applied to raw audio)
3. SpecAugment (applied to mel-spectrogram)

The augmentations can be enabled/disabled independently for ablation studies.

Example:
    >>> # Full augmentation
    >>> dataset = AugmentedIRMASDataset(
    ...     data_dir='data/raw/IRMAS-TrainingData',
    ...     augment_pitch=True,
    ...     augment_stretch=True,
    ...     augment_specaug=True,
    ... )
    >>>
    >>> # SpecAugment only (for ablation)
    >>> dataset = AugmentedIRMASDataset(
    ...     data_dir='data/raw/IRMAS-TrainingData',
    ...     augment_pitch=False,
    ...     augment_stretch=False,
    ...     augment_specaug=True,
    ... )
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio.io import load_audio
from src.augment.pitch import random_pitch_shift
from src.augment.specaugment import random_spec_augment
from src.augment.stretch import random_time_stretch
from src.data.irmas import IRMAS_CLASSES, index_training_data
from src.features.extraction import extract_melspectrogram


class AugmentedIRMASDataset(Dataset):
    """PyTorch Dataset for IRMAS with configurable augmentation.

    This dataset loads raw audio, applies audio-level augmentations
    (pitch shift, time stretch), extracts mel-spectrograms, and then
    applies spectrogram-level augmentation (SpecAugment).

    The augmentation pipeline is:
        Raw Audio → [Pitch Shift] → [Time Stretch] → Mel-Spectrogram → [SpecAugment]

    Each augmentation can be independently enabled/disabled for ablation studies.

    Attributes:
        data_dir: Path to IRMAS-TrainingData directory.
        augment_pitch: Whether to apply pitch shifting.
        augment_stretch: Whether to apply time stretching.
        augment_specaug: Whether to apply SpecAugment.
        target_sr: Target sample rate for audio.
        n_mels: Number of mel frequency bands.
        num_classes: Number of instrument classes (11).
    """

    def __init__(
        self,
        data_dir: str | Path,
        # Augmentation toggles (for ablation studies)
        augment_pitch: bool = False,
        augment_stretch: bool = False,
        augment_specaug: bool = False,
        # Pitch shift parameters
        pitch_shift_max_steps: float = 2.0,
        pitch_shift_probability: float = 0.5,
        # Time stretch parameters
        time_stretch_rate_range: Tuple[float, float] = (0.8, 1.2),
        time_stretch_probability: float = 0.5,
        # SpecAugment parameters
        specaug_freq_mask_max: int = 27,
        specaug_time_mask_max: int = 40,
        specaug_num_freq_masks: int = 2,
        specaug_num_time_masks: int = 2,
        specaug_probability: float = 0.8,
        # Audio/feature parameters
        target_sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        # Dataset subset
        indices: Optional[List[int]] = None,
    ):
        """Initialize augmented IRMAS dataset.

        Args:
            data_dir: Path to IRMAS-TrainingData directory.
            augment_pitch: Enable pitch shifting (default: False).
            augment_stretch: Enable time stretching (default: False).
            augment_specaug: Enable SpecAugment (default: False).
            pitch_shift_max_steps: Max semitones for pitch shift (default: 2.0).
            pitch_shift_probability: Probability of pitch shift (default: 0.5).
            time_stretch_rate_range: Rate range for stretch (default: (0.8, 1.2)).
            time_stretch_probability: Probability of stretch (default: 0.5).
            specaug_freq_mask_max: Max frequency mask size (default: 27).
            specaug_time_mask_max: Max time mask size (default: 40).
            specaug_num_freq_masks: Number of frequency masks (default: 2).
            specaug_num_time_masks: Number of time masks (default: 2).
            specaug_probability: Probability of SpecAugment (default: 0.8).
            target_sr: Target sample rate (default: 22050).
            n_mels: Number of mel bands (default: 128).
            n_fft: FFT window size (default: 2048).
            hop_length: Hop length (default: 512).
            indices: Optional list of indices for train/val split.
        """
        self.data_dir = Path(data_dir)

        # Store augmentation settings
        self.augment_pitch = augment_pitch
        self.augment_stretch = augment_stretch
        self.augment_specaug = augment_specaug

        # Pitch shift parameters
        self.pitch_shift_max_steps = pitch_shift_max_steps
        self.pitch_shift_probability = pitch_shift_probability

        # Time stretch parameters
        self.time_stretch_rate_range = time_stretch_rate_range
        self.time_stretch_probability = time_stretch_probability

        # SpecAugment parameters
        self.specaug_freq_mask_max = specaug_freq_mask_max
        self.specaug_time_mask_max = specaug_time_mask_max
        self.specaug_num_freq_masks = specaug_num_freq_masks
        self.specaug_num_time_masks = specaug_num_time_masks
        self.specaug_probability = specaug_probability

        # Audio/feature parameters
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Calculate expected number of samples for 3-second audio
        self.target_duration = 3.0  # IRMAS training clips are 3 seconds
        self.target_samples = int(self.target_sr * self.target_duration)

        # Index the dataset
        self.dataframe = index_training_data(self.data_dir)

        # Filter by indices if provided (for train/val split)
        if indices is not None:
            self.dataframe = self.dataframe.iloc[indices].reset_index(drop=True)

        self.num_classes = len(IRMAS_CLASSES)

        # Print configuration summary
        self._print_config()

    def _print_config(self):
        """Print dataset configuration summary."""
        print("=" * 60)
        print("AugmentedIRMASDataset initialized")
        print("=" * 60)
        print(f"  Samples: {len(self.dataframe)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Target SR: {self.target_sr}Hz")
        print(f"  Mel bands: {self.n_mels}")
        print()
        print("Augmentation settings:")
        print(f"  Pitch shift:  {'✓ ENABLED' if self.augment_pitch else '✗ disabled'}")
        if self.augment_pitch:
            print(f"    - Max steps: ±{self.pitch_shift_max_steps} semitones")
            print(f"    - Probability: {self.pitch_shift_probability}")
        print(
            f"  Time stretch: {'✓ ENABLED' if self.augment_stretch else '✗ disabled'}"
        )
        if self.augment_stretch:
            print(f"    - Rate range: {self.time_stretch_rate_range}")
            print(f"    - Probability: {self.time_stretch_probability}")
        print(
            f"  SpecAugment:  {'✓ ENABLED' if self.augment_specaug else '✗ disabled'}"
        )
        if self.augment_specaug:
            print(
                f"    - Freq masks: {self.specaug_num_freq_masks} (max {self.specaug_freq_mask_max})"
            )
            print(
                f"    - Time masks: {self.specaug_num_time_masks} (max {self.specaug_time_mask_max})"
            )
            print(f"    - Probability: {self.specaug_probability}")
        print("=" * 60)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dataframe)

    def _pad_or_crop_audio(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is exactly target_samples long.

        Time stretching changes audio length, so we need to pad or crop
        to maintain consistent input size for the model.

        Args:
            audio: Input audio array.

        Returns:
            Audio array of exactly target_samples length.
        """
        current_length = len(audio)

        if current_length == self.target_samples:
            return audio
        elif current_length > self.target_samples:
            # Crop from center (keeps the middle portion)
            start = (current_length - self.target_samples) // 2
            return audio[start : start + self.target_samples]
        else:
            # Pad with zeros (silence) at the end
            padding = self.target_samples - current_length
            return np.pad(audio, (0, padding), mode="constant", constant_values=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample with augmentation applied.

        Augmentation pipeline:
        1. Load raw audio
        2. Apply pitch shift (if enabled)
        3. Apply time stretch (if enabled)
        4. Pad/crop to fixed length (if time stretch was applied)
        5. Extract mel-spectrogram
        6. Apply SpecAugment (if enabled)
        7. Convert to tensor

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

        # Step 1: Load raw audio
        audio, sr = load_audio(filepath, target_sr=self.target_sr, mono=True)

        # Step 2: Apply pitch shift (if enabled)
        if self.augment_pitch:
            audio = random_pitch_shift(
                audio,
                sr=self.target_sr,
                max_steps=self.pitch_shift_max_steps,
                probability=self.pitch_shift_probability,
            )

        # Step 3: Apply time stretch (if enabled)
        if self.augment_stretch:
            audio = random_time_stretch(
                audio,
                rate_range=self.time_stretch_rate_range,
                probability=self.time_stretch_probability,
            )
            # Step 4: Ensure fixed length after stretching
            audio = self._pad_or_crop_audio(audio)

        # Step 5: Extract mel-spectrogram
        melspec = extract_melspectrogram(
            audio,
            sample_rate=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Step 6: Apply SpecAugment (if enabled)
        if self.augment_specaug:
            melspec = random_spec_augment(
                melspec,
                freq_mask_max=self.specaug_freq_mask_max,
                time_mask_max=self.specaug_time_mask_max,
                num_freq_masks=self.specaug_num_freq_masks,
                num_time_masks=self.specaug_num_time_masks,
                probability=self.specaug_probability,
            )

        # Step 7: Convert to tensor and add channel dimension
        # Shape: (n_mels, time) -> (1, n_mels, time)
        melspec_tensor = torch.from_numpy(melspec).float().unsqueeze(0)

        return melspec_tensor, label

    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in this dataset.

        Returns:
            Dictionary mapping class names to counts.
        """
        counts = self.dataframe["instrument"].value_counts().to_dict()
        return counts

    def get_augmentation_config(self) -> dict:
        """Get the current augmentation configuration.

        Useful for logging experiment settings.

        Returns:
            Dictionary with all augmentation parameters.
        """
        return {
            "augment_pitch": self.augment_pitch,
            "augment_stretch": self.augment_stretch,
            "augment_specaug": self.augment_specaug,
            "pitch_shift_max_steps": self.pitch_shift_max_steps,
            "pitch_shift_probability": self.pitch_shift_probability,
            "time_stretch_rate_range": self.time_stretch_rate_range,
            "time_stretch_probability": self.time_stretch_probability,
            "specaug_freq_mask_max": self.specaug_freq_mask_max,
            "specaug_time_mask_max": self.specaug_time_mask_max,
            "specaug_num_freq_masks": self.specaug_num_freq_masks,
            "specaug_num_time_masks": self.specaug_num_time_masks,
            "specaug_probability": self.specaug_probability,
        }


def create_augmented_datasets(
    data_dir: str | Path,
    train_indices: List[int],
    val_indices: List[int],
    augment_pitch: bool = True,
    augment_stretch: bool = True,
    augment_specaug: bool = True,
    **kwargs,
) -> Tuple[AugmentedIRMASDataset, "IRMASDataset"]:
    """Create train (augmented) and validation (non-augmented) datasets.

    This is a convenience function that creates:
    - Training dataset WITH augmentation enabled
    - Validation dataset WITHOUT augmentation (for fair evaluation)

    Args:
        data_dir: Path to IRMAS-TrainingData.
        train_indices: Indices for training split.
        val_indices: Indices for validation split.
        augment_pitch: Enable pitch shift for training.
        augment_stretch: Enable time stretch for training.
        augment_specaug: Enable SpecAugment for training.
        **kwargs: Additional arguments passed to dataset constructors.

    Returns:
        Tuple of (train_dataset, val_dataset).

    Example:
        >>> train_idx, val_idx = create_stratified_train_val_split(data_dir)
        >>> train_ds, val_ds = create_augmented_datasets(
        ...     data_dir,
        ...     train_idx,
        ...     val_idx,
        ...     augment_pitch=True,
        ...     augment_stretch=True,
        ...     augment_specaug=True,
        ... )
    """
    # Import here to avoid circular imports
    from src.data.dataset import IRMASDataset

    # Training dataset WITH augmentation
    train_dataset = AugmentedIRMASDataset(
        data_dir=data_dir,
        augment_pitch=augment_pitch,
        augment_stretch=augment_stretch,
        augment_specaug=augment_specaug,
        indices=train_indices,
        **kwargs,
    )

    # Validation dataset WITHOUT augmentation
    # Use the original IRMASDataset for clean validation
    val_dataset = IRMASDataset(
        data_dir=data_dir,
        indices=val_indices,
        target_sr=kwargs.get("target_sr", 22050),
        n_mels=kwargs.get("n_mels", 128),
        n_fft=kwargs.get("n_fft", 2048),
        hop_length=kwargs.get("hop_length", 512),
    )

    return train_dataset, val_dataset
