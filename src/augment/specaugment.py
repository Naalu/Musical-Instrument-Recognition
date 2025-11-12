"""
SpecAugment for spectrogram masking.

This module implements SpecAugment (Park et al., 2019), which applies
frequency and time masking directly to spectrograms for data augmentation.

Reference:
    Park et al. "SpecAugment: A Simple Data Augmentation Method for
    Automatic Speech Recognition" (2019)
    https://arxiv.org/abs/1904.08779
"""

from typing import Optional

import numpy as np


def frequency_mask(
    spectrogram: np.ndarray,
    max_mask_size: int = 27,
    num_masks: int = 1,
    mask_value: float = 0.0,
) -> np.ndarray:
    """
    Apply frequency masking to a spectrogram.

    Masks out horizontal bands (frequency bins) in the spectrogram by setting
    them to a constant value. This helps the model learn to be robust to
    missing frequency information.

    Args:
        spectrogram: Input spectrogram as numpy array of shape (n_mels, n_frames)
                     where n_mels is the number of frequency bins (mel bands)
                     and n_frames is the number of time frames.
        max_mask_size: Maximum number of consecutive frequency bins to mask.
                       Default: 27 bins (from original SpecAugment paper).
        num_masks: Number of frequency masks to apply. Default: 1.
        mask_value: Value to use for masked regions. Default: 0.0.

    Returns:
        Augmented spectrogram with frequency masking applied.
        Same shape as input: (n_mels, n_frames).

    Example:
        >>> spec = np.random.randn(128, 130)  # 128 mel bands, 130 frames
        >>> # Mask one frequency band (up to 27 bins wide)
        >>> spec_aug = frequency_mask(spec, max_mask_size=27, num_masks=1)
        >>> # Mask two frequency bands
        >>> spec_aug = frequency_mask(spec, max_mask_size=27, num_masks=2)

    Note:
        The mask is applied to contiguous frequency bins. For example, if
        the random selection chooses to mask bins 50-76, all frequencies
        in that range will be set to mask_value across all time frames.
    """
    # Make a copy to avoid modifying the original
    spec_aug = spectrogram.copy()
    n_mels = spec_aug.shape[0]

    for _ in range(num_masks):
        # Randomly choose mask size (between 0 and max_mask_size)
        mask_size = np.random.randint(0, max_mask_size + 1)

        if mask_size == 0:
            continue

        # Randomly choose starting frequency bin
        # Ensure mask doesn't go beyond spectrogram bounds
        max_start = max(0, n_mels - mask_size)
        if max_start == 0:
            f_start = 0
        else:
            f_start = np.random.randint(0, max_start)

        # Apply mask (set all time frames for these frequencies to mask_value)
        spec_aug[f_start : f_start + mask_size, :] = mask_value

    return spec_aug


def time_mask(
    spectrogram: np.ndarray,
    max_mask_size: int = 40,
    num_masks: int = 1,
    mask_value: float = 0.0,
) -> np.ndarray:
    """
    Apply time masking to a spectrogram.

    Masks out vertical stripes (time frames) in the spectrogram by setting
    them to a constant value. This helps the model learn to be robust to
    missing temporal information.

    Args:
        spectrogram: Input spectrogram as numpy array of shape (n_mels, n_frames)
                     where n_mels is the number of frequency bins (mel bands)
                     and n_frames is the number of time frames.
        max_mask_size: Maximum number of consecutive time frames to mask.
                       Default: 40 frames (from original SpecAugment paper).
        num_masks: Number of time masks to apply. Default: 1.
        mask_value: Value to use for masked regions. Default: 0.0.

    Returns:
        Augmented spectrogram with time masking applied.
        Same shape as input: (n_mels, n_frames).

    Example:
        >>> spec = np.random.randn(128, 130)  # 128 mel bands, 130 frames
        >>> # Mask one time segment (up to 40 frames wide)
        >>> spec_aug = time_mask(spec, max_mask_size=40, num_masks=1)
        >>> # Mask two time segments
        >>> spec_aug = time_mask(spec, max_mask_size=40, num_masks=2)

    Note:
        The mask is applied to contiguous time frames. For example, if
        the random selection chooses to mask frames 60-99, all frequencies
        in those time frames will be set to mask_value.
    """
    # Make a copy to avoid modifying the original
    spec_aug = spectrogram.copy()
    n_frames = spec_aug.shape[1]

    for _ in range(num_masks):
        # Randomly choose mask size (between 0 and max_mask_size)
        mask_size = np.random.randint(0, max_mask_size + 1)

        if mask_size == 0:
            continue

        # Randomly choose starting time frame
        # Ensure mask doesn't go beyond spectrogram bounds
        max_start = max(0, n_frames - mask_size)
        if max_start == 0:
            t_start = 0
        else:
            t_start = np.random.randint(0, max_start)

        # Apply mask (set all frequencies for these time frames to mask_value)
        spec_aug[:, t_start : t_start + mask_size] = mask_value

    return spec_aug


def spec_augment(
    spectrogram: np.ndarray,
    freq_mask_max: int = 27,
    time_mask_max: int = 40,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    mask_value: float = 0.0,
) -> np.ndarray:
    """
    Apply SpecAugment (frequency + time masking) to a spectrogram.

    This is the complete SpecAugment transformation that applies both
    frequency and time masking to the input spectrogram.

    Args:
        spectrogram: Input spectrogram as numpy array of shape (n_mels, n_frames).
        freq_mask_max: Maximum size of frequency masks (in bins). Default: 27.
        time_mask_max: Maximum size of time masks (in frames). Default: 40.
        num_freq_masks: Number of frequency masks to apply. Default: 1.
        num_time_masks: Number of time masks to apply. Default: 1.
        mask_value: Value to use for masked regions. Default: 0.0.
                    For log-mel spectrograms, 0.0 or the minimum value
                    of the spectrogram works well.

    Returns:
        Augmented spectrogram with both frequency and time masking applied.
        Same shape as input: (n_mels, n_frames).

    Example:
        >>> spec = np.random.randn(128, 130)  # 128 mel bands, 130 frames
        >>> # Apply standard SpecAugment (1 freq mask + 1 time mask)
        >>> spec_aug = spec_augment(spec, freq_mask_max=27, time_mask_max=40)
        >>> # Apply stronger augmentation (2 of each mask)
        >>> spec_aug = spec_augment(
        ...     spec,
        ...     freq_mask_max=27,
        ...     time_mask_max=40,
        ...     num_freq_masks=2,
        ...     num_time_masks=2
        ... )

    Note:
        Parameters are based on the original SpecAugment paper for
        audio classification tasks. For IRMAS with 128 mel bands and
        ~130 time frames, the defaults work well:
        - freq_mask_max=27 (≈21% of 128 mel bands)
        - time_mask_max=40 (≈31% of 130 time frames)
    """
    # Apply frequency masking first
    spec_aug = frequency_mask(
        spectrogram,
        max_mask_size=freq_mask_max,
        num_masks=num_freq_masks,
        mask_value=mask_value,
    )

    # Then apply time masking
    spec_aug = time_mask(
        spec_aug,
        max_mask_size=time_mask_max,
        num_masks=num_time_masks,
        mask_value=mask_value,
    )

    return spec_aug


def random_spec_augment(
    spectrogram: np.ndarray,
    freq_mask_max: int = 27,
    time_mask_max: int = 40,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    mask_value: float = 0.0,
    probability: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly apply SpecAugment with a given probability.

    This function randomly decides whether to apply SpecAugment. This is
    useful during training to create a mix of augmented and non-augmented
    samples.

    Args:
        spectrogram: Input spectrogram as numpy array of shape (n_mels, n_frames).
        freq_mask_max: Maximum size of frequency masks (in bins). Default: 27.
        time_mask_max: Maximum size of time masks (in frames). Default: 40.
        num_freq_masks: Number of frequency masks to apply. Default: 1.
        num_time_masks: Number of time masks to apply. Default: 1.
        mask_value: Value to use for masked regions. Default: 0.0.
        probability: Probability of applying SpecAugment (0.0 to 1.0).
                     Default: 0.5 (apply to 50% of samples).
        seed: Random seed for reproducibility. If None, uses current random state.

    Returns:
        Augmented spectrogram (if applied) or original spectrogram (if not applied).
        Same shape as input: (n_mels, n_frames).

    Example:
        >>> spec = np.random.randn(128, 130)
        >>> # 50% chance to apply SpecAugment
        >>> spec_aug = random_spec_augment(spec, probability=0.5)
        >>> # Always apply (probability=1.0)
        >>> spec_aug = random_spec_augment(spec, probability=1.0)
        >>> # Never apply (probability=0.0)
        >>> spec_aug = random_spec_augment(spec, probability=0.0)

    Note:
        The probability parameter allows you to control augmentation intensity
        during training. Setting probability=0.5 means that on average, half
        of your training batches will have SpecAugment applied.
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Randomly decide whether to apply SpecAugment
    if np.random.random() < probability:
        return spec_augment(
            spectrogram,
            freq_mask_max=freq_mask_max,
            time_mask_max=time_mask_max,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
            mask_value=mask_value,
        )
    else:
        # Return original spectrogram (no augmentation)
        return spectrogram
