"""
Time stretch augmentation for audio signals.

This module provides time stretching capabilities for audio augmentation,
which helps the model generalize to variations in tempo and performance speed.
"""

from typing import Optional

import librosa
import numpy as np


def time_stretch(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Change the speed of an audio signal without changing its pitch.

    Time stretching modifies the duration of audio while preserving pitch.
    This is useful for data augmentation because it simulates tempo variations
    and different performance speeds.

    Args:
        audio: Input audio signal as numpy array of shape (n_samples,)
        rate: Speed factor. Values > 1.0 speed up (shorter duration),
              values < 1.0 slow down (longer duration).
              Examples:
              - rate=1.2 → 20% faster (audio becomes 0.83× original length)
              - rate=0.8 → 20% slower (audio becomes 1.25× original length)
              - rate=1.0 → no change

    Returns:
        Time-stretched audio signal as numpy array. Length will be
        approximately (n_samples / rate).

    Example:
        >>> audio, sr = librosa.load('guitar.wav', sr=22050)
        >>> # Speed up by 20% (play 1.2× faster)
        >>> audio_fast = time_stretch(audio, rate=1.2)
        >>> # Slow down by 20% (play 0.8× speed)
        >>> audio_slow = time_stretch(audio, rate=0.8)
        >>> print(f"Original: {len(audio)} samples")
        >>> print(f"Fast (1.2×): {len(audio_fast)} samples")
        >>> print(f"Slow (0.8×): {len(audio_slow)} samples")

    Note:
        Uses librosa's time stretching algorithm which employs phase vocoding
        to maintain audio quality while changing duration. The output length
        will differ from the input length by a factor of 1/rate.
    """
    # Use librosa's time_stretch function
    # Phase vocoder algorithm preserves pitch while changing duration
    stretched_audio = librosa.effects.time_stretch(y=audio, rate=rate)

    return stretched_audio


def random_time_stretch(
    audio: np.ndarray,
    rate_range: tuple = (0.8, 1.2),
    probability: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly apply time stretching with a given probability.

    This function randomly decides whether to apply time stretching, and if so,
    randomly samples the stretch rate from the specified range.

    Args:
        audio: Input audio signal as numpy array of shape (n_samples,)
        rate_range: Tuple of (min_rate, max_rate) for random sampling.
                    Default: (0.8, 1.2) allows 20% slower to 20% faster.
                    - min_rate < 1.0: audio will be slowed down
                    - max_rate > 1.0: audio will be sped up
                    - rate=1.0: no change
        probability: Probability of applying time stretch (0.0 to 1.0).
                     Default: 0.5 (apply to 50% of samples).
        seed: Random seed for reproducibility. If None, uses current random state.

    Returns:
        Augmented audio signal. Either time-stretched (if applied) or
        original audio (if not applied). Note: output length may differ
        from input length if stretching is applied.

    Example:
        >>> audio, sr = librosa.load('piano.wav', sr=22050)
        >>> # 50% chance to stretch between 0.8× and 1.2× speed
        >>> audio_aug = random_time_stretch(audio, rate_range=(0.8, 1.2), probability=0.5)
        >>> # Always stretch, wider range (0.7× to 1.3×)
        >>> audio_aug = random_time_stretch(audio, rate_range=(0.7, 1.3), probability=1.0)
        >>> # Never stretch (probability=0)
        >>> audio_aug = random_time_stretch(audio, rate_range=(0.8, 1.2), probability=0.0)

    Note:
        The probability parameter allows you to control augmentation intensity
        during training. Setting probability=0.5 means that on average, half
        of your training batches will have time-stretched samples.

        Output length will vary: if rate=1.2, output is ~0.83× input length.
        Your feature extraction should handle variable-length audio or you
        should pad/crop to fixed length after augmentation.
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Randomly decide whether to apply time stretch
    if np.random.random() < probability:
        # Sample a random rate from the specified range
        # np.random.uniform samples from a continuous uniform distribution
        rate = np.random.uniform(rate_range[0], rate_range[1])

        # Apply time stretch
        return time_stretch(audio, rate)
    else:
        # Return original audio (no augmentation)
        return audio
