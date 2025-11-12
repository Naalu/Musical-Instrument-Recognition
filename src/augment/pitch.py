"""
Pitch shift augmentation for audio signals.

This module provides pitch shifting capabilities for audio augmentation,
which helps the model generalize to variations in instrument tuning and pitch.
"""

from typing import Optional

import librosa
import numpy as np


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
) -> np.ndarray:
    """
    Shift the pitch of an audio signal by a specified number of semitones.

    Pitch shifting changes the fundamental frequency of the audio without
    changing its duration. This is useful for data augmentation because it
    simulates instruments playing at different pitches/tunings.

    Args:
        audio: Input audio signal as numpy array of shape (n_samples,)
        sr: Sample rate in Hz (e.g., 22050)
        n_steps: Number of semitones to shift. Positive values shift up,
                 negative values shift down. Float values are allowed for
                 fractional semitone shifts (e.g., 0.5 = quarter tone).

    Returns:
        Pitch-shifted audio signal as numpy array of shape (n_samples,)

    Example:
        >>> audio, sr = librosa.load('cello.wav', sr=22050)
        >>> # Shift up by 2 semitones (one whole step)
        >>> audio_shifted = pitch_shift(audio, sr, n_steps=2.0)
        >>> # Shift down by 1 semitone (one half step)
        >>> audio_shifted = pitch_shift(audio, sr, n_steps=-1.0)

    Note:
        Uses librosa's pitch shifting algorithm which employs phase vocoding
        to maintain audio quality while changing pitch.
    """
    # Use librosa's pitch_shift function
    # This uses a phase vocoder algorithm to shift pitch without changing duration
    shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    return shifted_audio


def random_pitch_shift(
    audio: np.ndarray,
    sr: int,
    max_steps: float = 2.0,
    probability: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly apply pitch shifting with a given probability.

    This function randomly decides whether to apply pitch shifting, and if so,
    randomly samples the shift amount from [-max_steps, +max_steps].

    Args:
        audio: Input audio signal as numpy array of shape (n_samples,)
        sr: Sample rate in Hz (e.g., 22050)
        max_steps: Maximum number of semitones to shift (in either direction).
                   The actual shift will be sampled from [-max_steps, +max_steps].
                   Default: 2.0 semitones (one whole step).
        probability: Probability of applying pitch shift (0.0 to 1.0).
                     Default: 0.5 (apply to 50% of samples).
        seed: Random seed for reproducibility. If None, uses current random state.
              NOTE: In production training, use src.utils.seed.seed_everything()
              at the start of your script instead of passing seed here. This
              parameter is primarily for unit testing and debugging.

    Returns:
        Augmented audio signal. Either pitch-shifted (if applied) or
        original audio (if not applied).

    Example:
        >>> # Production usage (no seed parameter)
        >>> audio, sr = librosa.load('guitar.wav', sr=22050)
        >>> audio_aug = random_pitch_shift(audio, sr, max_steps=2.0, probability=0.5)
        >>>
        >>> # Testing usage (with seed for reproducibility)
        >>> audio_aug = random_pitch_shift(audio, sr, seed=42)

    Note:
        For reproducible training experiments, set the global random seed using
        src.utils.seed.seed_everything(42) at the start of your training script
        rather than passing seed to individual augmentation calls. This ensures
        all random operations (model init, data loading, augmentation) are
        reproducible together.
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Randomly decide whether to apply pitch shift
    if np.random.random() < probability:
        # Sample a random shift amount from [-max_steps, +max_steps]
        # np.random.uniform samples from a continuous uniform distribution
        n_steps = np.random.uniform(-max_steps, max_steps)

        # Apply pitch shift
        return pitch_shift(audio, sr, n_steps)
    else:
        # Return original audio (no augmentation)
        return audio
