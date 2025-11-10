"""Audio feature extraction for musical instrument classification.

Extracts mel-spectrograms from audio waveforms. These 2D time-frequency
representations serve as inputs to the CNN model.

Example:
    >>> from src.features.extraction import extract_melspectrogram
    >>> from src.audio.io import load_audio
    >>>
    >>> # Load audio
    >>> waveform, sr = load_audio('audio.wav', target_sr=22050)
    >>>
    >>> # Extract mel-spectrogram
    >>> melspec = extract_melspectrogram(waveform, sr)
    >>> print(melspec.shape)
    (128, 130)  # (n_mels, time_frames) for 3-second audio
"""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_melspectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
) -> np.ndarray:
    """Extract mel-spectrogram from audio waveform.

    Computes a mel-scaled spectrogram, which provides better frequency
    resolution at lower frequencies (like human hearing). Returns log-scaled
    values for better dynamic range representation.

    Args:
        waveform: Audio waveform as 1D numpy array.
        sample_rate: Sample rate of the waveform (Hz).
        n_mels: Number of mel frequency bands (default: 128).
        n_fft: FFT window size (default: 2048).
        hop_length: Number of samples between frames (default: 512).
        fmin: Minimum frequency (Hz) (default: 0.0).
        fmax: Maximum frequency (Hz) (default: sr/2).
        power: Exponent for magnitude spectrogram (2.0 = power, 1.0 = magnitude).

    Returns:
        Log-mel spectrogram as 2D numpy array with shape (n_mels, time_frames).
        Values are in log scale (dB). This is a grayscale image suitable for CNNs.

    Example:
        >>> waveform, sr = load_audio('audio.wav', target_sr=22050)
        >>> melspec = extract_melspectrogram(waveform, sr, n_mels=128)
        >>> print(melspec.shape)
        (128, 130)  # 128 mel bands, ~130 time frames for 3s audio
    """
    # Compute mel spectrogram
    melspec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    # Convert to log scale (dB)
    # Add small epsilon to avoid log(0)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    return melspec_db


def extract_melspectrogram_from_file(
    filepath: str | Path,
    target_sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """Extract mel-spectrogram directly from audio file.

    Convenience function that combines audio loading and feature extraction.

    Args:
        filepath: Path to audio file.
        target_sr: Target sample rate for loading (default: 22050).
        n_mels: Number of mel frequency bands (default: 128).
        n_fft: FFT window size (default: 2048).
        hop_length: Hop length in samples (default: 512).
        **kwargs: Additional arguments passed to extract_melspectrogram().

    Returns:
        Tuple of (melspectrogram, sample_rate):
        - melspectrogram: Log-mel spectrogram, shape (n_mels, time_frames)
        - sample_rate: Sample rate used for extraction

    Example:
        >>> melspec, sr = extract_melspectrogram_from_file(
        ...     'data/raw/IRMAS-TrainingData/pia/001.wav',
        ...     target_sr=22050,
        ...     n_mels=128
        ... )
    """
    from src.audio.io import load_audio

    # Load audio
    waveform, sr = load_audio(filepath, target_sr=target_sr)

    # Extract mel-spectrogram
    melspec = extract_melspectrogram(
        waveform, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, **kwargs
    )

    return melspec, sr


def save_melspectrogram(
    melspec: np.ndarray,
    filepath: str | Path,
) -> None:
    """Save mel-spectrogram to numpy file.

    Saves as compressed .npz file for efficient storage.

    Args:
        melspec: Mel-spectrogram array to save.
        filepath: Output file path (will add .npz extension if missing).

    Example:
        >>> melspec, sr = extract_melspectrogram_from_file('audio.wav')
        >>> save_melspectrogram(melspec, 'cache/audio_melspec.npz')
    """
    filepath = Path(filepath)

    # Add .npz extension if not present
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(".npz")

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy file
    np.savez_compressed(filepath, melspectrogram=melspec)


def load_melspectrogram(filepath: str | Path) -> np.ndarray:
    """Load mel-spectrogram from numpy file.

    Args:
        filepath: Path to saved mel-spectrogram (.npz file).

    Returns:
        Mel-spectrogram array.

    Example:
        >>> melspec = load_melspectrogram('cache/audio_melspec.npz')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Mel-spectrogram file not found: {filepath}")

    # Load from numpy file
    data = np.load(filepath)
    melspec = data["melspectrogram"]

    return melspec


def visualize_melspectrogram(
    melspec: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    title: str = "Mel-Spectrogram",
    save_path: Optional[str | Path] = None,
) -> None:
    """Visualize mel-spectrogram as a plot.

    Creates a nice visualization of the mel-spectrogram with proper
    frequency and time axes.

    Args:
        melspec: Mel-spectrogram to visualize (n_mels, time_frames).
        sample_rate: Sample rate used for extraction.
        hop_length: Hop length used for extraction (for time axis).
        title: Plot title.
        save_path: If provided, save plot to this path instead of showing.

    Example:
        >>> melspec, sr = extract_melspectrogram_from_file('audio.wav')
        >>> visualize_melspectrogram(melspec, sr, title="Piano - 3s")
    """
    plt.figure(figsize=(10, 4))

    # Display mel-spectrogram
    img = librosa.display.specshow(
        melspec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="viridis",
    )

    plt.colorbar(img, format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def get_melspectrogram_shape(
    duration: float,
    sample_rate: int = 22050,
    hop_length: int = 512,
    n_mels: int = 128,
) -> Tuple[int, int]:
    """Calculate expected mel-spectrogram shape for given audio duration.

    Useful for pre-allocating arrays or verifying output shapes.

    Args:
        duration: Audio duration in seconds.
        sample_rate: Sample rate in Hz.
        hop_length: Hop length in samples.
        n_mels: Number of mel bands.

    Returns:
        Tuple of (n_mels, n_frames) representing the spectrogram shape.

    Example:
        >>> # For 3-second IRMAS training samples
        >>> shape = get_melspectrogram_shape(duration=3.0, sample_rate=22050)
        >>> print(shape)
        (128, 130)
    """
    # Calculate number of samples
    n_samples = int(duration * sample_rate)

    # Calculate number of frames
    # Formula: (n_samples - n_fft) / hop_length + 1
    # For simplicity, approximate as: n_samples / hop_length
    n_frames = int(np.ceil(n_samples / hop_length))

    return (n_mels, n_frames)


def normalize_melspectrogram(
    melspec: np.ndarray,
    method: str = "min_max",
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize mel-spectrogram values.

    Args:
        melspec: Mel-spectrogram array.
        method: Normalization method:
            - 'min_max': Scale to [0, 1] range
            - 'standard': Zero mean, unit variance (z-score)
            - 'mean_norm': Subtract mean, divide by std + eps
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized mel-spectrogram (same shape as input).

    Example:
        >>> melspec, sr = extract_melspectrogram_from_file('audio.wav')
        >>> melspec_norm = normalize_melspectrogram(melspec, method='min_max')
        >>> print(melspec_norm.min(), melspec_norm.max())
        0.0 1.0
    """
    if method == "min_max":
        # Scale to [0, 1]
        min_val = melspec.min()
        max_val = melspec.max()

        if max_val - min_val > eps:
            return (melspec - min_val) / (max_val - min_val)
        else:
            return melspec

    elif method == "standard" or method == "mean_norm":
        # Zero mean, unit variance
        mean = melspec.mean()
        std = melspec.std()

        if std > eps:
            return (melspec - mean) / (std + eps)
        else:
            return melspec - mean

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_features_batch(
    filepaths: list[str | Path],
    target_sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    show_progress: bool = True,
) -> list[np.ndarray]:
    """Extract mel-spectrograms from multiple audio files.

    Processes a batch of files and returns a list of spectrograms.

    Args:
        filepaths: List of paths to audio files.
        target_sr: Target sample rate (default: 22050).
        n_mels: Number of mel bands (default: 128).
        n_fft: FFT window size (default: 2048).
        hop_length: Hop length (default: 512).
        show_progress: If True, print progress.

    Returns:
        List of mel-spectrograms (one per file).

    Example:
        >>> files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
        >>> melspecs = extract_features_batch(files, n_mels=128)
        >>> print(f"Extracted {len(melspecs)} spectrograms")
    """
    melspecs = []
    total = len(filepaths)

    for i, filepath in enumerate(filepaths):
        if show_progress and (i % 10 == 0 or i == total - 1):
            print(f"Processing: {i + 1}/{total} ({100 * (i + 1) // total}%)")

        try:
            melspec, _ = extract_melspectrogram_from_file(
                filepath,
                target_sr=target_sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            melspecs.append(melspec)
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            melspecs.append(None)

    return melspecs
