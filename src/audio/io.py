"""Audio file I/O operations using librosa.

Handles loading audio files, resampling, and basic preprocessing.
All audio is loaded as mono (single channel) at target sample rate.

Example:
    >>> from src.audio.io import load_audio
    >>> waveform, sr = load_audio('audio.wav', target_sr=22050)
    >>> print(waveform.shape, sr)
    (66150,) 22050  # 3 seconds at 22.05 kHz
"""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    filepath: str | Path,
    target_sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Uses librosa for loading and resampling. Automatically converts to mono
    if requested. This is the primary audio loading function for IRMAS data.

    Args:
        filepath: Path to audio file (.wav, .mp3, .flac, etc.).
        target_sr: Target sample rate in Hz (default: 22050 for IRMAS).
        mono: If True, convert to mono by averaging channels.
        duration: Only load up to this duration in seconds (None = load all).
        offset: Start reading after this time in seconds.

    Returns:
        Tuple of (waveform, sample_rate):
        - waveform: Audio as numpy array, shape (n_samples,) if mono or (channels, n_samples)
        - sample_rate: Sample rate of returned audio (matches target_sr)

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If audio file cannot be read.

    Example:
        >>> # Load 3-second IRMAS training sample
        >>> waveform, sr = load_audio('data/raw/IRMAS-TrainingData/pia/001.wav')
        >>> print(f"Duration: {len(waveform)/sr:.2f}s at {sr}Hz")
        Duration: 3.00s at 22050Hz

        >>> # Load only first 1 second
        >>> waveform, sr = load_audio('audio.wav', duration=1.0)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        # Load audio with librosa
        # sr=target_sr does resampling automatically
        # mono=True mixes down to single channel
        waveform, sample_rate = librosa.load(
            filepath, sr=target_sr, mono=mono, duration=duration, offset=offset
        )

        return waveform, sample_rate

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {filepath}: {e}")


def get_audio_info(filepath: str | Path) -> dict:
    """Get metadata about an audio file without loading it.

    Fast way to get duration, sample rate, channels without loading
    the entire audio file into memory.

    Args:
        filepath: Path to audio file.

    Returns:
        Dictionary with keys:
        - duration: Duration in seconds
        - sample_rate: Original sample rate
        - channels: Number of channels (1=mono, 2=stereo)
        - frames: Total number of audio frames

    Example:
        >>> info = get_audio_info('data/raw/IRMAS-TrainingData/pia/001.wav')
        >>> print(f"Duration: {info['duration']:.2f}s")
        Duration: 3.00s
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        info = sf.info(str(filepath))

        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to read audio info from {filepath}: {e}")


def save_audio(
    filepath: str | Path,
    waveform: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_16",
) -> None:
    """Save audio waveform to file.

    Useful for saving preprocessed audio or generated audio.

    Args:
        filepath: Output file path.
        waveform: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        subtype: Audio encoding (default: 'PCM_16' for 16-bit WAV).

    Example:
        >>> waveform, sr = load_audio('input.wav')
        >>> # ... process waveform ...
        >>> save_audio('output.wav', waveform, sr)
    """
    filepath = Path(filepath)

    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(filepath), waveform, sample_rate, subtype=subtype)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {filepath}: {e}")


def normalize_audio(waveform: np.ndarray, method: str = "peak") -> np.ndarray:
    """Normalize audio waveform.

    Args:
        waveform: Audio data as numpy array.
        method: Normalization method:
            - 'peak': Scale so maximum absolute value is 1.0
            - 'rms': Scale so RMS is 0.1 (approximate speaking level)

    Returns:
        Normalized waveform (same shape as input).

    Example:
        >>> waveform, sr = load_audio('quiet.wav')
        >>> waveform_norm = normalize_audio(waveform, method='peak')
        >>> assert np.abs(waveform_norm).max() == 1.0
    """
    if method == "peak":
        # Peak normalization: scale so max absolute value is 1.0
        peak = np.abs(waveform).max()
        if peak > 0:
            return waveform / peak
        else:
            return waveform

    elif method == "rms":
        # RMS normalization: scale so RMS level is target
        target_rms = 0.1
        current_rms = np.sqrt(np.mean(waveform**2))
        if current_rms > 0:
            return waveform * (target_rms / current_rms)
        else:
            return waveform

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def trim_silence(
    waveform: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence from audio.

    Uses librosa's trim function which detects silence based on
    energy threshold (in decibels below peak).

    Args:
        waveform: Audio data as numpy array.
        top_db: Threshold in dB below peak for silence detection.
        frame_length: Frame length for energy calculation.
        hop_length: Hop length for energy calculation.

    Returns:
        Trimmed waveform (may be shorter than input).

    Example:
        >>> waveform, sr = load_audio('audio_with_silence.wav')
        >>> trimmed = trim_silence(waveform, top_db=30)
        >>> print(f"Trimmed {len(waveform) - len(trimmed)} samples")
    """
    trimmed, _ = librosa.effects.trim(
        waveform, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a different sample rate.

    Uses high-quality resampling from librosa.

    Args:
        waveform: Audio data at original sample rate.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled waveform.

    Example:
        >>> # Load at original sample rate
        >>> waveform, sr = load_audio('audio.wav', target_sr=None)
        >>> # Manually resample to 22050 Hz
        >>> resampled = resample_audio(waveform, sr, 22050)
    """
    if orig_sr == target_sr:
        return waveform

    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
