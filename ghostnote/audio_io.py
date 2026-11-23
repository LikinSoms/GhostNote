"""
Audio I/O and evaluation utilities for GhostNote.
Handles reading and writing WAV, MP3, and other common formats using soundfile,
along with sample rate conversion and perceptual audio quality metrics (SNR, PSNR).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    """Container for audio samples and metadata."""
    samples: np.ndarray  # Shape (N,) for mono or (N, C) for multi-channel, float32 in [-1, 1]
    sample_rate: int
    format: str
    channels: int
    duration_seconds: float

    @property
    def mono(self) -> np.ndarray:
        """Returns 1D mono audio (downmixed if multi-channel)."""
        if self.samples.ndim == 1:
            return self.samples
        return np.mean(self.samples, axis=1)


def load_audio(file_path: Union[str, Path], target_sr: int = None) -> AudioData:
    """
    Load an audio file into normalized float32 format.
    Supports WAV, MP3, FLAC, AIFF, OGG, etc.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    
    # Clip any wild floating point excursions
    samples = np.clip(samples, -1.0, 1.0)
    
    # Handle sample rate conversion if needed
    if target_sr is not None and sr != target_sr:
        from scipy import signal
        num_samples = int(round(len(samples) * float(target_sr) / sr))
        if samples.ndim == 1:
            samples = signal.resample(samples, num_samples).astype(np.float32)
        else:
            samples = signal.resample(samples, num_samples, axis=0).astype(np.float32)
        sr = target_sr

    channels = 1 if samples.ndim == 1 else samples.shape[1]
    duration = len(samples) / sr

    return AudioData(
        samples=samples,
        sample_rate=sr,
        format=path.suffix.lstrip(".").upper(),
        channels=channels,
        duration_seconds=duration,
    )


def save_audio(
    file_path: Union[str, Path],
    samples: np.ndarray,
    sample_rate: int,
    subtype: str = None
) -> Path:
    """
    Save audio samples to disk (WAV or MP3).
    Ensures signal bounds [-1, 1] and proper float32/int16 encoding.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clamp samples to prevent clipping distortion
    clipped = np.clip(samples, -1.0, 1.0).astype(np.float32)
    
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        # MP3 write using soundfile
        sf.write(str(path), clipped, sample_rate, format="MP3")
    elif suffix == ".wav":
        # Write PCM_16 or specified subtype for maximum compatibility & regularized dither floor
        st = subtype if subtype else "PCM_16"
        sf.write(str(path), clipped, sample_rate, format="WAV", subtype=st)
    else:
        # Default fallback
        sf.write(str(path), clipped, sample_rate)
        
    return path


def compute_audio_metrics(original: np.ndarray, stego: np.ndarray) -> dict:
    """
    Compute Signal-to-Noise Ratio (SNR) and Peak Signal-to-Noise Ratio (PSNR)
    between original carrier and stego audio to evaluate imperceptibility.
    """
    orig = np.asarray(original, dtype=np.float64)
    stg = np.asarray(stego, dtype=np.float64)

    min_len = min(len(orig), len(stg))
    orig = orig[:min_len]
    stg = stg[:min_len]

    diff = orig - stg
    mse = np.mean(diff ** 2)
    orig_power = np.mean(orig ** 2)

    if mse == 0:
        snr = float("inf")
        psnr = float("inf")
    else:
        snr = 10.0 * np.log10(orig_power / (mse + 1e-12))
        max_val = max(np.max(np.abs(orig)), 1e-6)
        psnr = 20.0 * np.log10(max_val / np.sqrt(mse + 1e-12))

    # Cross-correlation / Normalized Pearson correlation
    norm_orig = orig - np.mean(orig)
    norm_stg = stg - np.mean(stg)
    denom = np.sqrt(np.sum(norm_orig ** 2) * np.sum(norm_stg ** 2))
    correlation = float(np.sum(norm_orig * norm_stg) / (denom + 1e-12))

    return {
        "snr_db": round(float(snr), 2),
        "psnr_db": round(float(psnr), 2),
        "mse": float(mse),
        "pearson_correlation": round(correlation, 6),
        "perceptual_distortion": "Imperceptible" if snr > 30 else ("Very Low" if snr > 20 else "Moderate"),
    }
