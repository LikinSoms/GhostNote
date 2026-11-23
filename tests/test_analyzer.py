"""Unit tests for the Psychoacoustic Audio Analyzer."""

import numpy as np
import pytest
from ghostnote.analyzer import AudioAnalyzer


@pytest.fixture
def analyzer():
    return AudioAnalyzer()


def test_tonal_audio_recommends_echo_hiding(analyzer):
    """Periodic multi-harmonic signals should strongly favor Echo Hiding."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(duration * sr))
    # Rich harmonic acoustic chord
    audio = (0.4 * np.sin(2 * np.pi * 261.63 * t) +
             0.3 * np.sin(2 * np.pi * 329.63 * t) +
             0.2 * np.sin(2 * np.pi * 392.00 * t)).astype(np.float32)

    result = analyzer.analyze(audio, sr)
    assert result.recommended_algorithm == "Echo Hiding"
    assert result.echo_score > result.dsss_score
    assert result.tonality_index > 0.8
    assert result.spectral_flatness < 0.15
    assert len(result.reasons) > 0


def test_broadband_noise_recommends_dsss(analyzer):
    """Broadband noise-like signals should strongly favor DSSS."""
    sr = 44100
    duration = 5.0
    audio = np.random.normal(0, 0.4, int(duration * sr)).astype(np.float32)

    result = analyzer.analyze(audio, sr)
    assert result.recommended_algorithm == "DSSS"
    assert result.dsss_score > result.echo_score
    assert result.spectral_flatness > 0.30
    assert result.spectral_entropy > 0.80
    assert len(result.reasons) > 0


def test_too_short_audio_raises_error(analyzer):
    """Signals shorter than 0.2s should raise an error."""
    sr = 44100
    short_audio = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ValueError, match="too short"):
        analyzer.analyze(short_audio, sr)


def test_capacity_estimates(analyzer):
    """Capacity estimates should be positive and proportional to duration."""
    sr = 44100
    duration = 10.0
    audio = np.random.normal(0, 0.2, int(duration * sr)).astype(np.float32)
    result = analyzer.analyze(audio, sr)
    assert "Echo Hiding" in result.capacity_estimate_bytes
    assert "DSSS" in result.capacity_estimate_bytes
    assert result.capacity_estimate_bytes["DSSS"] > 0
    assert result.capacity_estimate_bytes["Echo Hiding"] > 0
