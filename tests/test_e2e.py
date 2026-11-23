"""End-to-end tests for the complete StegoEngine pipeline."""

import numpy as np
import pytest
from ghostnote.engine import StegoEngine
from ghostnote.audio_io import save_audio


@pytest.fixture
def test_audio_files(tmp_path):
    sr = 44100
    dur = 40.0
    num_samples = int(dur * sr)
    t = np.linspace(0, dur, num_samples, endpoint=False)

    # Tonal acoustic audio (favors Echo Hiding)
    tonal = (0.35 * np.sin(2 * np.pi * 261.63 * t) +
             0.25 * np.sin(2 * np.pi * 329.63 * t) +
             0.20 * np.sin(2 * np.pi * 392.00 * t) +
             0.15 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
    tonal *= np.exp(-t * 0.05)
    tonal += 0.02 * np.random.normal(0, 0.2, num_samples).astype(np.float32)
    tonal_path = tmp_path / "carrier_tonal.wav"
    save_audio(tonal_path, tonal, sr, subtype="PCM_16")

    # Broadband noisy audio (favors DSSS)
    noisy = np.random.normal(0, 0.35, num_samples).astype(np.float32)
    noisy_path = tmp_path / "carrier_noisy.wav"
    save_audio(noisy_path, noisy, sr, subtype="PCM_16")

    return {
        "tonal": tonal_path,
        "noisy": noisy_path,
        "tmp_path": tmp_path,
    }


def test_e2e_auto_select_tonal(test_audio_files):
    """Verify that auto-selection on tonal audio chooses Echo Hiding and decrypts cleanly."""
    engine = StegoEngine()
    tonal_path = test_audio_files["tonal"]
    out_path = test_audio_files["tmp_path"] / "stego_tonal_out.wav"

    secret = "Nocturne Op 9 No 2 in E Flat Major"
    pwd = "FredericChopin1832!"

    embed_res = engine.embed(
        audio_path=tonal_path,
        message=secret,
        passphrase=pwd,
        output_path=out_path,
        algorithm="auto",
    )

    assert embed_res.success is True
    assert embed_res.algorithm_used == "Echo Hiding"
    assert embed_res.audio_metrics["snr_db"] > 10.0

    extract_res = engine.extract(out_path, passphrase=pwd)
    assert extract_res.success is True
    assert extract_res.plaintext == secret
    assert extract_res.algorithm == "Echo Hiding"


def test_e2e_auto_select_noisy(test_audio_files):
    """Verify that auto-selection on noisy audio chooses DSSS and decrypts cleanly."""
    engine = StegoEngine()
    noisy_path = test_audio_files["noisy"]
    out_path = test_audio_files["tmp_path"] / "stego_noisy_out.wav"

    secret = "Radar Jamming Frequency: 433.92 MHz"
    pwd = "SpectralShield2026#"

    embed_res = engine.embed(
        audio_path=noisy_path,
        message=secret,
        passphrase=pwd,
        output_path=out_path,
        algorithm="auto",
    )

    assert embed_res.success is True
    assert embed_res.algorithm_used == "DSSS"
    assert embed_res.audio_metrics["snr_db"] > 18.0

    extract_res = engine.extract(out_path, passphrase=pwd)
    assert extract_res.success is True
    assert extract_res.plaintext == secret
    assert extract_res.algorithm == "DSSS"


def test_e2e_dsss_mp3(test_audio_files):
    """Verify end-to-end DSSS embedding into MP3 file and extraction from MP3."""
    engine = StegoEngine()
    noisy_path = test_audio_files["noisy"]
    out_path = test_audio_files["tmp_path"] / "stego_noisy.mp3"

    secret = "Airborne Reconnaissance Unit 7"
    pwd = "TopSecretMP3Key!"

    embed_res = engine.embed(
        audio_path=noisy_path,
        message=secret,
        passphrase=pwd,
        output_path=out_path,
        algorithm="dsss",
    )

    assert embed_res.success is True
    assert out_path.suffix.lower() == ".mp3"

    extract_res = engine.extract(out_path, passphrase=pwd)
    assert extract_res.success is True
    assert extract_res.plaintext == secret
