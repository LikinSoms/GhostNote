"""Unit tests for Echo Hiding Steganography Engine."""

import numpy as np
import pytest
from scipy.signal import butter, lfilter
from ghostnote.echo_stego import EchoHidingEngine
from ghostnote.crypto import encrypt_message, decrypt_message, bytes_to_bits, bits_to_bytes, locate_sync_sequence
from ghostnote.audio_io import compute_audio_metrics, save_audio, load_audio


def test_echo_embedding_and_extraction_wav(tmp_path):
    """Verify Echo Hiding embeds and extracts bits with high fidelity on rich acoustic carrier."""
    sr = 44100
    engine = EchoHidingEngine(frame_length=2048, delay_0_samples=50, delay_1_samples=90, base_alpha=0.22)

    msg = "Secret Sonata in C Minor"
    pwd = "ChopinMasterPass123!"
    pkt = encrypt_message(msg, pwd, "Echo Hiding")
    bits = bytes_to_bits(pkt)

    # Rich harmonic acoustic signal
    dur = 32.0
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    carrier = (0.35 * np.sin(2 * np.pi * 261.63 * t) +
               0.25 * np.sin(2 * np.pi * 329.63 * t) +
               0.20 * np.sin(2 * np.pi * 392.00 * t) +
               0.15 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
    carrier *= np.exp(-t * 0.05)
    carrier += 0.02 * np.random.normal(0, 0.2, len(carrier)).astype(np.float32)

    # Save to audio file first to ensure realistic sampled carrier
    raw_carrier_file = tmp_path / "carrier_raw.wav"
    save_audio(raw_carrier_file, carrier, sr, subtype="PCM_16")
    carrier = load_audio(raw_carrier_file).samples

    stego = engine.embed(carrier, bits, key=pwd)
    metrics = compute_audio_metrics(carrier, stego)
    assert metrics["snr_db"] > 10.0

    wav_file = tmp_path / "stego_echo.wav"
    save_audio(wav_file, stego, sr, subtype="PCM_16")

    reloaded = load_audio(wav_file)
    extracted_bits = engine.extract(reloaded.samples, expected_num_bits=len(bits))
    diff = np.where(bits != extracted_bits[:len(bits)])[0]
    assert len(diff) == 0, f"Found {len(diff)} mismatched bits in Echo Hiding WAV test"

    sync_idx = locate_sync_sequence(extracted_bits)
    assert sync_idx is not None
    decrypted = decrypt_message(bits_to_bytes(extracted_bits[sync_idx + 32:]), pwd)
    assert decrypted.plaintext == msg
