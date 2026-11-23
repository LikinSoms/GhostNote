"""Unit tests for Direct Sequence Spread Spectrum (DSSS) Engine."""

import numpy as np
import pytest
from ghostnote.dsss_stego import DSSSEngine
from ghostnote.crypto import encrypt_message, decrypt_message, bytes_to_bits, bits_to_bytes, locate_sync_sequence
from ghostnote.audio_io import compute_audio_metrics, save_audio, load_audio


def test_dsss_embedding_and_extraction():
    """Verify DSSS embeds and extracts bits with zero errors on noise carrier."""
    sr = 44100
    engine = DSSSEngine(chip_length=1024, target_margin=0.040, max_delta=0.22)
    
    msg = "DSSS Tactical Spread Spectrum Test"
    pwd = "DsssPassword#2026"
    pkt = encrypt_message(msg, pwd, "DSSS")
    bits = bytes_to_bits(pkt)

    num_samples = len(bits) * 1024 + 44100
    carrier = np.random.normal(0, 0.35, num_samples).astype(np.float32)

    stego = engine.embed(carrier, bits, key=pwd)
    metrics = compute_audio_metrics(carrier, stego)
    assert metrics["snr_db"] > 18.0

    extracted_bits = engine.extract(stego, expected_num_bits=len(bits), key=pwd)
    diff = np.where(bits != extracted_bits[:len(bits)])[0]
    assert len(diff) == 0, f"Found {len(diff)} mismatched bits in DSSS memory test"

    sync_idx = locate_sync_sequence(extracted_bits)
    assert sync_idx is not None
    decrypted = decrypt_message(bits_to_bytes(extracted_bits[sync_idx + 32:]), pwd)
    assert decrypted.plaintext == msg


def test_dsss_mp3_compression_resilience(tmp_path):
    """Verify DSSS survives lossy MP3 compression without errors."""
    sr = 44100
    engine = DSSSEngine(chip_length=1024, target_margin=0.040, max_delta=0.22)

    msg = "MP3 Lossy Compression Resilience Test"
    pwd = "MP3SecretKey!42"
    pkt = encrypt_message(msg, pwd, "DSSS")
    bits = bytes_to_bits(pkt)

    num_samples = len(bits) * 1024 + 44100
    carrier = np.random.normal(0, 0.35, num_samples).astype(np.float32)

    stego = engine.embed(carrier, bits, key=pwd)
    mp3_file = tmp_path / "stego_test.mp3"
    save_audio(mp3_file, stego, sr)

    reloaded = load_audio(mp3_file)
    extracted_bits = engine.extract(reloaded.samples, expected_num_bits=len(bits), key=pwd)
    diff = np.where(bits != extracted_bits[:len(bits)])[0]
    assert len(diff) == 0, f"Found {len(diff)} mismatched bits after MP3 encode/decode"

    sync_idx = locate_sync_sequence(extracted_bits)
    assert sync_idx is not None
    decrypted = decrypt_message(bits_to_bytes(extracted_bits[sync_idx + 32:]), pwd)
    assert decrypted.plaintext == msg
