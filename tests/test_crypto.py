"""Unit tests for Cryptographic Framing, AES-256-GCM, and Bitstream Serializer."""

import numpy as np
import pytest
from ghostnote.crypto import (
    encrypt_message,
    decrypt_message,
    bytes_to_bits,
    bits_to_bytes,
    locate_sync_sequence,
    SYNC_BITS_32,
    MAGIC_BYTES,
)


def test_encryption_decryption_roundtrip():
    """Verify that encrypted text can be decrypted with the correct password."""
    message = "Highly Classified Operations Briefing 2026"
    password = "CorrectHorseBatteryStaple#99"
    algorithm = "Echo Hiding"

    packet = encrypt_message(message, password, algorithm)
    assert packet.startswith(MAGIC_BYTES)
    assert len(packet) > len(message) + 40

    decrypted = decrypt_message(packet, password)
    assert decrypted.plaintext == message
    assert decrypted.algorithm == "Echo Hiding"


def test_wrong_password_fails_decryption():
    """Verify that an incorrect password fails AES-GCM decryption."""
    packet = encrypt_message("Secret Message", "GoodPassword123!", "DSSS")
    with pytest.raises(ValueError, match="Decryption failed"):
        decrypt_message(packet, "WrongPassword123!")


def test_tampered_packet_fails_crc():
    """Verify that tampering with any byte causes a CRC32 integrity failure."""
    packet = bytearray(encrypt_message("Authentic Message", "Pass123", "DSSS"))
    # Corrupt one byte in the ciphertext body
    packet[40] ^= 0xFF
    with pytest.raises(ValueError, match="CRC-32"):
        decrypt_message(bytes(packet), "Pass123")


def test_bitstream_serialization():
    """Verify conversion between bytes and bits with sync sequence."""
    test_bytes = b"Hello, Steganography!"
    bits = bytes_to_bits(test_bytes, include_sync=True)
    assert len(bits) == len(SYNC_BITS_32) + len(test_bytes) * 8

    # Check sync location
    idx = locate_sync_sequence(bits)
    assert idx == 0

    # Extract payload bytes
    recovered_bytes = bits_to_bytes(bits[idx + 32:])
    assert recovered_bytes == test_bytes


def test_sync_pattern_tolerance():
    """Verify that locate_sync_sequence tolerates up to max_bit_errors."""
    bits = np.array(SYNC_BITS_32, copy=True)
    # Flip 2 bits
    bits[3] ^= 1
    bits[17] ^= 1

    idx = locate_sync_sequence(bits, max_bit_errors=2)
    assert idx == 0
