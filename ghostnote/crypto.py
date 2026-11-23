"""
Cryptographic framing and bitstream serialization for GhostNote.
Provides AES-256-GCM encryption with PBKDF2 key derivation,
robust framing with sync patterns, algorithm tags, and CRC-32 integrity checks.
"""

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# 32-bit synchronization sequence with sharp autocorrelation peak
# (0xA55A6996 -> 10100101 01011010 01101001 10010110)
SYNC_PATTERN_32 = 0xA55A6996
SYNC_BITS_32 = np.array([int(b) for b in f"{SYNC_PATTERN_32:032b}"], dtype=np.uint8)

MAGIC_BYTES = b"GNOT"  # GhostNote magic identifier
ALGO_ECHO = 0x01
ALGO_DSSS = 0x02

ALGO_MAP = {
    ALGO_ECHO: "Echo Hiding",
    ALGO_DSSS: "DSSS",
}
ALGO_REVERSE_MAP = {
    "echo hiding": ALGO_ECHO,
    "echo": ALGO_ECHO,
    "dsss": ALGO_DSSS,
}


@dataclass
class DecryptedPayload:
    """Decrypted secret message and its steganographic metadata."""
    plaintext: str
    algorithm: str
    salt: bytes
    nonce: bytes
    ciphertext_len: int


def derive_key(passphrase: str, salt: bytes, iterations: int = 100_000) -> bytes:
    """Derive a 256-bit AES key from a passphrase using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_message(
    plaintext: str,
    passphrase: str,
    algorithm_name: str
) -> bytes:
    """
    Encrypts a plaintext string using AES-256-GCM and packages it into a framed packet:
    [MAGIC: 4B] [ALGO: 1B] [SALT: 16B] [NONCE: 12B] [CIPHER_LEN: 4B] [CIPHERTEXT+TAG: N+16B] [CRC32: 4B]
    """
    algo_key = algorithm_name.strip().lower()
    algo_id = ALGO_REVERSE_MAP.get(algo_key, ALGO_DSSS)

    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = derive_key(passphrase, salt)

    aesgcm = AESGCM(key)
    plaintext_bytes = plaintext.encode("utf-8")
    
    # Associated Data binds the magic and algorithm to the encrypted payload
    associated_data = MAGIC_BYTES + bytes([algo_id])
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext_bytes, associated_data)

    cipher_len = len(ciphertext_with_tag)
    
    # Assemble header + payload
    header = MAGIC_BYTES + bytes([algo_id]) + salt + nonce + struct.pack(">I", cipher_len)
    payload_body = header + ciphertext_with_tag
    crc = zlib.crc32(payload_body) & 0xFFFFFFFF
    packet = payload_body + struct.pack(">I", crc)

    return packet


def decrypt_message(
    packet: bytes,
    passphrase: str
) -> DecryptedPayload:
    """
    Parses, validates CRC32, checks magic, and decrypts AES-256-GCM ciphertext.
    Automatically handles trailing audio bytes if packet contains extra data.
    Raises ValueError on checksum failure or invalid password.
    """
    header_min = 4 + 1 + 16 + 12 + 4  # 37 bytes header
    if len(packet) < header_min:
        raise ValueError(f"Packet too short ({len(packet)} bytes). Minimum {header_min} bytes.")

    magic = packet[:4]
    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid magic header: {magic!r}. Expected {MAGIC_BYTES!r}.")

    algo_id = packet[4]
    algo_name = ALGO_MAP.get(algo_id, f"Unknown (0x{algo_id:02x})")

    salt = packet[5:21]
    nonce = packet[21:33]
    cipher_len = struct.unpack(">I", packet[33:37])[0]

    # Total packet length is header (37) + ciphertext_with_tag (cipher_len) + CRC (4)
    total_len = 37 + cipher_len + 4
    if len(packet) < total_len:
        raise ValueError(f"Packet incomplete: received {len(packet)} bytes, expected {total_len} bytes.")

    # Trim any trailing audio frame bytes
    trimmed_packet = packet[:total_len]
    payload_body = trimmed_packet[:-4]
    expected_crc = struct.unpack(">I", trimmed_packet[-4:])[0]
    actual_crc = zlib.crc32(payload_body) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise ValueError("Packet checksum (CRC-32) verification failed. Audio payload may be corrupted.")

    ciphertext_with_tag = payload_body[37:37 + cipher_len]

    key = derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    associated_data = MAGIC_BYTES + bytes([algo_id])

    try:
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data)
        plaintext = decrypted_bytes.decode("utf-8")
    except Exception as exc:
        raise ValueError("Decryption failed. Password incorrect or ciphertext authentication tag mismatch.") from exc

    return DecryptedPayload(
        plaintext=plaintext,
        algorithm=algo_name,
        salt=salt,
        nonce=nonce,
        ciphertext_len=cipher_len,
    )


def bytes_to_bits(data: bytes, include_sync: bool = True) -> np.ndarray:
    """
    Converts bytes to a 1D NumPy array of bits (0 or 1, uint8).
    Optionally prepends the 32-bit Barker/sync pattern for frame synchronization.
    """
    bit_list = []
    if include_sync:
        bit_list.extend(SYNC_BITS_32.tolist())

    for byte in data:
        for shift in range(7, -1, -1):
            bit_list.append((byte >> shift) & 1)

    return np.array(bit_list, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converts a bit array of length multiple of 8 into bytes."""
    byte_list = bytearray()
    num_bytes = len(bits) // 8
    for i in range(num_bytes):
        chunk = bits[i * 8:(i + 1) * 8]
        val = 0
        for bit in chunk:
            val = (val << 1) | int(bit)
        byte_list.append(val)
    return bytes(byte_list)


def locate_sync_sequence(
    bits: np.ndarray,
    sync_pattern: np.ndarray = SYNC_BITS_32,
    max_bit_errors: int = 4
) -> Optional[int]:
    """
    Finds the index of the sync pattern in a received bitstream with tolerance
    for up to max_bit_errors.
    """
    pattern_len = len(sync_pattern)
    if len(bits) < pattern_len:
        return None

    # Vectorized sliding Hamming distance
    best_idx = None
    min_errors = pattern_len

    for idx in range(len(bits) - pattern_len + 1):
        window = bits[idx:idx + pattern_len]
        errors = np.count_nonzero(window != sync_pattern)
        if errors < min_errors:
            min_errors = errors
            best_idx = idx
            if errors == 0:
                break

    if min_errors <= max_bit_errors:
        return best_idx
    return None
