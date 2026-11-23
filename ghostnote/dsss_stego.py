"""
Direct Sequence Spread Spectrum (DSSS) Steganography Engine for GhostNote.
Spreads secret bit energy across a wide bandwidth using pseudo-random noise (PN)
sequences with Informed Host-Interference Cancellation to achieve zero bit-error
rates with high Signal-to-Noise Ratio (SNR > 25-35 dB).
"""

import hashlib
from typing import Optional, Tuple
import numpy as np

from ghostnote.crypto import SYNC_BITS_32


class DSSSEngine:
    """
    Embeds and extracts binary streams into audio carriers using DSSS
    with informed embedding and matched-filter correlation.
    """

    def __init__(
        self,
        chip_length: int = 1024,
        target_margin: float = 0.040,
        max_delta: float = 0.22,
    ):
        self.chip_length = chip_length
        self.target_margin = target_margin
        self.max_delta = max_delta

    def _generate_pn_sequence(self, seed_key: str, length: int) -> np.ndarray:
        """
        Generates a balanced bipolar pseudo-random noise sequence {-1.0, +1.0}
        from a seed key using SHA-256 for deterministic reproducibility.
        """
        digest = hashlib.sha256(seed_key.encode("utf-8") + b"::DSSS_PN_V2").digest()
        seed_int = int.from_bytes(digest[:8], byteorder="big")
        rng = np.random.default_rng(seed_int)
        pn = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
        return pn

    def calculate_capacity_bits(self, num_samples: int) -> int:
        """Returns the maximum number of bits that can fit in the audio."""
        return num_samples // self.chip_length

    def embed(
        self,
        audio: np.ndarray,
        bits: np.ndarray,
        key: str = "GhostNoteDefaultKey"
    ) -> np.ndarray:
        """
        Embeds a 1D bitstream {0, 1} into carrier audio using Informed DSSS.
        Preserves stereo if audio is 2D by embedding into the primary channel.
        """
        is_stereo = audio.ndim == 2
        carrier = audio[:, 0].copy() if is_stereo else audio.copy()
        num_samples = len(carrier)

        num_bits = len(bits)
        required_samples = num_bits * self.chip_length
        if required_samples > num_samples:
            max_bytes = (num_samples // self.chip_length) // 8
            raise ValueError(
                f"Audio carrier is too short for DSSS! "
                f"Carrier capacity is {num_samples // self.chip_length} bits ({max_bytes} bytes), "
                f"but encrypted payload requires {num_bits} bits ({num_bits // 8} bytes)."
            )

        pn = self._generate_pn_sequence(key, self.chip_length)
        stego = carrier.astype(np.float64)

        for i, bit in enumerate(bits):
            start = i * self.chip_length
            end = start + self.chip_length
            frame = carrier[start:end]

            # Bipolar target symbol: 0 -> -1.0, 1 -> +1.0
            symbol = 1.0 if bit == 1 else -1.0

            # Measure existing host signal cross-correlation with PN sequence
            host_corr = float(np.dot(frame, pn)) / self.chip_length

            # Informed embedding: calculate exact delta required to set correlation to symbol * margin
            target = symbol * self.target_margin
            delta = target - host_corr

            # Bound delta to prevent perceptual distortion
            delta = float(np.clip(delta, -self.max_delta, self.max_delta))

            stego[start:end] += delta * pn

        # Prevent clipping distortion
        stego = np.clip(stego, -1.0, 1.0).astype(np.float32)

        if is_stereo:
            out = audio.copy()
            out[:, 0] = stego
            return out

        return stego

    def extract(
        self,
        audio: np.ndarray,
        expected_num_bits: Optional[int] = None,
        key: str = "GhostNoteDefaultKey",
        search_sync: bool = True
    ) -> np.ndarray:
        """
        Extracts embedded bits from carrier audio using matched-filter correlation.
        """
        if audio.ndim == 2:
            signal_data = audio[:, 0]
        else:
            signal_data = audio

        pn = self._generate_pn_sequence(key, self.chip_length)
        num_samples = len(signal_data)

        # Synchronize sample offset using sync pattern if requested
        start_offset = 0
        sync_bits = SYNC_BITS_32
        sync_len = len(sync_bits)

        if search_sync and num_samples >= (sync_len + 8) * self.chip_length:
            start_offset = self._find_sync_offset(signal_data, pn, sync_bits)

        usable_samples = num_samples - start_offset
        total_possible_bits = usable_samples // self.chip_length

        if expected_num_bits is not None:
            bits_to_read = min(expected_num_bits, total_possible_bits)
        else:
            bits_to_read = total_possible_bits

        extracted = np.zeros(bits_to_read, dtype=np.uint8)

        for i in range(bits_to_read):
            start = start_offset + i * self.chip_length
            end = start + self.chip_length
            frame = signal_data[start:end]

            corr = float(np.dot(frame, pn))
            extracted[i] = 1 if corr > 0 else 0

        return extracted

    def _find_sync_offset(
        self,
        audio: np.ndarray,
        pn: np.ndarray,
        sync_bits: np.ndarray,
        search_range_samples: int = 1024,
        step: int = 64
    ) -> int:
        """Finds sample offset maximizing sync correlation peak."""
        sync_len = len(sync_bits)
        sync_bipolar = np.where(sync_bits == 1, 1.0, -1.0)
        max_search = min(search_range_samples, len(audio) - sync_len * self.chip_length)
        if max_search <= 0:
            return 0

        best_offset = 0
        max_score = -float("inf")

        for offset in range(0, max_search, step):
            score = 0.0
            for k in range(sync_len):
                s = offset + k * self.chip_length
                e = s + self.chip_length
                corr = np.dot(audio[s:e], pn)
                score += corr * sync_bipolar[k]

            if score > max_score:
                max_score = score
                best_offset = offset

        # Fine refinement around best_offset
        fine_start = max(0, best_offset - step)
        fine_end = min(max_search, best_offset + step)
        refined_offset = best_offset

        for offset in range(fine_start, fine_end, 8):
            score = 0.0
            for k in range(sync_len):
                s = offset + k * self.chip_length
                e = s + self.chip_length
                corr = np.dot(audio[s:e], pn)
                score += corr * sync_bipolar[k]

            if score > max_score:
                max_score = score
                refined_offset = offset

        return refined_offset
