"""
Echo Hiding Steganography Engine for GhostNote.
Embeds secret bits by adding imperceptible micro-delayed echoes below the human
auditory temporal integration threshold (Haas effect, ~0.8-1.5ms), decoded via
real cepstral prominence peak detection with localized background baseline subtraction.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import fft

from ghostnote.crypto import SYNC_BITS_32


class EchoHidingEngine:
    """
    Embeds and extracts binary streams into audio carriers using Echo Hiding
    with cepstral prominence detection and adaptive repetition.
    """

    def __init__(
        self,
        frame_length: int = 2048,
        delay_0_samples: int = 50,  # ~1.13 ms at 44.1 kHz
        delay_1_samples: int = 90,  # ~2.04 ms at 44.1 kHz
        base_alpha: float = 0.22,   # Echo attenuation factor (~ -13 dB)
        window_size: int = 8,       # Background baseline search window
        repetition: int = 1,        # 1x for maximum capacity
    ):
        self.frame_length = frame_length
        self.delay_0 = delay_0_samples
        self.delay_1 = delay_1_samples
        self.base_alpha = base_alpha
        self.w = window_size
        self.repetition = max(1, repetition)

    def calculate_capacity_bits(self, num_samples: int) -> int:
        """Returns the maximum number of logical bits that can fit in the audio."""
        raw_frames = num_samples // self.frame_length
        return raw_frames // self.repetition

    def embed(
        self,
        audio: np.ndarray,
        bits: np.ndarray,
        key: str = "GhostNoteDefaultKey"
    ) -> np.ndarray:
        """
        Embeds a 1D bitstream {0, 1} into carrier audio using Echo Hiding.
        Preserves stereo if audio is 2D by embedding into the primary channel.
        """
        is_stereo = audio.ndim == 2
        carrier = audio[:, 0].copy() if is_stereo else audio.copy()
        num_samples = len(carrier)

        # Apply repetition coding if enabled
        if self.repetition > 1:
            raw_bits = np.repeat(bits, self.repetition)
        else:
            raw_bits = bits

        num_raw_bits = len(raw_bits)
        required_samples = num_raw_bits * self.frame_length
        if required_samples > num_samples:
            max_bytes = self.calculate_capacity_bits(num_samples) // 8
            raise ValueError(
                f"Audio carrier is too short for Echo Hiding! "
                f"Carrier capacity is {self.calculate_capacity_bits(num_samples)} bits ({max_bytes} bytes), "
                f"but encrypted payload requires {len(bits)} bits ({len(bits) // 8} bytes)."
            )

        stego = carrier.astype(np.float64)

        for i, bit in enumerate(raw_bits):
            start = i * self.frame_length
            end = start + self.frame_length

            delay = self.delay_1 if bit == 1 else self.delay_0

            # Form delayed echo segment
            echo = np.zeros(self.frame_length, dtype=np.float64)
            for n in range(self.frame_length):
                src_idx = start + n - delay
                if src_idx >= 0:
                    echo[n] = carrier[src_idx]

            # Add attenuated echo to carrier frame
            stego[start:end] += self.base_alpha * echo

        # Prevent digital clipping
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
        Extracts embedded bits from carrier audio using real cepstral prominence peak detection.
        """
        if audio.ndim == 2:
            signal_data = audio[:, 0]
        else:
            signal_data = audio

        num_samples = len(signal_data)
        total_possible_frames = num_samples // self.frame_length

        if expected_num_bits is not None:
            raw_bits_to_read = min(expected_num_bits * self.repetition, total_possible_frames)
        else:
            raw_bits_to_read = total_possible_frames

        raw_extracted = np.zeros(raw_bits_to_read, dtype=np.uint8)
        win = np.hanning(self.frame_length)

        for i in range(raw_bits_to_read):
            start = i * self.frame_length
            end = start + self.frame_length
            frame = signal_data[start:end]

            spectrum = fft.rfft(frame * win)
            power_spec = np.abs(spectrum) ** 2 + 1e-12
            log_power = np.log(power_spec)
            cepstrum = fft.irfft(log_power, n=self.frame_length)

            # Prominence at delay_0 (subtract local neighborhood median excluding delay_0 itself)
            idx_0 = [k for k in range(self.delay_0 - self.w, self.delay_0 + self.w + 1) if k != self.delay_0 and 0 < k < len(cepstrum) // 2]
            bg_0 = np.median(cepstrum[idx_0]) if idx_0 else 0.0
            prom_0 = cepstrum[self.delay_0] - bg_0

            # Prominence at delay_1 (subtract local neighborhood median excluding delay_1 itself)
            idx_1 = [k for k in range(self.delay_1 - self.w, self.delay_1 + self.w + 1) if k != self.delay_1 and 0 < k < len(cepstrum) // 2]
            bg_1 = np.median(cepstrum[idx_1]) if idx_1 else 0.0
            prom_1 = cepstrum[self.delay_1] - bg_1

            raw_extracted[i] = 1 if prom_1 > prom_0 else 0

        # Majority voting decoding across repetition factor
        if self.repetition > 1:
            num_blocks = len(raw_extracted) // self.repetition
            reshaped = raw_extracted[:num_blocks * self.repetition].reshape(num_blocks, self.repetition)
            decoded = (np.sum(reshaped, axis=1) >= (self.repetition / 2)).astype(np.uint8)
            return decoded

        return raw_extracted
