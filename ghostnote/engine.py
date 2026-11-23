"""
Unified Steganography Engine for GhostNote.
Coordinates audio analysis, automatic algorithm selection (Echo Hiding vs DSSS),
AES-256 encryption, embedding, and tamper-evident extraction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

from ghostnote.analyzer import AudioAnalyzer, AudioAnalysisResult
from ghostnote.audio_io import AudioData, load_audio, save_audio, compute_audio_metrics
from ghostnote.crypto import (
    encrypt_message,
    decrypt_message,
    bytes_to_bits,
    bits_to_bytes,
    locate_sync_sequence,
    DecryptedPayload,
)
from ghostnote.dsss_stego import DSSSEngine
from ghostnote.echo_stego import EchoHidingEngine


@dataclass
class EmbedResult:
    """Outcome of an embedding operation."""
    success: bool
    output_path: Path
    algorithm_used: str
    passphrase_used: str
    message_length: int
    payload_bytes: int
    audio_metrics: dict
    analysis: AudioAnalysisResult
    details: str = ""


@dataclass
class ExtractResult:
    """Outcome of an extraction operation."""
    success: bool
    plaintext: str
    algorithm: str
    ciphertext_len: int
    details: str = ""


class StegoEngine:
    """
    Main orchestrator for GhostNote steganography operations.
    """

    def __init__(
        self,
        dsss_engine: Optional[DSSSEngine] = None,
        echo_engine: Optional[EchoHidingEngine] = None,
        analyzer: Optional[AudioAnalyzer] = None,
    ):
        self.dsss_engine = dsss_engine or DSSSEngine(
            chip_length=1024,
            target_margin=0.040,
            max_delta=0.22,
        )
        self.echo_engine = echo_engine or EchoHidingEngine(
            frame_length=2048,
            delay_0_samples=50,
            delay_1_samples=90,
            base_alpha=0.22,
            repetition=1,
        )
        self.analyzer = analyzer or AudioAnalyzer()

    def analyze_audio(self, audio_path: Union[str, Path]) -> Tuple[AudioData, AudioAnalysisResult]:
        """Loads audio and returns psychoacoustic suitability analysis."""
        audio_data = load_audio(audio_path)
        analysis = self.analyzer.analyze(audio_data.mono, audio_data.sample_rate)
        return audio_data, analysis

    def embed(
        self,
        audio_path: Union[str, Path],
        message: str,
        passphrase: str,
        output_path: Union[str, Path],
        algorithm: str = "auto",  # "auto", "echo", or "dsss"
    ) -> EmbedResult:
        """
        Analyzes audio (if algorithm='auto'), encrypts the message with AES-256-GCM,
        embeds it using the selected algorithm, and writes the output file (.wav or .mp3).
        """
        audio_data, analysis = self.analyze_audio(audio_path)
        original_samples = audio_data.samples.copy()
        
        # Estimate approximate payload size for capacity routing
        # (AES-GCM adds ~57 bytes overhead + 32-bit sync = 61 bytes = 488 bits + text bits)
        approx_needed_bits = (len(message.encode("utf-8")) + 61) * 8
        echo_cap_bits = self.echo_engine.calculate_capacity_bits(len(original_samples))
        dsss_cap_bits = self.dsss_engine.calculate_capacity_bits(len(original_samples))

        algo_choice = algorithm.strip().lower()
        if algo_choice in ("auto", "automatic"):
            chosen_algo = analysis.recommended_algorithm
            # If Echo Hiding is acoustically preferred but message exceeds capacity while DSSS fits, adapt to DSSS
            if chosen_algo == "Echo Hiding" and approx_needed_bits > echo_cap_bits:
                if approx_needed_bits <= dsss_cap_bits:
                    chosen_algo = "DSSS"
        elif algo_choice in ("echo", "echo hiding"):
            chosen_algo = "Echo Hiding"
        elif algo_choice in ("dsss", "spread spectrum"):
            chosen_algo = "DSSS"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'auto', 'echo', or 'dsss'.")

        # Encrypt plaintext message with AES-256-GCM
        packet = encrypt_message(message, passphrase, chosen_algo)
        payload_bits = bytes_to_bits(packet, include_sync=True)

        # Select engine and embed
        if chosen_algo == "Echo Hiding":
            stego_samples = self.echo_engine.embed(original_samples, payload_bits, passphrase)
        else:
            stego_samples = self.dsss_engine.embed(original_samples, payload_bits, passphrase)

        # Compute imperceptibility metrics
        metrics = compute_audio_metrics(original_samples, stego_samples)

        # Save stego file
        out_p = Path(output_path)
        saved_path = save_audio(out_p, stego_samples, audio_data.sample_rate)

        return EmbedResult(
            success=True,
            output_path=saved_path,
            algorithm_used=chosen_algo,
            passphrase_used=passphrase,
            message_length=len(message),
            payload_bytes=len(packet),
            audio_metrics=metrics,
            analysis=analysis,
            details=(
                f"Successfully embedded {len(message)} chars ({len(packet)} bytes encrypted) "
                f"using {chosen_algo}. SNR: {metrics['snr_db']} dB ({metrics['perceptual_distortion']})."
            ),
        )

    def extract(
        self,
        stego_path: Union[str, Path],
        passphrase: str,
        algorithm_hint: Optional[str] = None,
    ) -> ExtractResult:
        """
        Extracts and decrypts secret message from stego audio file.
        Attempts both Echo Hiding and DSSS if no hint is provided or if hint fails.
        """
        audio_data = load_audio(stego_path)
        samples = audio_data.samples

        # Determine attempt order
        algorithms_to_try = []
        if algorithm_hint:
            hint = algorithm_hint.strip().lower()
            if "echo" in hint:
                algorithms_to_try = [("Echo Hiding", self.echo_engine), ("DSSS", self.dsss_engine)]
            elif "dsss" in hint:
                algorithms_to_try = [("DSSS", self.dsss_engine), ("Echo Hiding", self.echo_engine)]
        
        if not algorithms_to_try:
            # Try both: DSSS then Echo Hiding
            algorithms_to_try = [("DSSS", self.dsss_engine), ("Echo Hiding", self.echo_engine)]

        last_error = ""
        for algo_name, engine in algorithms_to_try:
            try:
                extracted_bits = engine.extract(samples, key=passphrase)
                sync_idx = locate_sync_sequence(extracted_bits)
                if sync_idx is None:
                    continue

                raw_bytes = bits_to_bytes(extracted_bits[sync_idx + 32:])
                decrypted = decrypt_message(raw_bytes, passphrase)
                
                return ExtractResult(
                    success=True,
                    plaintext=decrypted.plaintext,
                    algorithm=decrypted.algorithm,
                    ciphertext_len=decrypted.ciphertext_len,
                    details=f"Extracted and decrypted successfully via {decrypted.algorithm}.",
                )
            except Exception as exc:
                last_error = str(exc)
                continue

        return ExtractResult(
            success=False,
            plaintext="",
            algorithm="Unknown",
            ciphertext_len=0,
            details=f"Failed to extract or decrypt payload. Ensure passphrase is correct. Last error: {last_error}",
        )
