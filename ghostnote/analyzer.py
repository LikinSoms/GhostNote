"""
Psychoacoustic Audio Analyzer for GhostNote.
Extracts spectral and temporal features to determine whether Echo Hiding
or Direct Sequence Spread Spectrum (DSSS) is optimal for a given audio file.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from scipy import signal, fft


@dataclass
class AudioAnalysisResult:
    """Detailed results of the psychoacoustic and spectral audio analysis."""
    recommended_algorithm: str  # "Echo Hiding" or "DSSS"
    confidence: str            # "High", "Moderate", "Marginal"
    echo_score: float          # 0 - 100
    dsss_score: float          # 0 - 100
    spectral_flatness: float   # 0 (pure tone) to 1 (pure noise)
    spectral_entropy: float    # 0 to 1
    spectral_crest: float      # Peak-to-average spectral ratio
    tonality_index: float      # Estimated tonality (0 to 1)
    transient_density: float   # Transients detected per second
    zero_crossing_rate: float  # Mean ZCR
    dynamic_range_db: float    # Dynamic range in dB
    cepstral_interference: float  # Host cepstral peak risk for echo delays
    silence_ratio: float       # Proportion of near-silent frames (< -45 dB)
    duration_seconds: float
    sample_rate: int
    reasons: List[str] = field(default_factory=list)
    capacity_estimate_bytes: Dict[str, int] = field(default_factory=dict)


class AudioAnalyzer:
    """
    Evaluates acoustic characteristics of an audio signal to recommend
    either Echo Hiding or DSSS based on psychoacoustic masking theory.
    """

    def __init__(self, frame_size: int = 2048, hop_size: int = 512):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def analyze(self, audio: np.ndarray, sample_rate: int) -> AudioAnalysisResult:
        """
        Run full psychoacoustic analysis on a 1D audio waveform.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sample_rate
        if duration < 0.2:
            raise ValueError(f"Audio is too short ({duration:.2f}s). Minimum 0.2s required.")

        # Compute STFT for time-frequency analysis
        f, t, zxx = signal.stft(
            audio,
            fs=sample_rate,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window="hann",
        )
        magnitude = np.abs(zxx) + 1e-12
        power = magnitude ** 2

        # 1. Spectral Flatness Measure (SFM)
        # SFM = Geometric Mean / Arithmetic Mean of power spectrum
        log_power = np.log(power)
        geo_mean = np.exp(np.mean(log_power, axis=0))
        arith_mean = np.mean(power, axis=0)
        sfm_per_frame = geo_mean / (arith_mean + 1e-12)
        mean_sfm = float(np.mean(sfm_per_frame))
        mean_sfm = max(0.0, min(1.0, mean_sfm))

        # 2. Spectral Entropy
        # Shannon entropy of normalized spectral distribution
        norm_power = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)
        entropy_per_frame = -np.sum(norm_power * np.log2(norm_power + 1e-12), axis=0)
        max_entropy = np.log2(power.shape[0])
        spectral_entropy = float(np.mean(entropy_per_frame) / max_entropy)
        spectral_entropy = max(0.0, min(1.0, spectral_entropy))

        # 3. Spectral Crest Factor (Peak to Average Ratio)
        peak_power = np.max(power, axis=0)
        crest_per_frame = peak_power / (arith_mean + 1e-12)
        mean_crest = float(np.mean(crest_per_frame))

        # 4. Tonality Index (derived from SFM and crest factor)
        # Tonality = 1 - SFM (higher for pure tones/harmonics, lower for noise)
        tonality_index = float(np.clip(1.0 - mean_sfm, 0.0, 1.0))

        # 5. Transient Density (Spectral Flux onset detection)
        flux = np.sum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]) ** 2, axis=0)
        flux_thresh = np.mean(flux) + 1.8 * np.std(flux)
        transient_peaks = np.where(flux > flux_thresh)[0]
        transient_density = float(len(transient_peaks) / max(duration, 0.1))

        # 6. Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        mean_zcr = float(zero_crossings)

        # 7. Dynamic Range & Silence Ratio
        frame_len = self.frame_size
        num_frames = len(audio) // frame_len
        if num_frames > 0:
            frames = audio[:num_frames * frame_len].reshape(num_frames, frame_len)
            rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            max_rms = np.max(rms_per_frame)
            silence_thresh = max_rms * 0.005  # -46 dB relative to peak
            silence_ratio = float(np.mean(rms_per_frame < silence_thresh))
            dynamic_range_db = float(20.0 * np.log10(max(max_rms, 1e-6) / (np.min(rms_per_frame) + 1e-6)))
        else:
            silence_ratio = 0.0
            dynamic_range_db = 40.0

        # 8. Cepstral Host Peak Interference Risk
        # For echo delays Delta_0 ~ 80 samples (~1.8ms), Delta_1 ~ 120 samples (~2.7ms)
        cepstral_interference = self._evaluate_cepstral_interference(audio, sample_rate)

        # Compute Comparative Scores
        echo_score, dsss_score, reasons = self._score_algorithms(
            mean_sfm=mean_sfm,
            spectral_entropy=spectral_entropy,
            mean_crest=mean_crest,
            tonality_index=tonality_index,
            transient_density=transient_density,
            dynamic_range_db=dynamic_range_db,
            silence_ratio=silence_ratio,
            cepstral_interference=cepstral_interference,
        )

        score_diff = abs(echo_score - dsss_score)
        if score_diff >= 18:
            confidence = "High"
        elif score_diff >= 8:
            confidence = "Moderate"
        else:
            confidence = "Marginal"

        recommended = "Echo Hiding" if echo_score >= dsss_score else "DSSS"

        # Payload Capacity Estimates (bytes)
        # Echo hiding: ~2048 sample frame at 44.1kHz = ~21.5 bps = ~2.68 bytes/sec
        # DSSS: ~1024 chip length at 44.1kHz = ~43 bps = ~5.38 bytes/sec
        echo_capacity_bytes = max(1, int((duration * sample_rate / 2048) / 8))
        dsss_capacity_bytes = max(1, int((duration * sample_rate / 1024) / 8))

        return AudioAnalysisResult(
            recommended_algorithm=recommended,
            confidence=confidence,
            echo_score=round(echo_score, 1),
            dsss_score=round(dsss_score, 1),
            spectral_flatness=round(mean_sfm, 4),
            spectral_entropy=round(spectral_entropy, 4),
            spectral_crest=round(mean_crest, 2),
            tonality_index=round(tonality_index, 4),
            transient_density=round(transient_density, 2),
            zero_crossing_rate=round(mean_zcr, 4),
            dynamic_range_db=round(dynamic_range_db, 1),
            cepstral_interference=round(cepstral_interference, 4),
            silence_ratio=round(silence_ratio, 3),
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            reasons=reasons,
            capacity_estimate_bytes={
                "Echo Hiding": echo_capacity_bytes,
                "DSSS": dsss_capacity_bytes,
            },
        )

    def _evaluate_cepstral_interference(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculates host cepstral activity around expected echo hiding delay bins.
        If the audio naturally has strong cepstral peaks at ~1.5 - 4ms,
        echo hiding extraction will suffer from host signal interference.
        """
        # Analyze first 5 seconds or full signal
        segment = audio[:min(len(audio), sample_rate * 5)]
        if len(segment) < 4096:
            return 0.1

        # Real cepstrum of segment
        win = np.hanning(len(segment))
        spectrum = fft.rfft(segment * win)
        log_spectrum = np.log(np.abs(spectrum) ** 2 + 1e-12)
        cepstrum = np.abs(fft.irfft(log_spectrum))

        # Echo delay region: 1.0 ms to 4.5 ms
        min_bin = int(0.0010 * sample_rate)
        max_bin = int(0.0045 * sample_rate)
        min_bin = max(1, min_bin)
        max_bin = min(len(cepstrum) // 2, max_bin)

        if max_bin <= min_bin:
            return 0.1

        region = cepstrum[min_bin:max_bin]
        baseline = np.median(cepstrum[min_bin // 2:max_bin * 2]) + 1e-12
        peak_prominence = float(np.max(region) / baseline)

        # Normalize to [0, 1] scale
        risk = float(np.clip((peak_prominence - 1.0) / 10.0, 0.0, 1.0))
        return risk

    def _score_algorithms(
        self,
        mean_sfm: float,
        spectral_entropy: float,
        mean_crest: float,
        tonality_index: float,
        transient_density: float,
        dynamic_range_db: float,
        silence_ratio: float,
        cepstral_interference: float,
    ) -> Tuple[float, float, List[str]]:
        """
        Calculates scores for Echo Hiding vs DSSS based on psychoacoustic rules.
        """
        reasons = []

        # Base starting scores
        echo_score = 50.0
        dsss_score = 50.0

        # 1. Spectral Flatness & Entropy
        # High SFM / high entropy -> broadband/noise-like -> DSSS ideal
        # Low SFM / low entropy -> tonal/harmonic -> Echo Hiding ideal
        if mean_sfm > 0.25:
            dsss_boost = min(28.0, (mean_sfm - 0.25) * 60.0)
            dsss_score += dsss_boost
            echo_score -= dsss_boost * 0.7
            reasons.append(
                f"High spectral flatness ({mean_sfm:.3f}) and entropy indicate broadband/noise-like audio; "
                "DSSS spread-spectrum noise is heavily masked by psychoacoustic frequency masking."
            )
        elif mean_sfm < 0.12:
            echo_boost = min(25.0, (0.12 - mean_sfm) * 100.0)
            echo_score += echo_boost
            dsss_score -= echo_boost * 0.7
            reasons.append(
                f"Low spectral flatness ({mean_sfm:.3f}) and high tonality ({tonality_index:.3f}); "
                "harmonic resonance seamlessly conceals micro-delays (Echo Hiding) like natural room reverberation."
            )

        # 2. Spectral Crest Factor & Tonality
        if mean_crest > 15.0:
            echo_score += 12.0
            dsss_score -= 8.0
            reasons.append(
                f"Pronounced harmonic peaks (Crest Factor: {mean_crest:.1f}); "
                "spread spectrum noise might be noticeable in quiet spectral valleys, favoring Echo Hiding."
            )
        elif mean_crest < 6.0:
            dsss_score += 10.0
            echo_score -= 6.0
            reasons.append(
                f"Flat spectral distribution (Crest Factor: {mean_crest:.1f}); "
                "ideal for Direct Sequence Spread Spectrum (DSSS) matched filter recovery."
            )

        # 3. Transient Density
        # Sharp percussive transients (drums, clicks) cause audible comb-filter flutter with Echo Hiding,
        # but spread-spectrum chips are masked well by transient energy.
        if transient_density > 4.0:
            dsss_score += 14.0
            echo_score -= 12.0
            reasons.append(
                f"Frequent sharp transients ({transient_density:.1f}/sec detected); "
                "Echo Hiding risks audible comb-filtering or pre-echo flutter, whereas DSSS excels."
            )
        elif transient_density < 1.0:
            echo_score += 8.0
            reasons.append(
                f"Smooth temporal progression ({transient_density:.1f} transients/sec); "
                "sustained musical or vocal decay smoothly absorbs echo delays without temporal smearing."
            )

        # 4. Host Cepstral Interference Risk
        if cepstral_interference > 0.4:
            echo_score -= 16.0
            dsss_score += 10.0
            reasons.append(
                f"Significant native cepstral activity detected in the 1-4ms delay range (risk: {cepstral_interference:.2f}); "
                "carrier signal's own echo features could cause decoding errors in Echo Hiding. DSSS is safer."
            )

        # 5. Silence Ratio
        if silence_ratio > 0.35:
            # Silence hurts echo hiding (ringing in dead silence) more than DSSS (which has amplitude gating)
            echo_score -= 10.0
            reasons.append(
                f"High proportion of silence/low energy ({silence_ratio * 100:.1f}%); "
                "discrete echoes may become audible during quiet passages."
            )

        # Normalize scores to [5.0, 95.0]
        echo_score = float(np.clip(echo_score, 5.0, 95.0))
        dsss_score = float(np.clip(dsss_score, 5.0, 95.0))

        return echo_score, dsss_score, reasons
