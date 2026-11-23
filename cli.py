#!/usr/bin/env python3
"""
GhostNote CLI - Intelligent Audio Steganography with Psychoacoustic Algorithm Selection.
"""

import sys
from pathlib import Path
import click
import numpy as np

from ghostnote.engine import StegoEngine
from ghostnote.audio_io import load_audio, save_audio


def render_bar(score: float, width: int = 24) -> str:
    """Renders a visual ASCII progress bar."""
    filled = int(round((score / 100.0) * width))
    filled = max(0, min(width, filled))
    return f"[{'#' * filled}{'-' * (width - filled)}] {score:5.1f}%"


@click.group(help="GhostNote: Audio Steganography with Echo Hiding & DSSS")
def main():
    pass


@main.command("analyze", help="Analyze audio file to find optimal algorithm (Echo Hiding vs DSSS)")
@click.argument("audio_path", type=click.Path(exists=True))
def analyze_cmd(audio_path):
    """Run psychoacoustic analysis on an audio file."""
    engine = StegoEngine()
    click.secho(f"\nAnalyzing '{audio_path}'...", fg="cyan", bold=True)
    
    try:
        audio_data, analysis = engine.analyze_audio(audio_path)
    except Exception as e:
        click.secho(f"Error analyzing audio: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo("\n" + "=" * 62)
    click.secho("  AUDIO PROFILE", fg="white", bold=True)
    click.echo("-" * 62)
    click.echo(f"  Format:           {audio_data.format}")
    click.echo(f"  Duration:         {audio_data.duration_seconds:.2f} seconds")
    click.echo(f"  Sample Rate:      {audio_data.sample_rate:,} Hz")
    click.echo(f"  Channels:         {audio_data.channels} ({'Stereo' if audio_data.channels == 2 else 'Mono'})")
    
    click.echo("\n" + "=" * 62)
    click.secho("  PSYCHOACOUSTIC METRICS", fg="white", bold=True)
    click.echo("-" * 62)
    click.echo(f"  Spectral Flatness:       {analysis.spectral_flatness:.4f}  (0=pure tone, 1=white noise)")
    click.echo(f"  Spectral Entropy:        {analysis.spectral_entropy:.4f}  (distribution uniformity)")
    click.echo(f"  Spectral Crest Factor:   {analysis.spectral_crest:.2f}  (peak to average power)")
    click.echo(f"  Tonality Index:          {analysis.tonality_index:.4f}  (harmonic concentration)")
    click.echo(f"  Transient Density:       {analysis.transient_density:.2f} /sec  (onset spikes)")
    click.echo(f"  Zero Crossing Rate:      {analysis.zero_crossing_rate:.4f}")
    click.echo(f"  Dynamic Range:           {analysis.dynamic_range_db:.1f} dB")
    click.echo(f"  Cepstral Host Risk:      {analysis.cepstral_interference:.3f}  (native echo conflict)")
    click.echo(f"  Silence Ratio:           {analysis.silence_ratio * 100:.1f}%")

    click.echo("\n" + "=" * 62)
    click.secho("  ALGORITHM SUITABILITY SCORES", fg="white", bold=True)
    click.echo("-" * 62)
    
    echo_color = "green" if analysis.echo_score >= analysis.dsss_score else "yellow"
    dsss_color = "green" if analysis.dsss_score > analysis.echo_score else "yellow"
    
    click.echo(f"  Echo Hiding:  ", nl=False)
    click.secho(render_bar(analysis.echo_score), fg=echo_color, bold=True)
    
    click.echo(f"  DSSS:         ", nl=False)
    click.secho(render_bar(analysis.dsss_score), fg=dsss_color, bold=True)

    click.echo("\n" + "=" * 62)
    click.secho(f"  RECOMMENDED ALGORITHM: {analysis.recommended_algorithm.upper()} ({analysis.confidence} Confidence)", fg="green", bold=True)
    click.echo("-" * 62)
    click.echo("  Analysis Findings:")
    for reason in analysis.reasons:
        click.echo(f"   * {reason}")
    
    click.echo("\n  Payload Capacity Estimates:")
    for algo, cap in analysis.capacity_estimate_bytes.items():
        click.echo(f"   * {algo}: ~{cap} bytes (~{cap * 8} bits)")
    click.echo("=" * 62 + "\n")


@main.command("embed", help="Encrypt and hide secret text inside audio file (.wav or .mp3)")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-m", "--message", prompt=True, help="Secret text message to encrypt and hide")
@click.option("-p", "--password", prompt=True, hide_input=True, confirmation_prompt=True, help="Passphrase for AES-256 encryption")
@click.option("-o", "--output", default="stego_output.wav", help="Output file path (.wav or .mp3)")
@click.option("--algo", type=click.Choice(["auto", "echo", "dsss"], case_sensitive=False), default="auto", help="Algorithm choice ('auto' uses audio analysis)")
def embed_cmd(audio_path, message, password, output, algo):
    """Encrypt and embed text into audio."""
    engine = StegoEngine()
    click.secho(f"\nProcessing carrier audio: '{audio_path}'...", fg="cyan")

    try:
        res = engine.embed(
            audio_path=audio_path,
            message=message,
            passphrase=password,
            output_path=output,
            algorithm=algo,
        )
    except Exception as e:
        click.secho(f"\nEmbedding failed: {e}", fg="red", bold=True, err=True)
        sys.exit(1)

    click.secho("\n[SUCCESS] Secret message encrypted and embedded successfully!", fg="green", bold=True)
    click.echo("-" * 62)
    click.echo(f"  Output Stego File:     {res.output_path}")
    click.echo(f"  Algorithm Used:        {res.algorithm_used}")
    click.echo(f"  Encryption:            AES-256-GCM + PBKDF2-SHA256")
    click.echo(f"  Message Length:        {res.message_length} characters")
    click.echo(f"  Encrypted Payload:     {res.payload_bytes} bytes")
    click.echo(f"  Signal-to-Noise Ratio: {res.audio_metrics['snr_db']} dB")
    click.echo(f"  Peak SNR (PSNR):       {res.audio_metrics['psnr_db']} dB")
    click.echo(f"  Perceptual Quality:    {res.audio_metrics['perceptual_distortion']}")
    click.echo("-" * 62 + "\n")


@main.command("extract", help="Decrypt and extract secret text from stego audio file")
@click.argument("stego_path", type=click.Path(exists=True))
@click.option("-p", "--password", prompt=True, hide_input=True, help="Passphrase used during encryption")
@click.option("--algo", type=click.Choice(["auto", "echo", "dsss"], case_sensitive=False), default="auto", help="Algorithm hint")
def extract_cmd(stego_path, password, algo):
    """Extract and decrypt secret message from stego audio."""
    engine = StegoEngine()
    click.secho(f"\nExtracting hidden payload from '{stego_path}'...", fg="cyan")

    hint = None if algo == "auto" else algo
    res = engine.extract(stego_path, passphrase=password, algorithm_hint=hint)

    if not res.success:
        click.secho(f"\n[EXTRACTION FAILED] {res.details}", fg="red", bold=True, err=True)
        sys.exit(1)

    click.secho("\n[SUCCESS] Secret message extracted and decrypted!", fg="green", bold=True)
    click.echo("-" * 62)
    click.echo(f"  Algorithm Detected:    {res.algorithm}")
    click.echo(f"  Ciphertext Size:       {res.ciphertext_len} bytes")
    click.echo("-" * 62)
    click.secho("  DECRYPTED MESSAGE:", fg="white", bold=True)
    click.secho(f"\n{res.plaintext}\n", fg="green", bold=True)
    click.echo("-" * 62 + "\n")


@main.command("generate-sample", help="Generate synthetic sample audio (tonal harmonic vs noisy broadband)")
@click.option("--type", "audio_type", type=click.Choice(["tonal", "noisy", "percussive", "speech"]), default="tonal")
@click.option("-o", "--output", default=None, help="Output filename (.wav or .mp3)")
@click.option("-d", "--duration", default=32.0, type=float, help="Duration in seconds (default: 32.0s)")
def generate_sample_cmd(audio_type, output, duration):
    """Generate sample audio clips for testing steganography algorithms."""
    sr = 44100
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    if audio_type == "tonal":
        # Multi-harmonic acoustic chord (favors Echo Hiding)
        sig = (0.35 * np.sin(2 * np.pi * 261.63 * t) +   # C4
               0.25 * np.sin(2 * np.pi * 329.63 * t) +   # E4
               0.20 * np.sin(2 * np.pi * 392.00 * t) +   # G4
               0.15 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)    # C5
        sig *= np.exp(-t * 0.05)  # slight decay
        sig += 0.02 * np.random.normal(0, 0.2, num_samples).astype(np.float32)
        default_name = "sample_tonal.wav"

    elif audio_type == "noisy":
        # Broadband noise with texture (favors DSSS)
        sig = np.random.normal(0, 0.3, num_samples).astype(np.float32)
        default_name = "sample_noisy.wav"

    elif audio_type == "speech":
        # Formant-synthesized speech-like vowels
        sig = np.zeros(num_samples)
        f0 = 130.0  # Fundamental frequency
        # Formants for vowel /a/ ~ 700Hz, 1200Hz, 2500Hz
        for harm in range(1, 35):
            freq = harm * f0
            # Envelope shaping
            gain = (1.0 / harm) * (
                np.exp(-((freq - 700) / 120)**2) * 2.5 +
                np.exp(-((freq - 1220) / 150)**2) * 1.8 +
                np.exp(-((freq - 2600) / 250)**2) * 0.9 + 0.1
            )
            sig += gain * np.sin(2 * np.pi * freq * t)
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.6
        default_name = "sample_speech.wav"

    elif audio_type == "percussive":
        # Repeated transient beats
        sig = np.zeros(num_samples)
        bpm = 120
        beat_interval = int(sr * 60 / bpm)
        for beat in range(0, num_samples, beat_interval):
            burst_len = min(4000, num_samples - beat)
            decay = np.exp(-np.linspace(0, 8, burst_len))
            noise_hit = np.random.normal(0, 0.5, burst_len) * decay
            sig[beat:beat + burst_len] += noise_hit
        default_name = "sample_percussive.wav"

    out_file = output or default_name
    save_audio(out_file, sig.astype(np.float32), sr)
    click.secho(f"Generated {duration}s '{audio_type}' sample audio -> {out_file}", fg="green", bold=True)


@main.command("ui", help="Launch interactive web browser dashboard")
@click.option("--port", default=8000, type=int, help="Port to bind to (default: 8000)")
@click.option("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
def ui_cmd(host, port):
    """Start the GhostNote Web UI server."""
    import uvicorn
    click.secho(f"\nStarting GhostNote Web UI on http://{host}:{port} ...", fg="green", bold=True)
    uvicorn.run("ghostnote.web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
