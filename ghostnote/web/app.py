"""
FastAPI Web Application Backend for GhostNote.
Provides REST API endpoints and web interface for audio analysis,
encryption & embedding, decryption & extraction, and sample generation.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ghostnote.engine import StegoEngine
from ghostnote.audio_io import load_audio, save_audio
from ghostnote.analyzer import AudioAnalyzer

app = FastAPI(
    title="GhostNote Audio Steganography",
    description="Intelligent audio steganography using Echo Hiding and DSSS with AES-256 encryption.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "ghostnote_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "ghostnote_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = StegoEngine()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main single-page web app."""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>GhostNote UI Loading...</h1>")


@app.post("/api/analyze")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """Uploads and analyzes audio to recommend Echo Hiding or DSSS."""
    suffix = Path(file.filename).suffix or ".wav"
    temp_in = UPLOAD_DIR / f"upload_{os.urandom(6).hex()}{suffix}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_data, analysis = engine.analyze_audio(temp_in)

        return {
            "success": True,
            "filename": file.filename,
            "profile": {
                "format": audio_data.format,
                "duration_seconds": round(audio_data.duration_seconds, 2),
                "sample_rate": audio_data.sample_rate,
                "channels": audio_data.channels,
            },
            "scores": {
                "echo_score": analysis.echo_score,
                "dsss_score": analysis.dsss_score,
                "recommended_algorithm": analysis.recommended_algorithm,
                "confidence": analysis.confidence,
            },
            "metrics": {
                "spectral_flatness": analysis.spectral_flatness,
                "spectral_entropy": analysis.spectral_entropy,
                "spectral_crest": analysis.spectral_crest,
                "tonality_index": analysis.tonality_index,
                "transient_density": analysis.transient_density,
                "zero_crossing_rate": analysis.zero_crossing_rate,
                "dynamic_range_db": analysis.dynamic_range_db,
                "cepstral_interference": analysis.cepstral_interference,
                "silence_ratio": analysis.silence_ratio,
            },
            "reasons": analysis.reasons,
            "capacity": analysis.capacity_estimate_bytes,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/embed")
async def embed_endpoint(
    file: UploadFile = File(...),
    message: str = Form(...),
    password: str = Form(...),
    algorithm: str = Form("auto"),
    output_format: str = Form("wav"),
):
    """Encrypts message and embeds it into audio."""
    if not message.strip():
        raise HTTPException(status_code=400, detail="Secret message cannot be empty.")
    if not password:
        raise HTTPException(status_code=400, detail="Passphrase cannot be empty.")

    suffix = Path(file.filename).suffix or ".wav"
    temp_in = UPLOAD_DIR / f"embed_in_{os.urandom(6).hex()}{suffix}"
    out_ext = ".mp3" if output_format.lower() == "mp3" else ".wav"
    out_filename = f"stego_{os.urandom(8).hex()}{out_ext}"
    temp_out = OUTPUT_DIR / out_filename

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = engine.embed(
            audio_path=temp_in,
            message=message,
            passphrase=password,
            output_path=temp_out,
            algorithm=algorithm,
        )

        return {
            "success": True,
            "algorithm_used": result.algorithm_used,
            "message_length": result.message_length,
            "payload_bytes": result.payload_bytes,
            "audio_metrics": result.audio_metrics,
            "download_url": f"/api/download/{out_filename}",
            "filename": out_filename,
            "details": result.details,
            "analysis": {
                "echo_score": result.analysis.echo_score,
                "dsss_score": result.analysis.dsss_score,
                "recommended": result.analysis.recommended_algorithm,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    password: str = Form(...),
    algorithm_hint: Optional[str] = Form(None),
):
    """Extracts and decrypts secret message from stego audio."""
    if not password:
        raise HTTPException(status_code=400, detail="Passphrase required for decryption.")

    suffix = Path(file.filename).suffix or ".wav"
    temp_in = UPLOAD_DIR / f"extract_in_{os.urandom(6).hex()}{suffix}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        hint = None if not algorithm_hint or algorithm_hint == "auto" else algorithm_hint
        result = engine.extract(temp_in, passphrase=password, algorithm_hint=hint)

        if not result.success:
            return {
                "success": False,
                "details": result.details,
            }

        return {
            "success": True,
            "plaintext": result.plaintext,
            "algorithm": result.algorithm,
            "ciphertext_len": result.ciphertext_len,
            "details": result.details,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.get("/api/sample/{sample_type}")
async def get_sample_endpoint(sample_type: str, duration: float = 32.0):
    """Generates synthetic sample audio for instant testing."""
    sr = 44100
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    sample_key = sample_type.lower()
    if sample_key == "tonal":
        sig = (0.35 * np.sin(2 * np.pi * 261.63 * t) +
               0.25 * np.sin(2 * np.pi * 329.63 * t) +
               0.20 * np.sin(2 * np.pi * 392.00 * t) +
               0.15 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
        sig *= np.exp(-t * 0.05)
        # Add realistic acoustic room texture
        sig += 0.02 * np.random.normal(0, 0.2, num_samples).astype(np.float32)
        fname = "sample_tonal.wav"
    elif sample_key == "noisy":
        sig = np.random.normal(0, 0.35, num_samples).astype(np.float32)
        fname = "sample_noisy.wav"
    elif sample_key == "speech":
        sig = np.zeros(num_samples, dtype=np.float32)
        f0 = 130.0
        for harm in range(1, 35):
            freq = harm * f0
            gain = (1.0 / harm) * (
                np.exp(-((freq - 700) / 120)**2) * 2.5 +
                np.exp(-((freq - 1220) / 150)**2) * 1.8 +
                np.exp(-((freq - 2600) / 250)**2) * 0.9 + 0.1
            )
            sig += gain * np.sin(2 * np.pi * freq * t)
        sig = (sig / (np.max(np.abs(sig)) + 1e-6) * 0.6).astype(np.float32)
        fname = "sample_speech.wav"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sample type: {sample_type}")

    out_file = OUTPUT_DIR / f"{sample_key}_{os.urandom(4).hex()}.wav"
    save_audio(out_file, sig, sr, subtype="PCM_24")

    return FileResponse(
        path=out_file,
        media_type="audio/wav",
        filename=fname,
    )


@app.get("/api/download/{filename}")
async def download_file_endpoint(filename: str):
    """Downloads or streams an audio file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired.")

    media_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )
