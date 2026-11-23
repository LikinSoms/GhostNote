# GhostNote

GhostNote is an audio steganography project with a command line interface and a FastAPI web application. It supports Echo Hiding and DSSS embedding, with AES-GCM encryption and audio analysis to help select an embedding method.

## Requirements

Python 3.10 or newer is recommended. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Run the command line interface

```powershell
python cli.py --help
```

## Run the web application

```powershell
python -m uvicorn ghostnote.web.app:app --reload
```

Then open <http://127.0.0.1:8000>.

## Run tests

```powershell
python -m pytest
```
