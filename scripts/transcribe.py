# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "qwen-asr",
#     "torchaudio",
# ]
# ///
"""
Voicebox built-in transcription using Qwen3-ASR.
Supports audio files (wav, mp3, flac, etc.) and video files (mp4, mkv, mov, etc.)

Usage:
    uv run transcribe.py <file> [--language <lang>]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv"}

MODEL_SIZES = {
    "Qwen/Qwen3-ASR-1.7B": "~3.5GB",
    "Qwen/Qwen3-ASR-0.6B": "~1.5GB",
}


def is_model_cached(model_id):
    """Check if a HuggingFace model is already downloaded."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_dir / folder_name / "snapshots"
    return model_dir.exists() and any(model_dir.iterdir())


def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-y", tmp.name],
            check=True, capture_output=True,
        )
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to extract audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video for voicebox")
    parser.add_argument("file", help="Path to audio or video file")
    parser.add_argument("--language", "-l", default=None, help="Language code (auto-detect if omitted)")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-ASR-1.7B", help="ASR model")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Handle video files
    audio_path = args.file
    temp_audio = None
    ext = os.path.splitext(args.file.lower())[1]
    if ext in VIDEO_EXTENSIONS:
        print("Extracting audio from video...", file=sys.stderr)
        temp_audio = extract_audio_from_video(args.file)
        audio_path = temp_audio

    try:
        # Detect device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        from qwen_asr import Qwen3ASRModel

        if not is_model_cached(args.model):
            size = MODEL_SIZES.get(args.model, "~3GB")
            print(f"First-time setup: downloading {args.model} ({size})...", file=sys.stderr)
            print("This is a one-time download — future runs will be instant.", file=sys.stderr)
        else:
            print(f"Loading ASR model on {device}...", file=sys.stderr)

        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=torch.float32,
            device_map=device,
        )

        print("Transcribing...", file=sys.stderr)
        results = model.transcribe(audio=audio_path, language=args.language)
        # Print only the text to stdout (stderr has status messages)
        print(results[0].text)
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)


if __name__ == "__main__":
    main()
