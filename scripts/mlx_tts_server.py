#!/usr/bin/env python3
"""Standalone MLX TTS server — lightweight HTTP wrapper around mlx-audio.
Starts on port 8765, loads models on first use, keeps them warm.
Designed to fill in when EnConvo is not running."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-audio>=0.2.0", "soundfile>=0.13.0"]
# ///

import json
import os
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = int(os.environ.get("MLX_TTS_PORT", "8765"))
PID_FILE = Path("/tmp/mlx_tts_server.pid")

# Cache loaded models
_models = {}


def get_model(model_id):
    if model_id not in _models:
        from mlx_audio.tts.utils import load_model
        print(f"[mlx-tts] Loading {model_id}...")
        _models[model_id] = load_model(model_id)
        print(f"[mlx-tts] Ready: {model_id}")
    return _models[model_id]


class TTSHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "port": PORT}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/tts":
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            text = body.get("input_text", "")
            model_id = body.get("model_id", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
            voice = body.get("voice", "Chelsie")
            instruct = body.get("instruct")

            model = get_model(model_id)

            # Determine generation mode
            if "CustomVoice" in model_id:
                kwargs = {"text": text, "speaker": voice}
                if instruct:
                    kwargs["instruct"] = instruct
                results = list(model.generate_custom_voice(**kwargs))
            elif "VoiceDesign" in model_id:
                results = list(model.generate_voice_design(
                    text=text,
                    language="English",
                    instruct=instruct or "A calm, clear narrator voice",
                ))
            else:
                results = list(model.generate(text=text))

            # Collect audio
            import numpy as np
            audio_parts = []
            sample_rate = 24000
            for r in results:
                if hasattr(r, "audio"):
                    a = r.audio
                    if hasattr(a, "numpy"):
                        a = a.numpy()
                    audio_parts.append(a.flatten())
                    if hasattr(r, "sample_rate"):
                        sample_rate = r.sample_rate

            if not audio_parts:
                raise ValueError("No audio generated")

            audio = np.concatenate(audio_parts)
            out_path = "/tmp/mlx_tts_server_output.wav"

            import soundfile as sf
            sf.write(out_path, audio, sample_rate)
            duration = len(audio) / sample_rate

            result = {"audio_path": out_path, "sample_rate": sample_rate, "duration": round(duration, 2)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    def cleanup(sig, frame):
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    server = HTTPServer(("127.0.0.1", PORT), TTSHandler)
    print(f"[mlx-tts] Server running on http://127.0.0.1:{PORT}")
    print(f"[mlx-tts] PID: {os.getpid()}")
    try:
        server.serve_forever()
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
