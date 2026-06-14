#!/usr/bin/env python3
"""
Standalone Kokoro TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Usage:
    python kokoro_worker.py

Commands:
    {"cmd": "init", "model": "mlx-community/Kokoro-82M-bf16", "voice": "af_heart"}
    {"cmd": "generate", "text": "Hello world"}
"""

import sys
import json
import base64
import traceback
import numpy as np

# Add logging to worker
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.voice = None
        
    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice
            # Test generation to ensure everything works
            list(self.model.generate(text="test", voice=voice, speed=1.0))
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, text):
        """Stream audio to stdout one segment at a time.

        Each segment Kokoro yields is emitted immediately as its own JSON line
        ({"segment": <b64 PCM16>}), so the service can start playback after the
        first segment instead of waiting for the whole clause to synthesize.
        A final {"done": true} marks the end; errors emit {"error": ...}.
        Prints directly rather than returning, so the caller must not re-print.
        """
        try:
            if not self.model:
                print(json.dumps({"error": "Not initialized"}), flush=True)
                return

            had_signal = False
            for result in self.model.generate(text=text, voice=self.voice, speed=1.0):
                # Convert MLX array to numpy immediately
                audio_data = np.array(result.audio, copy=True)
                if audio_data.size == 0:
                    continue
                print(
                    f"Generated segment shape: {audio_data.shape}, min: {audio_data.min():.4f}, max: {audio_data.max():.4f}",
                    file=sys.stderr,
                )
                if float(np.max(np.abs(audio_data))) >= 1e-6:
                    had_signal = True

                # Convert to 16-bit PCM and stream this segment out now.
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"segment": audio_b64}), flush=True)

            if not had_signal:
                # No segments, or everything was silent — surface as an error so
                # the service raises rather than playing nothing.
                print(json.dumps({"error": "No audio"}), flush=True)
                return

            print(json.dumps({"done": True}), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({"error": f"{str(e)}\n{traceback.format_exc()}"}), flush=True)


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = Worker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            if req["cmd"] == "init":
                # init returns a single response dict (printed here).
                resp = worker.initialize(req["model"], req["voice"])
                print(json.dumps(resp), flush=True)
            elif req["cmd"] == "generate":
                # generate streams its own JSON lines (segments + done/error).
                worker.generate(req["text"])
            else:
                print(json.dumps({"error": "Unknown command"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()