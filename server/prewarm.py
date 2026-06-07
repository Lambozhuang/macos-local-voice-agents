#!/usr/bin/env python3
"""
Pre-download / pre-cache every Kokoro voice the study uses, so nothing is
fetched from the network mid-session.

Kokoro-82M is a SINGLE model; the 20 "voices" are small per-voice embedding
files that mlx-audio fetches lazily the first time each voice is generated.
This script forces every one of them into the local HuggingFace cache by
loading the model once and running a one-word generation per voice.

Run once on the Mac (online), then run the server with HF_HUB_OFFLINE=1:

    uv run prewarm.py
    HF_HUB_OFFLINE=1 uv run bot.py --host 0.0.0.0

Re-run only if you add new voices to voices.py.
"""

import sys

from voices import KOKORO_MODEL, KOKORO_VOICES

try:
    from mlx_audio.tts.utils import load_model
except ImportError:
    print("ERROR: mlx_audio not available. Run inside the server env (uv run prewarm.py).")
    sys.exit(1)


def main():
    print(f"Loading model {KOKORO_MODEL} (downloads on first run)...")
    model = load_model(KOKORO_MODEL)
    print("Model loaded. Warming each voice...\n")

    ok, failed = [], []
    for i, voice in enumerate(KOKORO_VOICES, 1):
        try:
            # Exhaust the generator so the per-voice file is actually fetched + used.
            list(model.generate(text="test", voice=voice, speed=1.0))
            print(f"  [{i:2}/{len(KOKORO_VOICES)}] {voice:14} ok")
            ok.append(voice)
        except Exception as e:
            print(f"  [{i:2}/{len(KOKORO_VOICES)}] {voice:14} FAILED: {e}")
            failed.append(voice)

    print(f"\nDone. {len(ok)} cached, {len(failed)} failed.")
    if failed:
        print("Failed voices (remove from voices.py or check the id):", ", ".join(failed))
        sys.exit(1)
    print("All voices cached. You can now run the server offline:")
    print("    HF_HUB_OFFLINE=1 uv run bot.py --host 0.0.0.0")


if __name__ == "__main__":
    main()
