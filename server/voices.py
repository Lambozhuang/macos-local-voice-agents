# Single source of truth for the Kokoro TTS voices the study uses.
# Imported by both bot.py (to validate the client's requested voice) and
# prewarm.py (to cache every voice file before the study runs).
# Order matches the KokoroVoice enum dropdown in the Unity client.

KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"

KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
    "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
]

DEFAULT_VOICE = "af_heart"
ALLOWED_VOICES = set(KOKORO_VOICES)
