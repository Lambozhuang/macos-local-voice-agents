# macOS local voice-agent server (thesis fork)

The voice-agent server used in the master's thesis *The Impact of Network Degradation on
Quality of Experience in Real-Time Conversation with LLM-powered Virtual Agents* (KTH).
It runs the entire realtime voice pipeline — VAD, turn detection, speech recognition,
LLM, speech synthesis — locally on one Mac and talks to a Unity XR client
([iva-cui](https://github.com/Lambozhuang/iva-cui)) over serverless WebRTC.

Forked from [kwindla/macos-local-voice-agents](https://github.com/kwindla/macos-local-voice-agents)
(a Pipecat example of a fully local macOS voice agent). Main changes from upstream:

- **Multi-agent registry** (`server/agents_config.py`): ten personas (`t0`…`t9`, training +
  city + hotel + museum roles), each a persona prompt + per-agent FACTS block + a shared
  style leash, with a per-agent Kokoro voice. The client selects the agent per connection
  via `agent_id` in the `/api/offer` request; one server process serves them all.
- The client is the Unity XR app instead of the upstream React console (`client/` keeps
  the upstream debug console for testing without a headset).

## Pipeline

One Pipecat (1.3.0, Python 3.12) process, all models local, port **7860**:

| Stage | Model |
|---|---|
| Voice activity detection | Silero VAD (`stop_secs=0.2`) |
| End-of-turn detection | smart-turn v3.2 weights (`smart-turn-v3.2-cpu.onnx`; the class is named V3), CPU, 1.0 s limit |
| Speech recognition | MLX Whisper `large-v3-turbo` (4-bit) |
| Language model | via LM Studio at `127.0.0.1:1234` (study: Meta-Llama-3.1-8B-Instruct Q5_K_M) |
| Speech synthesis | Kokoro-82M (bf16) at 24 kHz mono, isolated subprocess, one voice embedding per agent |

Transport is Pipecat's `SmallWebRTCTransport` (aiortc): Opus audio both ways plus an RTVI
data channel with live transcripts, speaking on/off events and per-stage metrics.
Signaling is a single `POST /api/offer`; the agent greets only after the client sends
`client-ready`. No cloud, no API keys.

Note on Pipecat 1.3.0 turn-taking: VAD and turn-taking are configured on the user
aggregator (`LLMUserAggregatorParams`), and `LocalSmartTurnAnalyzerV3` is supplied
automatically by the default `UserTurnStrategies` — smart-turn is active without an
explicit `turn_analyzer=` argument.

## Running

1. Start an LM Studio server (Developer tab) on `127.0.0.1:1234` with your model loaded.
2. Start the bot (first run downloads model weights and is slow):

```shell
cd server
HF_HUB_OFFLINE=1 uv run bot.py --host 0.0.0.0
```

`--host 0.0.0.0` is required for LAN clients (Unity). Leave `HF_HUB_OFFLINE` off for the
very first run so the weights can download. For a quick test without the headset, run the
debug web client: `cd client && npm i && npm run dev`.
