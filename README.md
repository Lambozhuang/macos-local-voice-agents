# Local voice agents on macOS with Pipecat

![screenshot](assets/debug-console-screenshot.png)

Pipecat is an open-source, vendor-neutral framework for building real-time voice (and video) AI applications.

This repository contains an example of a voice agent running with all local models on macOS. On an M-series mac, you can achieve voice-to-voice latency of <800 ms with relatively strong models.

This bot runs on Pipecat 1.3.0. The [server/bot.py](server/bot.py) file uses these models:

  - Silero VAD (`stop_secs=0.2`)
  - smart-turn v3 for semantic end-of-turn detection
  - MLX Whisper (`large-v3-turbo-q4`) for STT
  - A local OpenAI-compatible LLM served by LM Studio (model-agnostic; e.g. Gemma 3n)
  - Kokoro TTS (`Kokoro-82M`), run in an isolated subprocess

Note on turn detection: in Pipecat 1.3.0, VAD and turn-taking moved out of
`TransportParams` and into the user aggregator (`LLMUserAggregatorParams`). The
VAD is configured explicitly; `LocalSmartTurnAnalyzerV3` is supplied
*automatically* by the default `UserTurnStrategies` (its default stop strategy),
so smart-turn is active without an explicit `turn_analyzer=` argument. This was
verified against the Pipecat 1.3.0 source (`pipecat/turns/user_turn_strategies.py`,
tag [`v1.3.0`](https://github.com/pipecat-ai/pipecat/tree/v1.3.0)); see also the
[Pipecat docs](https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview).

But you can swap any of them out for other models, or completely reconfigure the pipeline. It's easy to add tool calling, MCP server integrations, use parallel pipelines to do async inference alongside the voice conversations, add custom processing steps, configure interrupt handling to work differently, etc.

The bot and web client here communicate using a low-latency, local, serverless WebRTC connection. For more information on serverless WebRTC, see the Pipecat [SmallWebRTCTransport docs](https://docs.pipecat.ai/server/services/transport/small-webrtc) and this [article](https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/). You could switch over to a different Pipecat transport (for example, a WebSocket-based transport), but WebRTC is the best choice for realtime audio.

For a deep dive into voice AI, including network transport, optimizing for latency, and notes on designing tool calling and complex workflows, see the [Voice AI & Voice Agents Illustrated Guide](https://voiceaiandvoiceagents.com/).

# Models and dependencies

Silero VAD and MLX Whisper run inside the Pipecat process. When the agent code starts, it will need to download model weights that aren't already cached, so first startup can take some time.

The LLM service in this bot uses the OpenAI-compatible chat completion HTTP API. So you will need to run a local OpenAI-compatible LLM server. 

One easy, high-performance, way to run a local LLM server on macOS is [LM Studio](https://lmstudio.ai/). From inside the LM Studio graphical interface, go to the "Developer" tab on the far left to start an HTTP server.

# Run the voice agent

The core voice agent code lives in a single file: [server/bot.py](server/bot.py). There's one custom service here that's not included in Pipecat core: we implemented a local MLX-Audio frame processor on top of the excellent [mlx-audio library](https://github.com/Blaizzy/mlx-audio).

Note that the first time you start the bot it will take some time to initialize the three models. It can be 30 seconds or more before the bot is fully ready to go. Subsequent startups will be much faster.

It's not a bad idea to run a quick `mlx-audio.generate` process from the command line before you run the bot the first time, so you're not waiting for a relatively bug HuggingFace model download for the voice model.

```shell
mlx-audio.generate --model "Marvis-AI/marvis-tts-250m-v0.1" --text "Hello, I'm Pipecat!" --output "output.wav"
# or
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello, I'm Pipecat!" --output "output.wav"
```

```shell
cd server/
```

If you're using uv

```
uv run bot.py
```

If you're using pip

```
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python bot.py
```

After you run the first time and have all the models cached, you can set the HF_HUB_OFFLINE environment variable to prevent the Hugging Face libraries from going to the network and checking for model updates. This makes the initial bot startup and first conversation turn a lot faster.

```
HF_HUB_OFFLINE=1 uv run bot.py
```

# Start the web client

The web client is a React app. You can connect to your local macOS agent using any client that can negotiate a serverless WebRTC connection. The client in this repo is based on [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) and just uses that library's standard debug console template.

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```