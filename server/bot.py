import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver

from tts_mlx_isolated import TTSMLXIsolated

import re
from pipecat.frames.frames import Frame, LLMTextFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class EndMarkerFilter(FrameProcessor):
    """Strip the literal <END> farewell marker from LLM text before it reaches TTS,
    so Kokoro never speaks it. Inserted in the pipeline AFTER the RTVI processor +
    LLM, so the client still receives <END> in bot-llm-text (its Done-button signal);
    only the text feeding TTS is cleaned.

    The LLM streams token by token, so the marker may arrive split across
    LLMTextFrames ('<','END','>'); we buffer the minimal tail that could be a partial
    marker and flush the rest, then drain on the end-of-response frame. Tolerant of
    whitespace variants like '< END >'. All non-text frames pass through untouched.
    """

    _MARKER_RE = re.compile(r"<\s*END\s*>", re.IGNORECASE)
    _MAXHOLD = len("< END >")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buf = ""

    async def _emit(self, text: str):
        if text:
            await self.push_frame(LLMTextFrame(text))

    async def _scan_and_flush(self, *, final: bool):
        # Remove any complete markers already in the buffer.
        while True:
            m = self._MARKER_RE.search(self._buf)
            if not m:
                break
            self._buf = self._buf[: m.start()] + self._buf[m.end():]
        if final:
            await self._emit(self._buf)
            self._buf = ""
            return
        # Hold back only a trailing run that could be the start of a split marker
        # (e.g. "<", "< E", "< END"); emit everything before it.
        keep = 0
        tail = self._buf[-self._MAXHOLD:]
        for i in range(len(tail)):
            if re.fullmatch(r"<\s*(E(N(D\s*)?)?)?", tail[i:], re.IGNORECASE):
                keep = len(tail) - i
                break
        if keep:
            safe, self._buf = self._buf[:-keep], self._buf[-keep:]
        else:
            safe, self._buf = self._buf, ""
        await self._emit(safe)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._buf += frame.text
            await self._scan_and_flush(final=False)
            return  # swallow the original; cleaned text is re-emitted by _emit
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._scan_and_flush(final=True)
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

# Kokoro voices the client may request via the /api/offer "voice" field. An
# unknown id falls back to the default so a bad request can't crash the worker
# (the Kokoro worker errors on an unknown voice during its test generation).
# Single source of truth in voices.py (shared with prewarm.py).
from voices import ALLOWED_VOICES, DEFAULT_VOICE

# Per-agent persona + default voice, keyed by agent_id (t0..t9) which Unity sends
# in the /api/offer. Ported from the legacy transition_prompts_*.py role prompts.
from agents_config import AGENTS, DEFAULT_AGENT

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION = """
"You are Pipecat, a friendly, helpful chatbot.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

Start the conversation by saying, "Hello, I'm Pipecat!" Then stop and wait for the user.
"""


async def run_bot(webrtc_connection, voice: str = DEFAULT_VOICE, agent_id: str | None = None):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model=MLXModel.LARGE_V3_TURBO_Q4.value))

    tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice=voice, sample_rate=24000)
    # tts = TTSMLXIsolated(model="Marvis-AI/marvis-tts-250m-v0.1", voice=None)

    llm = OpenAILLMService(
        api_key="dummyKey",
        base_url="http://127.0.0.1:1234/v1",
        settings=OpenAILLMService.Settings(
            model="local-model",  # LM Studio ignores this; uses whatever is loaded
            max_tokens=4096,
        ),
    )

    # When Unity sends an agent_id (t0..t9), use that agent's task persona.
    # When there's no agent_id (e.g. the plain web console), fall back to the
    # generic Pipecat assistant so the two clients can share one server.
    if agent_id is not None:
        system_prompt = AGENTS.get(agent_id, AGENTS[DEFAULT_AGENT])["prompt"]
    else:
        system_prompt = SYSTEM_INSTRUCTION
    context = LLMContext(
        [
            {
                "role": "user",
                "content": system_prompt,
            }
        ],
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor()

    end_filter = EndMarkerFilter()  # strip <END> from text feeding TTS (kept in bot-llm-text for the client)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            end_filter,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency_secs):
        logger.info(f"⏱  First bot speech: {latency_secs:.3f}s after client connect")

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_secs):
        logger.info(f"⏱  User→bot latency: {latency_secs:.3f}s")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        events = breakdown.chronological_events()
        if events:
            lines = "\n    ".join(events)
            logger.info(f"⏱  Breakdown:\n    {lines}")

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi), latency_observer, MetricsLogObserver()],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print(f"Client connected: {client}")
        asyncio.create_task(tts.prewarm())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print(f"Client disconnected: {client}")
        await task.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await runner.run()


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        # agent_id (t0..t9) selects a task persona + its default voice. Unity
        # sends one; the plain web console doesn't, so a missing/empty agent_id
        # means "generic Pipecat assistant" and the two clients coexist.
        agent_id = request.get("agent_id") or None
        if agent_id is not None and agent_id not in AGENTS:
            agent_id = DEFAULT_AGENT
        # Explicit non-empty voice in the offer overrides (testing); else use the
        # agent's default voice from the registry, or the global default when no
        # agent_id was given.
        requested_voice = request.get("voice") or ""
        if requested_voice in ALLOWED_VOICES:
            voice = requested_voice
        elif agent_id is not None:
            voice = AGENTS[agent_id]["voice"]
        else:
            voice = DEFAULT_VOICE
        logger.info(f"New connection: agent_id={agent_id}, voice={voice}")
        background_tasks.add_task(run_bot, pipecat_connection, voice, agent_id)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
