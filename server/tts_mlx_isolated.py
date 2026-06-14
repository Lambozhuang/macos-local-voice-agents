#
# Process-isolated Kokoro TTS service
# Uses a separate process to avoid Metal threading conflicts on Apple Silicon
#

import asyncio
import subprocess
import json
import base64
import sys
from typing import AsyncGenerator, Optional
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


class ClauseTextAggregator(SimpleTextAggregator):
    """Flushes on clause boundaries (`, ` and `: `) in addition to sentence endings.

    Reduces first-audio latency for batch TTS: synthesis starts after the first
    clause rather than waiting for a full sentence.
    Commas/colons followed by a non-space character (e.g. "1,000") are NOT split.
    """

    _CLAUSE_PUNCT = frozenset({",", ":"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clause_pending = False

    async def aggregate(self, text: str) -> AsyncGenerator[Aggregation, None]:
        if self._aggregation_type == AggregationType.TOKEN:
            if text:
                yield Aggregation(text=text, type=AggregationType.TOKEN)
            return

        for char in text:
            self._text += char

            result = await self._check_sentence_with_lookahead(char)
            if result:
                self._clause_pending = False
                yield result
                continue

            if self._needs_lookahead:
                continue

            if self._clause_pending and char == " ":
                flush_text = self._text[:-1].strip()
                if flush_text:
                    self._text = ""
                    self._clause_pending = False
                    yield Aggregation(text=flush_text, type=AggregationType.SENTENCE)
            elif self._text and self._text[-1] in self._CLAUSE_PUNCT:
                self._clause_pending = True
            elif char not in {" ", "\t", "\n"}:
                # Non-space after a pending comma means something like "1,000" — don't split
                self._clause_pending = False

    async def handle_interruption(self):
        await super().handle_interruption()
        self._clause_pending = False

    async def reset(self):
        await super().reset()
        self._clause_pending = False


class TTSMLXIsolated(TTSService):
    """Completely isolated Kokoro TTS using subprocess to avoid Metal issues."""

    def __init__(
        self,
        *,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        """Initialize the isolated Kokoro TTS service."""
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(model=model, voice=voice, language=None),
            **kwargs,
        )

        self._model_name = model
        self._voice = voice
        self._device = device

        self._process = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Use clause-aware aggregator so synthesis starts after the first clause
        # (e.g. "Hello," → TTS) rather than waiting for a full sentence.
        self._text_aggregator = ClauseTextAggregator(aggregation_type=AggregationType.SENTENCE)

        # Get path to worker script
        self._worker_script = self._get_worker_script_path()

        self._mlx_settings = {
            "model": model,
            "voice": voice,
            "sample_rate": sample_rate,
        }

    def _get_worker_script_path(self) -> str:
        """Get the path to the standalone worker script."""
        # Look for kokoro_worker.py in the same directory as this file
        current_dir = Path(__file__).parent
        if self._model_name.startswith("Marvis-AI"):
            worker_path = current_dir / "marvis_worker.py"
        else:
            worker_path = current_dir / "kokoro_worker.py"

        logger.info(f"Using worker script: {worker_path}")

        if not worker_path.exists():
            raise FileNotFoundError(
                f"Worker script not found at {worker_path}. "
                "Make sure worker script is in the same directory as tts_mlx_isolated.py"
            )

        return str(worker_path)

    def _start_worker(self):
        """Start the worker process."""
        try:
            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )
            logger.info(f"Started {self._model_name} worker process: {self._process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False

    def _send_command(self, command: dict) -> dict:
        """Send command to worker and get response."""
        import select as _select
        import time

        try:
            if not self._process or self._process.poll() is not None:
                logger.debug("Starting worker process...")
                if not self._start_worker():
                    return {"error": "Failed to start worker"}

            # Send command
            cmd_json = json.dumps(command) + "\n"
            logger.debug(f"Sending command: {command}")
            self._process.stdin.write(cmd_json)
            self._process.stdin.flush()

            # Read response, skipping blank/non-JSON lines the subprocess may emit
            # (e.g. MLX Metal kernel compilation output or tqdm artefacts).
            # Use a per-line timeout that resets on each readable line so the total
            # wall-clock budget is generous enough for first-time model loading.
            timeout = 60.0
            deadline = time.monotonic() + timeout

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"error": "Worker response timeout"}

                ready, _, _ = _select.select([self._process.stdout], [], [], remaining)
                if not ready:
                    return {"error": "Worker response timeout"}

                response_line = self._process.stdout.readline()
                if not response_line:
                    if self._process.poll() is not None:
                        return {"error": "Worker process died"}
                    return {"error": "No response from worker"}

                response_line = response_line.strip()
                if not response_line:
                    continue  # skip blank lines emitted during model loading

                try:
                    response_data = json.loads(response_line)
                except json.JSONDecodeError:
                    logger.debug(f"Skipping non-JSON worker output: {response_line!r}")
                    continue

                if "audio" in response_data:
                    logger.debug(
                        f"Worker response: success with {len(response_data.get('audio', ''))} chars of audio data"
                    )
                else:
                    logger.debug(f"Worker response: {response_line}")
                return response_data

        except Exception as e:
            logger.error(f"Worker communication error: {e}")
            return {"error": str(e)}

    async def _initialize_if_needed(self) -> bool:
        """Initialize the worker if not already done. Safe to call concurrently."""
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._send_command,
                {"cmd": "init", "model": self._model_name, "voice": self._voice},
            )

            if result.get("success"):
                self._initialized = True
                logger.info("Kokoro worker initialized")
                return True

            error_msg = result.get("error", "Unknown error")
            logger.error(f"Worker initialization failed: {error_msg}")
            if self._process and self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error(f"Worker process stderr: {stderr_output}")
            return False

    async def prewarm(self):
        """Pre-warm the worker so the first synthesis has no initialization delay."""
        logger.debug(f"{self}: Pre-warming TTS worker")
        await self._initialize_if_needed()

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using isolated worker process."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Initialize worker if needed
            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")

            # Generate audio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._send_command, {"cmd": "generate", "text": text}
            )

            if not result.get("success"):
                raise RuntimeError(f"Audio generation failed: {result.get('error')}")

            # Decode audio
            audio_b64 = result["audio"]
            audio_bytes = base64.b64decode(audio_b64)

            await self.stop_ttfb_metrics()

            # Stream audio
            CHUNK_SIZE = self.chunk_size
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        """Clean up worker process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except:
                try:
                    self._process.kill()
                except:
                    pass
            self._process = None

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown."""
        self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)
