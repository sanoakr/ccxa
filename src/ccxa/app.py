"""Main application orchestrator with state machine."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ccxa.audio.capture import AudioCapture
from ccxa.audio.noise_gate import NoiseGate
from ccxa.audio.playback import ChimePlayer
from ccxa.config import AppConfig
from ccxa.llm.engine import LLMEngine
from ccxa.llm.prompt import build_messages
from ccxa.llm.tools import (
    detect_currency_query,
    detect_detailed_request,
    detect_goodbye,
    detect_search_request,
    detect_time_query,
    detect_weather_query,
    get_currency_context,
    get_current_time_context,
    get_weather_context,
)
from ccxa.search.web import WebSearch
from ccxa.stt.transcriber import Transcriber
from ccxa.state import AppState
from ccxa.tts.speaker import Speaker
from ccxa.vad.silero import SileroVAD, VADEvent
from ccxa.wakeword.detector import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceChatApp:
    """Main application: manages the state machine and coordinates all components."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = AppState.IDLE
        self._running = False

        # Audio pipeline
        self.audio = AudioCapture(config.audio)
        self.noise_gate = NoiseGate(
            sample_rate=config.audio.sample_rate,
            low_cut=config.audio.low_cut_hz,
            high_cut=config.audio.high_cut_hz,
            gate_threshold_db=config.audio.noise_gate_db,
        )
        self.chime = ChimePlayer(chime_path=config.audio.chime_path)

        # VAD
        self.vad = SileroVAD(
            threshold=config.vad.threshold,
            min_speech_ms=config.vad.min_speech_ms,
            min_silence_ms=config.vad.min_silence_ms,
            sample_rate=config.audio.sample_rate,
        )

        # Wake word
        self.wakeword = WakeWordDetector(
            phrases=config.wakeword.phrases,
            max_duration_ms=config.wakeword.max_duration_ms,
            fuzzy_threshold=config.wakeword.fuzzy_threshold,
        )

        # STT
        self.stt = Transcriber(
            model=config.stt.model,
            language=config.stt.language,
        )

        # LLM
        self.llm = LLMEngine(
            model=config.llm.model,
            max_tokens_short=config.llm.max_tokens_short,
            max_tokens_long=config.llm.max_tokens_long,
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
        )

        # TTS
        self.speaker = Speaker(voice=config.tts.voice, rate=config.tts.rate)

        # Web search
        self.search = WebSearch(max_results=config.search.max_results)

        # Barge-in settings
        self._bargein_rms = config.audio.bargein_rms_threshold
        self._bargein_chunks = config.audio.bargein_consecutive_chunks
        self._bargein_counter: int = 0

        # Conversation state
        self.conversation_history: list[dict[str, str]] = []
        self._last_activity: float = 0
        self._rms_log_counter: int = 0

    def _flush_audio(self) -> None:
        """Discard buffered audio and reset VAD state.

        Call this whenever the bot finishes speaking so that stale audio
        (including the bot's own voice picked up by the mic) is not
        processed as user input.
        """
        self.audio.flush()
        self.vad.reset()
        logger.debug("Audio buffer and VAD state flushed")

    def _load_models(self) -> None:
        """Load STT and LLM models sequentially (called from background thread)."""
        self.stt.load()
        self.llm.load()

    async def run(self) -> None:
        """Main entry point. Starts all components and runs the event loop."""
        self._running = True
        logger.info("Starting voice chat assistant...")

        # Start audio capture immediately (VAD is already loaded in __init__)
        self.audio.start()
        logger.info("Audio capture started. Waiting for models to load...")

        # Load models sequentially in a background thread to avoid
        # import deadlock (mlx_audio and mlx_lm share internal modules).
        await asyncio.to_thread(self._load_models)
        logger.info(
            "All models loaded. Say 'チチクサ' to start a conversation."
        )

        try:
            await asyncio.gather(
                self._audio_processing_loop(),
                self._timeout_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self.audio.stop()
            logger.info("Application stopped")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self.speaker.stop()

    async def _audio_processing_loop(self) -> None:
        """Continuously process audio chunks based on current state."""
        while self._running:
            chunk = self.audio.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # Log RMS level periodically (every ~2 seconds = 62 chunks at 32ms)
            self._rms_log_counter += 1
            if self._rms_log_counter >= 62:
                rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
                logger.debug(
                    "[%s] mic RMS=%.1f peak=%d",
                    self.state.value,
                    rms,
                    np.max(np.abs(chunk)),
                )
                self._rms_log_counter = 0

            if self.state == AppState.IDLE:
                # Feed raw audio to VAD (no noise gate — VAD handles noise well
                # and the gate can accidentally suppress short wake words)
                await self._handle_idle(chunk)
            elif self.state == AppState.LISTENING:
                filtered = self.noise_gate.process(chunk)
                await self._handle_listening(filtered)
            elif self.state == AppState.SPEAKING:
                await self._handle_speaking(chunk)

    async def _handle_speaking(self, chunk: np.ndarray) -> None:
        """SPEAKING state: monitor for barge-in (user speaking over the bot).

        If the mic RMS exceeds the threshold for several consecutive chunks,
        the user is likely speaking — interrupt the TTS immediately.
        """
        rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
        if rms >= self._bargein_rms:
            self._bargein_counter += 1
            if self._bargein_counter >= self._bargein_chunks:
                logger.info(
                    "Barge-in detected (RMS=%.0f, %d consecutive chunks)",
                    rms,
                    self._bargein_counter,
                )
                await self.speaker.stop()
                self._bargein_counter = 0
        else:
            self._bargein_counter = 0

    async def _handle_idle(self, chunk: np.ndarray) -> None:
        """IDLE state: listen for wake word via VAD + STT."""
        event = self.vad.process_chunk(chunk)
        if event is None:
            return

        logger.info(
            "VAD detected speech segment: %dms", event.duration_ms
        )

        # Only process short utterances for wake word
        if event.duration_ms > self.wakeword.max_duration_ms:
            logger.debug(
                "Segment too long for wake word (%dms > %dms), ignoring",
                event.duration_ms,
                self.wakeword.max_duration_ms,
            )
            return

        # Transcribe the short utterance
        if not self.stt.is_loaded:
            logger.debug("STT not loaded yet, ignoring segment")
            return

        text = await asyncio.to_thread(self.stt.transcribe, event.audio)
        logger.info("Wake word STT result: '%s'", text)

        if not text:
            return

        # Check for wake word
        if self.wakeword.check(text):
            await self._activate()

    async def _activate(self) -> None:
        """Transition from IDLE to LISTENING."""
        logger.info("Wake word detected! Activating conversation.")
        await self.chime.play_chime()
        self._flush_audio()
        self.state = AppState.LISTENING
        self._last_activity = time.monotonic()
        self.conversation_history.clear()

    async def _handle_listening(self, chunk: np.ndarray) -> None:
        """LISTENING state: detect speech segments and process them."""
        event = self.vad.process_chunk(chunk)
        if event is None:
            if self.vad.is_speaking:
                self._last_activity = time.monotonic()
            return

        self._last_activity = time.monotonic()
        await self._process_utterance(event)

    async def _process_utterance(self, event: VADEvent) -> None:
        """Process a completed speech segment: STT -> intent -> LLM -> TTS."""
        self.state = AppState.PROCESSING

        # STT
        text = await asyncio.to_thread(self.stt.transcribe, event.audio)
        if not text.strip():
            logger.debug("Empty transcription, returning to LISTENING")
            self.state = AppState.LISTENING
            return

        logger.info("User: %s", text)

        # Check for goodbye
        if detect_goodbye(text, self.config.goodbye_phrases):
            self.state = AppState.SPEAKING
            self._bargein_counter = 0
            await self.speaker.speak("さようなら、またお話しましょう。")
            self._flush_audio()
            self.state = AppState.IDLE
            self.conversation_history.clear()
            return

        # Detect intents
        detailed = detect_detailed_request(text)
        search_query = detect_search_request(text)
        is_time_query = detect_time_query(text)
        currency = detect_currency_query(text)
        weather = detect_weather_query(text)

        extra_context = None

        if search_query:
            # Web search flow
            self.state = AppState.SPEAKING
            self._bargein_counter = 0
            await self.speaker.speak("Web検索するのでちょっと待ってください。")
            self._flush_audio()
            self.state = AppState.PROCESSING

            search_results = await asyncio.to_thread(
                self.search.search, search_query
            )
            extra_context = (
                f"以下はウェブ検索結果です:\n{search_results}\n\n"
                "この情報を元に回答してください。"
            )

        elif weather:
            location_slug, display_name = weather
            extra_context = await asyncio.to_thread(
                get_weather_context, location_slug, display_name
            )

        elif currency:
            currency_code, display_name = currency
            extra_context = await asyncio.to_thread(
                get_currency_context, currency_code, display_name
            )

        elif is_time_query:
            extra_context = get_current_time_context()

        # Generate LLM response
        messages = build_messages(
            self.conversation_history,
            text,
            detailed=detailed,
            extra_context=extra_context,
        )
        response = await asyncio.to_thread(self.llm.generate, messages, detailed)

        logger.info("Assistant: %s", response)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        # Speak response (can be interrupted by barge-in)
        self.state = AppState.SPEAKING
        self._bargein_counter = 0
        completed = await self.speaker.speak(response)

        # Flush stale audio (bot's own voice + anything buffered during processing)
        self._flush_audio()
        self.state = AppState.LISTENING
        self._last_activity = time.monotonic()

        if not completed:
            logger.info("Response was interrupted by user")

    async def _timeout_loop(self) -> None:
        """Monitor for conversation timeout and return to IDLE."""
        while self._running:
            await asyncio.sleep(1.0)
            if self.state == AppState.LISTENING:
                elapsed = time.monotonic() - self._last_activity
                if elapsed > self.config.conversation_timeout:
                    logger.info("Conversation timeout, returning to IDLE")
                    self.state = AppState.IDLE
                    self.conversation_history.clear()
