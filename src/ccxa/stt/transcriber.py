"""Speech-to-text via mlx-audio (lilfugu / Qwen3-ASR on Apple Silicon)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """STT engine using mlx-audio on Apple Silicon."""

    def __init__(
        self,
        model: str = "holotherapper/lilfugu-8bit",
        language: str = "Japanese",
    ) -> None:
        self._model_id = model
        self._language = language
        self._model = None

    def load(self) -> None:
        """Load the model. Call from background thread at startup."""
        from mlx_audio.stt import load

        logger.info("Loading STT model: %s", self._model_id)
        self._model = load(self._model_id)
        logger.info("STT model loaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 PCM audio to text. Blocking call."""
        if self._model is None:
            return ""

        audio_float = audio.astype(np.float32) / 32768.0
        result = self._model.generate(
            audio_float,
            language=self._language,
        )
        text = result.text.strip() if hasattr(result, "text") else ""
        logger.debug("STT result: '%s'", text)
        return text
