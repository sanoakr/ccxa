"""Silero VAD wrapper with speech segment detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VADEvent:
    """A completed speech segment detected by VAD."""

    audio: np.ndarray
    duration_ms: int


class SileroVAD:
    """Wraps Silero VAD to detect speech segments from audio chunks."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 800,
        sample_rate: int = 16000,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._min_speech_samples = sample_rate * min_speech_ms // 1000
        self._min_silence_samples = sample_rate * min_silence_ms // 1000

        self._speech_buffer: list[np.ndarray] = []
        self.is_speaking = False
        self._silence_counter = 0

        # Load Silero VAD model via silero-vad package
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._model.eval()
        logger.info("Silero VAD loaded")

    def process_chunk(self, chunk: np.ndarray) -> VADEvent | None:
        """Process a single audio chunk (~30ms of int16 PCM).

        Returns a VADEvent when a complete speech segment is detected
        (speech followed by sufficient silence), or None otherwise.
        """
        audio_float = torch.from_numpy(chunk.astype(np.float32) / 32768.0)
        prob = self._model(audio_float, self._sample_rate).item()

        if prob >= self._threshold:
            self.is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(chunk)
        elif self.is_speaking:
            self._silence_counter += len(chunk)
            self._speech_buffer.append(chunk)
            if self._silence_counter >= self._min_silence_samples:
                segment = np.concatenate(self._speech_buffer)
                self._reset()
                if len(segment) >= self._min_speech_samples:
                    duration_ms = len(segment) * 1000 // self._sample_rate
                    return VADEvent(audio=segment, duration_ms=duration_ms)
        return None

    def reset(self) -> None:
        """Reset all internal state. Call after flushing the audio buffer."""
        self._speech_buffer.clear()
        self.is_speaking = False
        self._silence_counter = 0
        self._model.reset_states()

    # keep private alias for internal use
    _reset = reset
