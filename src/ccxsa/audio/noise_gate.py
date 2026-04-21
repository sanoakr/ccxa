"""Bandpass filter and amplitude noise gate for microphone input."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class NoiseGate:
    """Applies a Butterworth bandpass filter and amplitude gate to audio chunks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        low_cut: float = 80.0,
        high_cut: float = 7500.0,
        gate_threshold_db: float = -40.0,
    ) -> None:
        nyquist = sample_rate / 2.0
        low = max(low_cut / nyquist, 0.001)
        high = min(high_cut / nyquist, 0.999)
        self._sos = butter(4, [low, high], btype="band", output="sos")
        self._threshold = 10 ** (gate_threshold_db / 20.0)
        # Initialize filter state for continuity across chunks
        zi = sosfilt_zi(self._sos)
        self._zi = np.zeros_like(zi)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Apply bandpass filter and noise gate. Returns filtered int16 chunk."""
        # Convert to float for filtering
        audio = chunk.astype(np.float64) / 32768.0

        # Bandpass filter
        filtered, self._zi = sosfilt(self._sos, audio, zi=self._zi)

        # Amplitude noise gate
        rms = np.sqrt(np.mean(filtered**2))
        if rms < self._threshold:
            return np.zeros(len(chunk), dtype=np.int16)

        # Back to int16
        return np.clip(filtered * 32768.0, -32768, 32767).astype(np.int16)
