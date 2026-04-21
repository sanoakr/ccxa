"""Microphone input via sounddevice with ring buffer."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from ccxsa.config import AudioConfig
from ccxsa.utils.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioCapture:
    """Manages microphone input stream.

    Opens the device at its native channel count and downmixes to mono
    for the ring buffer, avoiding PortAudio channel mismatch errors.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.sample_rate = config.sample_rate
        self.chunk_samples = self.sample_rate * config.chunk_ms // 1000
        self._device = config.input_device

        # Detect native channel count of the input device
        dev_info = sd.query_devices(self._device or sd.default.device[0], "input")
        self._device_channels: int = dev_info["max_input_channels"]
        logger.info(
            "Input device '%s': %d native channels",
            dev_info["name"],
            self._device_channels,
        )

        # Ring buffer is always mono (1 channel)
        self.ring_buffer = RingBuffer(
            capacity_frames=self.sample_rate * 30, channels=1
        )
        self._stream: sd.InputStream | None = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        # Downmix to mono if stereo/multi-channel
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1).astype(np.int16)
        else:
            mono = indata[:, 0].astype(np.int16)
        self.ring_buffer.write(mono)

    def start(self) -> None:
        """Start the audio input stream."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self._device_channels,
            dtype="int16",
            blocksize=self.chunk_samples,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Audio capture started: %dHz, %dch->mono, device=%s",
            self.sample_rate,
            self._device_channels,
            self._device or "default",
        )

    def flush(self) -> None:
        """Discard all buffered audio data."""
        self.ring_buffer.flush()

    def stop(self) -> None:
        """Stop the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    def read_chunk(self) -> np.ndarray | None:
        """Read one chunk (chunk_ms worth of mono samples) from the ring buffer."""
        return self.ring_buffer.read(self.chunk_samples)
