"""Lock-free single-producer single-consumer ring buffer for audio frames."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Thread-safe SPSC ring buffer backed by a numpy array.

    The producer (audio callback thread) calls write().
    The consumer (asyncio task) calls read().
    No locks are needed because each index is only mutated by one side.
    """

    def __init__(
        self, capacity_frames: int, channels: int = 1, dtype: type = np.int16
    ) -> None:
        self._buf = np.zeros((capacity_frames, channels), dtype=dtype)
        self._capacity = capacity_frames
        self._channels = channels
        self._write_idx: int = 0
        self._read_idx: int = 0

    def write(self, frames: np.ndarray) -> int:
        """Write frames into the buffer. Returns number of frames actually written."""
        if frames.ndim == 1:
            frames = frames.reshape(-1, self._channels)
        n = frames.shape[0]
        available = self._capacity - self.available()
        if n > available:
            n = available
        if n == 0:
            return 0

        wi = self._write_idx % self._capacity
        end = wi + n
        if end <= self._capacity:
            self._buf[wi:end] = frames[:n]
        else:
            first = self._capacity - wi
            self._buf[wi:] = frames[:first]
            self._buf[: n - first] = frames[first:n]
        self._write_idx += n
        return n

    def read(self, num_frames: int) -> np.ndarray | None:
        """Read up to num_frames from the buffer. Returns None if empty."""
        avail = self.available()
        if avail == 0:
            return None
        n = min(num_frames, avail)

        ri = self._read_idx % self._capacity
        end = ri + n
        if end <= self._capacity:
            data = self._buf[ri:end].copy()
        else:
            first = self._capacity - ri
            data = np.concatenate(
                [self._buf[ri:].copy(), self._buf[: n - first].copy()]
            )
        self._read_idx += n
        if self._channels == 1:
            return data.ravel()
        return data

    def available(self) -> int:
        """Number of frames available for reading."""
        return self._write_idx - self._read_idx

    def flush(self) -> None:
        """Discard all buffered data. Called by the consumer side."""
        self._read_idx = self._write_idx
