"""Application state machine."""

from __future__ import annotations

from enum import Enum


class AppState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
