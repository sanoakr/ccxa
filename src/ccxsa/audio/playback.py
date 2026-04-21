"""Sound playback utilities."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class ChimePlayer:
    """Plays notification sounds via macOS afplay."""

    def __init__(self, chime_path: str = "assets/chime.aiff") -> None:
        self._chime_path = chime_path

    async def play_chime(self) -> None:
        """Play the wake word detection chime."""
        proc = await asyncio.create_subprocess_exec(
            "afplay",
            self._chime_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        logger.debug("Chime played")
