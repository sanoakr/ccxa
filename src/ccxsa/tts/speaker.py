"""Text-to-speech via macOS say command."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class Speaker:
    """macOS TTS wrapper using the say command.

    By default, uses the system default voice (no -v flag) which typically
    produces more natural speech than explicitly named voices.
    """

    def __init__(self, voice: str | None = None, rate: int | None = None) -> None:
        self._voice = voice
        self._rate = rate
        self._process: asyncio.subprocess.Process | None = None
        self.interrupted = False

    @property
    def is_speaking(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def speak(self, text: str) -> bool:
        """Speak text asynchronously. Blocks until speech completes or interrupted.

        Returns True if completed normally, False if interrupted.
        """
        if not text.strip():
            return True
        self.interrupted = False
        logger.debug("TTS: '%s'", text)
        cmd: list[str] = ["say"]
        if self._voice:
            cmd.extend(["-v", self._voice])
        if self._rate:
            cmd.extend(["-r", str(self._rate)])
        cmd.append(text)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._process.wait()
        self._process = None
        return not self.interrupted

    async def stop(self) -> None:
        """Interrupt current speech."""
        if self._process and self._process.returncode is None:
            self.interrupted = True
            self._process.terminate()
            await self._process.wait()
            self._process = None
            logger.info("TTS interrupted by barge-in")
