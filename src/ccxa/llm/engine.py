"""LLM inference via mlx_lm.server (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


class LLMEngine:
    """Ternary Bonsai 8B inference via mlx_lm.server + OpenAI client."""

    def __init__(
        self,
        model: str = "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
        max_tokens_short: int = 150,
        max_tokens_long: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        port: int = 8080,
        base_url: str | None = None,
        **_kwargs: object,
    ) -> None:
        self._model_id = model
        self._max_tokens_short = max_tokens_short
        self._max_tokens_long = max_tokens_long
        self._temperature = temperature
        self._top_p = top_p
        self._port = port
        self._base_url = base_url or f"http://localhost:{port}/v1"
        self._server_process: subprocess.Popen | None = None
        self._client = None
        self._manages_server = base_url is None

    def load(self) -> None:
        """Start mlx_lm.server (if needed) and create OpenAI client."""
        from openai import OpenAI

        if self._manages_server:
            logger.info(
                "Starting mlx_lm.server with model: %s on port %d",
                self._model_id,
                self._port,
            )
            self._server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm.server",
                    "--model",
                    self._model_id,
                    "--port",
                    str(self._port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self._client = OpenAI(base_url=self._base_url, api_key="local")
        self._wait_for_server()
        logger.info("LLM server ready at %s", self._base_url)

    def _wait_for_server(self, timeout: float = 120.0) -> None:
        """Poll the server until it responds or timeout."""
        import urllib.error
        import urllib.request

        url = self._base_url.rstrip("/") + "/models"
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._server_process and self._server_process.poll() is not None:
                stderr = (
                    self._server_process.stderr.read().decode()
                    if self._server_process.stderr
                    else ""
                )
                raise RuntimeError(
                    f"mlx_lm.server exited unexpectedly: {stderr}"
                )
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1.0)
        raise TimeoutError(
            f"mlx_lm.server did not become ready within {timeout}s"
        )

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def generate(
        self, messages: list[dict[str, str]], detailed: bool = False
    ) -> str:
        """Generate a response via the OpenAI-compatible API."""
        if self._client is None:
            return "すみません、まだ準備中です。"

        max_tokens = self._max_tokens_long if detailed else self._max_tokens_short

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        return response.choices[0].message.content.strip()

    def shutdown(self) -> None:
        """Stop the managed mlx_lm.server process."""
        if self._server_process is not None:
            logger.info(
                "Stopping mlx_lm.server (pid=%d)", self._server_process.pid
            )
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
