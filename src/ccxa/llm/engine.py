"""LLM inference engine using MLX for Ternary Bonsai 8B."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LLMEngine:
    """Bonsai 8B inference via mlx-lm."""

    def __init__(
        self,
        model: str = "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
        max_tokens_short: int = 150,
        max_tokens_long: int = 500,
        **_kwargs: object,
    ) -> None:
        self._model_id = model
        self._max_tokens_short = max_tokens_short
        self._max_tokens_long = max_tokens_long
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load the model and tokenizer. Call from background thread."""
        import mlx_lm

        logger.info("Loading LLM model: %s", self._model_id)
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        logger.info("LLM model loaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, messages: list[dict[str, str]], detailed: bool = False) -> str:
        """Generate a response from chat messages. Blocking call."""
        import mlx_lm

        if self._model is None or self._tokenizer is None:
            return "すみません、まだ準備中です。"

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_tokens = self._max_tokens_long if detailed else self._max_tokens_short

        response = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response.strip()
