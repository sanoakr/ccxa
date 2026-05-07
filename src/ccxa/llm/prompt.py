"""System prompt construction for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
音声アシスタントです。日本語の話し言葉で1〜2文で答えてください。記号・マークダウン・箇条書き・SUMMARY・TAGSは使用不可。「他に何かありますか」などの確認・締め括りは言わないでください。質問は回答に必要な情報が足りない時だけにしてください。
"""

SYSTEM_PROMPT_DETAILED = """\
音声アシスタントです。日本語の話し言葉で詳しく答えてください。記号・マークダウン・箇条書き・SUMMARY・TAGSは使用不可。「他に何かありますか」などの確認・締め括りは言わないでください。質問は回答に必要な情報が足りない時だけにしてください。
"""


def build_messages(
    conversation_history: list[dict[str, str]],
    user_text: str,
    detailed: bool = False,
    extra_context: str | None = None,
) -> list[dict[str, str]]:
    """Build the message list for LLM chat completion."""
    prompt = SYSTEM_PROMPT_DETAILED if detailed else SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]

    # Keep last 10 turns of conversation history
    messages.extend(conversation_history[-10:])

    content = user_text
    if extra_context:
        content += f"\n\n{extra_context}"

    messages.append({"role": "user", "content": content})
    return messages
