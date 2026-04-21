"""System prompt construction for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
あなたは日本語の音声アシスタント「チチクサ」です。ユーザーと自然な日本語で会話してください。

## 基本ルール
- 回答は簡潔に、1〜2文で答えてください。
- 話し言葉で自然に応答してください。敬語を使いますが堅すぎないようにしてください。
- 音声で読み上げられるので、マークダウンや記号（*、#、-、```など）は絶対に使わないでください。
- 数字は読みやすい形で書いてください。
- 箇条書きではなく、文章で答えてください。

## 「詳しく」モード
ユーザーが「詳しく」「もっと教えて」「詳細に」と言った場合のみ、5〜10文程度で詳しく説明してください。
それ以外は必ず短く答えてください。
"""


def build_messages(
    conversation_history: list[dict[str, str]],
    user_text: str,
    detailed: bool = False,
    extra_context: str | None = None,
) -> list[dict[str, str]]:
    """Build the message list for LLM chat completion."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep last 10 turns of conversation history
    messages.extend(conversation_history[-10:])

    content = user_text
    if extra_context:
        content += f"\n\n{extra_context}"
    if detailed:
        content += "\n\n（ユーザーは詳しい説明を求めています。詳細に回答してください。）"

    messages.append({"role": "user", "content": content})
    return messages
