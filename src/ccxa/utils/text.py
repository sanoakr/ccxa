"""Text utility functions."""

from __future__ import annotations

import re

# Detects whether an LLM response is a question requiring a follow-up answer.
# Matches: explicit ？/?, or Japanese interrogative endings (ですか, ますか, etc.)
_QUESTION_RE = re.compile(
    r"[？?]\s*$"
    r"|(?:です|ます|でしょう)か[。！\s]*$"
    r"|かな[。\s]*$"
    r"|かしら[。\s]*$",
)

# Filler / courtesy questions that do NOT require a follow-up answer from the user.
# These are social formulas the LLM adds at the end of a response; they should
# be treated as statements so the session ends after one turn.
_FILLER_RE = re.compile(
    r"他に.{0,25}(?:ありますか|でしょうか|ございますか)"  # 他に何かありますか etc.
    r"|何か.{0,15}(?:ご質問|ご要望|ご希望|お手伝い)"      # 何かご質問はありますか etc.
    r"|お手伝いでき"                                       # お手伝いできることはありますか
    r"|ご不明な点"                                         # ご不明な点はありますか
    r"|ご要望.{0,10}(?:ありますか|でしょうか)"
    r"|まだ.{0,15}(?:ありますか|でしょうか|ご希望)",
)


def is_question(text: str) -> bool:
    """Return True if the response is a clarifying question needing a follow-up.

    Filler/courtesy questions (e.g. 「他に何かありますか？」) return False even
    though they end with ？, because they do not require information from the user.
    """
    stripped = text.strip()
    if _FILLER_RE.search(stripped):
        return False
    return bool(_QUESTION_RE.search(stripped))
