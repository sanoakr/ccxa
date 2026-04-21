"""Wake word detection via VAD + STT keyword matching.

Since no pre-trained Japanese wake word model exists in openWakeWord,
this module uses a two-stage approach:
1. Silero VAD detects short utterances (250ms-2000ms) during IDLE state
2. The utterance is transcribed by STT
3. The transcription is checked against configured wake word phrases

Matching is intentionally fuzzy because Whisper often misspells unusual
Japanese words.  We check exact substring match first, then fall back to
edit-distance matching on the hiragana reading.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Katakana → Hiragana mapping
_NORMALIZE_MAP = str.maketrans(
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー",
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ"
    "ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖー",
)

# Common kanji that Whisper may output for "ちちくさ"
_KANJI_ALIASES = {
    "千草": "ちくさ",
    "乳草": "ちちくさ",
    "父草": "ちちくさ",
    "知地草": "ちちくさ",
    "チチクサ": "ちちくさ",
}


def _normalize(text: str) -> str:
    """Normalize: lowercase, katakana→hiragana, strip punctuation/whitespace."""
    text = text.lower().translate(_NORMALIZE_MAP)
    # Replace known kanji aliases
    for kanji, reading in _KANJI_ALIASES.items():
        text = text.replace(kanji.lower().translate(_NORMALIZE_MAP), reading)
    text = re.sub(r"[^\w]", "", text)
    return text


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


class WakeWordDetector:
    """Detects wake word by matching STT output against configured phrases."""

    def __init__(
        self,
        phrases: list[str] | None = None,
        max_duration_ms: int = 2000,
        fuzzy_threshold: int = 2,
    ) -> None:
        raw_phrases = phrases or ["チチクサ", "ちちくさ"]
        self._phrases = list({_normalize(p) for p in raw_phrases})
        self.max_duration_ms = max_duration_ms
        self._fuzzy_threshold = fuzzy_threshold
        logger.info("Wake word phrases (normalized): %s", self._phrases)

    def check(self, transcription: str) -> bool:
        """Check if a transcription matches the wake word.

        1. Exact substring match (after normalization)
        2. Fuzzy match: edit distance <= threshold on each sliding window
        """
        normalized = _normalize(transcription)

        for phrase in self._phrases:
            # Exact substring
            if phrase in normalized:
                logger.info(
                    "Wake word EXACT match: '%s' in '%s'",
                    phrase,
                    transcription,
                )
                return True

            # Fuzzy: slide a window of len(phrase) ±1 over normalized text
            for window_len in range(
                max(1, len(phrase) - 1), len(phrase) + 2
            ):
                for start in range(len(normalized) - window_len + 1):
                    window = normalized[start : start + window_len]
                    dist = _edit_distance(phrase, window)
                    if dist <= self._fuzzy_threshold:
                        logger.info(
                            "Wake word FUZZY match (dist=%d): '%s' ~ '%s' in '%s'",
                            dist,
                            phrase,
                            window,
                            transcription,
                        )
                        return True

        return False
