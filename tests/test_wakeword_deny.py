"""Tests for wake word deny_phrases gate."""

from ccxa.wakeword.detector import WakeWordDetector

_DENY = ["アレクサ", "ヘイシリ", "オーケーグーグル", "ねえグーグル", "コルタナ"]


def _det(**kwargs) -> WakeWordDetector:
    return WakeWordDetector(phrases=["チチクサ"], deny_phrases=_DENY, fuzzy_threshold=1, **kwargs)


# --- deny gate rejects device wake words ---

def test_alexa_denied():
    det = _det()
    assert not det.check("アレクサ")
    assert not det.check("ねえアレクサ")
    assert not det.check("アレクサ、今日の天気は")


def test_siri_denied():
    det = _det()
    assert not det.check("ヘイシリ")
    assert not det.check("へいしり、音楽かけて")


def test_google_denied():
    det = _det()
    assert not det.check("オーケーグーグル")
    assert not det.check("ねえグーグル")
    assert not det.check("ねえグーグル、道を教えて")


def test_cortana_denied():
    det = _det()
    assert not det.check("コルタナ")


# --- deny gate does not block the actual wake word ---

def test_wake_word_still_works():
    det = _det()
    assert det.check("チチクサ")
    assert det.check("ちちくさ")
    assert det.check("えーとチチクサお願い")


def test_fuzzy_wake_word_still_works():
    det = _det()
    assert det.check("ちちぐさ")   # 1 substitution
    assert det.check("ちくさ")     # 1 deletion
    assert det.check("チチクサー")  # 1 insertion


# --- empty deny list preserves original fuzzy behaviour (Alexa bug) ---

def test_empty_deny_allows_alexa_fuzzy():
    """Without deny_phrases, アレクサ (dist=2) still triggers — documents original bug."""
    det = WakeWordDetector(phrases=["チチクサ"], deny_phrases=[], fuzzy_threshold=2)
    assert det.check("アレクサ")


# --- deny phrases normalize correctly (katakana / hiragana both work) ---

def test_deny_normalizes_katakana_and_hiragana():
    det_kata = WakeWordDetector(phrases=["チチクサ"], deny_phrases=["アレクサ"], fuzzy_threshold=2)
    det_hira = WakeWordDetector(phrases=["チチクサ"], deny_phrases=["あれくさ"], fuzzy_threshold=2)
    assert not det_kata.check("アレクサ")
    assert not det_hira.check("アレクサ")


# --- unrelated input is not affected by deny gate ---

def test_unrelated_input_not_denied():
    det = _det()
    assert not det.check("こんにちは")
    assert not det.check("おはようございます")
