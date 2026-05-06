"""Tests for is_question helper."""

import pytest

from ccxa.utils.text import is_question


@pytest.mark.parametrize("text", [
    "どの駅ですか？",
    "何時に来ますか？",
    "それはどういう意味でしょうか？",
    "本当にそれでいいですか？",
    "もう少し詳しく教えてもらえますか？",
    "行きますか?",         # ASCII question mark
    "どうかな。",
    "そうかしら。",
    # 文中に文があり末尾が質問
    "わかりました。どの駅ですか？",
    # 末尾が質問（後続スペースあり）
    "どの駅ですか？  ",
])
def testis_question_true(text: str) -> None:
    assert is_question(text), f"Expected question: {text!r}"


@pytest.mark.parametrize("text", [
    "次の電車は9時35分です。",
    "今日の天気は晴れです。",
    "さようなら、またお話しましょう。",
    "承知しました。",
    "電車の時間を教えてください。",   # request, not a question from the agent
    "それは難しい質問ですね。",       # contains 質問 but is not a question itself
    "えっ？ それは知りませんでした。今日の天気は晴れです。",  # ？が文中にあり末尾は文
    # フィラー質問（社交的な確認）→ False
    "他に何かありますか？",
    "他にお手伝いできることはありますか？",
    "何かご質問はありますか？",
    "まだ他にご希望ですか？",
    "ご不明な点はありますか？",
    "何かご要望はございますか？",
    "承知しました。他に何かご希望はありますか？",
    "",
    "   ",
])
def testis_question_false(text: str) -> None:
    assert not is_question(text), f"Expected non-question: {text!r}"
