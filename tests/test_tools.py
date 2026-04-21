"""Tests for intent detection tools."""

from ccxa.llm.tools import (
    detect_currency_query,
    detect_detailed_request,
    detect_goodbye,
    detect_search_request,
    detect_time_query,
    detect_weather_query,
    get_current_time_context,
)


def test_detect_time_query():
    assert detect_time_query("今何時？")
    assert detect_time_query("今日は何日ですか")
    assert detect_time_query("何曜日ですか")
    assert not detect_time_query("天気を教えて")


def test_detect_search_request():
    assert detect_search_request("東京の天気を調べて") is not None
    assert detect_search_request("Pythonについて調べて") is not None
    assert detect_search_request("こんにちは") is None


def test_detect_search_request_extracts_query():
    query = detect_search_request("東京の天気を調べて")
    assert query is not None
    assert "東京" in query


def test_detect_detailed_request():
    assert detect_detailed_request("詳しく教えて")
    assert detect_detailed_request("もっと教えてほしい")
    assert not detect_detailed_request("ありがとう")


def test_detect_goodbye():
    phrases = ["さようなら", "終わり", "バイバイ"]
    assert detect_goodbye("さようなら", phrases)
    assert detect_goodbye("もう終わり", phrases)
    assert not detect_goodbye("こんにちは", phrases)


def test_get_current_time_context():
    ctx = get_current_time_context()
    assert "現在の日時情報" in ctx
    assert "年" in ctx
    assert "月" in ctx
    assert "日" in ctx


def test_detect_currency_query_usd():
    result = detect_currency_query("1ドルいくら？")
    assert result is not None
    assert result[0] == "USD"


def test_detect_currency_query_eur():
    result = detect_currency_query("1ユーロ何円？")
    assert result is not None
    assert result[0] == "EUR"


def test_detect_currency_query_gbp():
    result = detect_currency_query("ポンドのレートは？")
    assert result is not None
    assert result[0] == "GBP"


def test_detect_currency_query_none():
    assert detect_currency_query("こんにちは") is None
    assert detect_currency_query("ドルで買い物した") is None  # no rate keyword


def test_detect_weather_query_kusatsu():
    result = detect_weather_query("草津の天気は？")
    assert result is not None
    assert result[0] == "Kusatsu"


def test_detect_weather_query_tokyo():
    result = detect_weather_query("東京の天気教えて")
    assert result is not None
    assert result[0] == "Tokyo"


def test_detect_weather_query_kyoto():
    result = detect_weather_query("京都は雨？")
    assert result is not None
    assert result[0] == "Kyoto"


def test_detect_weather_query_default():
    # No city mentioned -> defaults to Kusatsu
    result = detect_weather_query("今日の天気は？")
    assert result is not None
    assert result[0] == "Kusatsu"


def test_detect_weather_query_none():
    assert detect_weather_query("こんにちは") is None
