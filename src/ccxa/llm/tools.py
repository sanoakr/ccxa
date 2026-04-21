"""Tool execution for time queries, currency rates, and keyword-based intent detection."""

from __future__ import annotations

import datetime
import json
import logging
import re
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Time-related keywords
_TIME_KEYWORDS = re.compile(
    r"(今何時|何時|時間|何日|何月|何曜日|今日は|今日の日付|いまなんじ|きょうは)"
)


def detect_time_query(text: str) -> bool:
    """Check if the text is asking about the current time or date."""
    return bool(_TIME_KEYWORDS.search(text))


def get_current_time_context() -> str:
    """Get the current time formatted for injection into LLM context."""
    now = datetime.datetime.now()
    weekdays = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
    return (
        f"現在の日時情報: {now.year}年{now.month}月{now.day}日 "
        f"{weekdays[now.weekday()]} "
        f"{now.hour}時{now.minute}分"
    )


def detect_search_request(text: str) -> str | None:
    """Check if the text contains a search request.

    Returns the search query if found, None otherwise.
    """
    if "調べて" not in text:
        return None
    # Extract query: remove "調べて" and common particles
    query = text.replace("調べて", "").strip()
    query = re.sub(r"^(を|について|って|の?こと)?", "", query)
    query = re.sub(r"(を|について|って|の?こと)?$", "", query)
    return query.strip() or None


# Currency keywords -> API currency code
_CURRENCY_MAP: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(ドル|どる)"), "USD", "ドル"),
    (re.compile(r"(ユーロ|ゆーろ)"), "EUR", "ユーロ"),
    (re.compile(r"(ポンド|ぽんど)"), "GBP", "ポンド"),
    (re.compile(r"(元|人民元|げん)"), "CNY", "人民元"),
    (re.compile(r"(ウォン|うぉん)"), "KRW", "ウォン"),
    (re.compile(r"(フラン|ふらん)"), "CHF", "フラン"),
    (re.compile(r"(豪ドル|オーストラリアドル)"), "AUD", "豪ドル"),
]

_RATE_KEYWORDS = re.compile(r"(いくら|何円|なんえん|レート|れーと|為替|かわせ)")


def detect_currency_query(text: str) -> tuple[str, str] | None:
    """Check if the text is asking about a currency exchange rate.

    Returns (currency_code, display_name) or None.
    """
    if not _RATE_KEYWORDS.search(text):
        return None
    for pattern, code, name in _CURRENCY_MAP:
        if pattern.search(text):
            return code, name
    return None


def fetch_currency_rate(currency_code: str) -> float | None:
    """Fetch the exchange rate from Frankfurter API. Blocking call."""
    url = f"https://api.frankfurter.dev/v2/rate/{currency_code}/JPY"
    req = urllib.request.Request(url, headers={"User-Agent": "ccxa/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("rate")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        logger.error("Currency rate fetch failed: %s", e)
        return None


def get_currency_context(currency_code: str, display_name: str) -> str:
    """Fetch rate and return context string for LLM."""
    rate = fetch_currency_rate(currency_code)
    if rate is None:
        return f"{display_name}の為替レートを取得できませんでした。"
    return (
        f"為替レート情報: 1{display_name}は現在{rate}円です。"
    )


# Weather: location name (Japanese) -> wttr.in location slug
_WEATHER_LOCATIONS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(草津|くさつ)"), "Kusatsu", "草津"),
    (re.compile(r"(京都|きょうと)"), "Kyoto", "京都"),
    (re.compile(r"(東京|とうきょう)"), "Tokyo", "東京"),
    (re.compile(r"(大阪|おおさか)"), "Osaka", "大阪"),
    (re.compile(r"(名古屋|なごや)"), "Nagoya", "名古屋"),
    (re.compile(r"(横浜|よこはま)"), "Yokohama", "横浜"),
    (re.compile(r"(神戸|こうべ)"), "Kobe", "神戸"),
    (re.compile(r"(福岡|ふくおか)"), "Fukuoka", "福岡"),
    (re.compile(r"(札幌|さっぽろ)"), "Sapporo", "札幌"),
    (re.compile(r"(仙台|せんだい)"), "Sendai", "仙台"),
    (re.compile(r"(広島|ひろしま)"), "Hiroshima", "広島"),
    (re.compile(r"(那覇|なは|沖縄|おきなわ)"), "Naha", "那覇"),
]

_WEATHER_KEYWORDS = re.compile(r"(天気|てんき|気温|きおん|雨|あめ|晴|はれ|曇|くもり|暑|寒)")

# English weather description -> Japanese
_WEATHER_DESC_JA: dict[str, str] = {
    "Clear": "晴れ",
    "Sunny": "晴れ",
    "Partly Cloudy": "一部曇り",
    "Partly cloudy": "一部曇り",
    "Cloudy": "曇り",
    "Overcast": "曇り",
    "Mist": "霧",
    "Fog": "霧",
    "Light rain": "小雨",
    "Light drizzle": "霧雨",
    "Moderate rain": "雨",
    "Heavy rain": "大雨",
    "Light snow": "小雪",
    "Moderate snow": "雪",
    "Heavy snow": "大雪",
    "Thunderstorm": "雷雨",
    "Patchy rain nearby": "一時雨",
    "Patchy rain possible": "一時雨の可能性",
    "Patchy snow possible": "一時雪の可能性",
    "Blowing snow": "吹雪",
    "Freezing fog": "凍霧",
    "Freezing drizzle": "凍る霧雨",
    "Light freezing rain": "軽い凍雨",
    "Light rain shower": "にわか雨",
    "Light sleet": "みぞれ",
    "Moderate or heavy rain shower": "強いにわか雨",
    "Torrential rain shower": "豪雨",
}


def detect_weather_query(text: str) -> tuple[str, str] | None:
    """Check if the text is asking about weather.

    Returns (wttr_slug, display_name) or None.
    Default location is Kusatsu if no specific city is mentioned.
    """
    if not _WEATHER_KEYWORDS.search(text):
        return None
    for pattern, slug, name in _WEATHER_LOCATIONS:
        if pattern.search(text):
            return slug, name
    # Default to Kusatsu if asking about weather without specifying a city
    return "Kusatsu", "草津"


def fetch_weather(location_slug: str) -> dict | None:
    """Fetch weather from wttr.in. Blocking call."""
    url = f"https://wttr.in/{location_slug}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "ccxa/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.error("Weather fetch failed: %s", e)
        return None


def get_weather_context(location_slug: str, display_name: str) -> str:
    """Fetch weather and return context string for LLM."""
    data = fetch_weather(location_slug)
    if data is None:
        return f"{display_name}の天気情報を取得できませんでした。"

    cur = data["current_condition"][0]
    desc_en = cur["weatherDesc"][0]["value"].strip()
    desc = _WEATHER_DESC_JA.get(desc_en, desc_en)
    temp = cur["temp_C"]
    feels = cur["FeelsLikeC"]
    humidity = cur["humidity"]

    # Today's high/low
    today_hl = ""
    if data.get("weather"):
        today = data["weather"][0]
        today_hl = f" 今日の最高{today.get('maxtempC', '?')}度、最低{today.get('mintempC', '?')}度。"

    # Tomorrow forecast
    forecast = ""
    if data.get("weather") and len(data["weather"]) >= 2:
        tmr = data["weather"][1]
        try:
            hourly = tmr.get("hourly", [])
            mid = hourly[len(hourly) // 2] if hourly else {}
            tmr_desc_en = mid.get("weatherDesc", [{}])[0].get("value", "").strip()
        except (IndexError, KeyError):
            tmr_desc_en = ""
        tmr_desc = _WEATHER_DESC_JA.get(tmr_desc_en, tmr_desc_en)
        tmr_max = tmr.get("maxtempC", "?")
        tmr_min = tmr.get("mintempC", "?")
        if tmr_desc:
            forecast = f" 明日は{tmr_desc}、最高{tmr_max}度、最低{tmr_min}度の予報です。"

    return (
        f"{display_name}の天気情報: 現在{desc}、気温{temp}度、"
        f"体感{feels}度、湿度{humidity}%。{today_hl}{forecast}"
    )


def detect_detailed_request(text: str) -> bool:
    """Check if the user is asking for a detailed explanation."""
    return any(kw in text for kw in ["詳しく", "もっと教えて", "詳細に", "くわしく"])


def detect_goodbye(text: str, goodbye_phrases: list[str]) -> bool:
    """Check if the user is saying goodbye."""
    return any(phrase in text for phrase in goodbye_phrases)
