"""Web search via DuckDuckGo."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WebSearch:
    """DuckDuckGo web search wrapper."""

    def __init__(self, max_results: int = 3) -> None:
        self._max_results = max_results

    def search(self, query: str) -> str:
        """Execute search and return formatted results. Blocking call."""
        from ddgs import DDGS

        logger.info("Web search: '%s'", query)
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(query, max_results=self._max_results, region="jp-jp")
                )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return "検索に失敗しました。"

        if not results:
            return "検索結果が見つかりませんでした。"

        formatted = []
        for r in results:
            formatted.append(f"タイトル: {r['title']}\n内容: {r['body']}")
        return "\n---\n".join(formatted)
