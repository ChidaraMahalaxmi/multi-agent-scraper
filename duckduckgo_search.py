from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _is_http_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_noise_result(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if "duckduckgo.com" in host:
        return True
    if path.endswith((".js", ".css", ".xml")):
        return True
    return False


def search_duckduckgo(query: str, max_results: int = 10, timeout: int = 15):
    if not query:
        return []

    try:
        response = requests.post(
            SEARCH_URL,
            data={"q": query},
            headers=HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for block in soup.select(".result"):
        link_tag = block.select_one("a.result__a")
        if not link_tag:
            continue

        href = link_tag.get("href", "").strip()
        if not _is_http_url(href):
            continue
        if _is_noise_result(href):
            continue

        title = link_tag.get_text(" ", strip=True)
        snippet_tag = block.select_one(".result__snippet")
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""

        result_id = href.rstrip("/").split("/")[-1] or str(len(results) + 1)
        results.append(
            {
                "source": "duckduckgo",
                "id": result_id,
                "title": title,
                "summary": snippet,
                "published": "",
                "authors": [],
                "html_link": href,
                "pdf_link": "",
            }
        )

        if len(results) >= max_results:
            break

    return results

