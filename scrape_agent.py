from __future__ import annotations

import os
import re
from typing import List

import requests
from bs4 import BeautifulSoup


HEADERS = {"User-Agent": "Mozilla/5.0"}
SCRAPE_USE_PLAYWRIGHT = os.getenv("SCRAPE_USE_PLAYWRIGHT", "0") == "1"
SCRAPE_PLAYWRIGHT_FIRST = os.getenv("SCRAPE_PLAYWRIGHT_FIRST", "0") == "1"

BLOCK_MARKERS = [
    "captcha",
    "verify you are human",
    "robot",
    "access denied",
    "forbidden",
    "cf-challenge",
]


def _clean_full_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "svg"]):
        tag.decompose()

    parts: List[str] = []
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text) >= 40:
            parts.append(text)

    return "\n".join(parts)


def _fetch_requests(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200 and "text/html" in response.headers.get("content-type", ""):
            return response.text
    except Exception:
        return ""
    return ""


def _fetch_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ""

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(700)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return ""


def _looks_blocked(html: str) -> bool:
    if not html:
        return True
    lowered = html.lower()
    return any(marker in lowered for marker in BLOCK_MARKERS)


def scrape_node(state):
    docs = []
    use_playwright = os.getenv("SCRAPE_USE_PLAYWRIGHT", "0") == "1"
    playwright_first = os.getenv("SCRAPE_PLAYWRIGHT_FIRST", "0") == "1"

    for paper in state["filtered_papers"]:
        url = paper.get("html_link", "")

        html = ""
        if use_playwright and playwright_first:
            html = _fetch_playwright(url)
            if _looks_blocked(html):
                html = _fetch_requests(url)
        else:
            html = _fetch_requests(url)
            if (not html or _looks_blocked(html)) and use_playwright:
                html = _fetch_playwright(url)

        if not html:
            continue

        full_text = _clean_full_text(html)
        if len(full_text) < 600:
            continue

        docs.append(
            {
                "paper_id": paper.get("id"),
                "title": paper.get("title", ""),
                "source": paper.get("source", "arxiv"),
                "url": url,
                "full_text": full_text,
                "char_count": len(full_text),
            }
        )

    state["scraped_docs"] = docs

    if docs:
        coverage = min(sum(doc["char_count"] for doc in docs) / (len(docs) * 5000.0), 1.0)
    else:
        coverage = 0.0

    state["agent_confidences"]["scrape"] = coverage
    if not use_playwright:
        state["agent_confidences"]["scrape_mode"] = 1.0
    elif playwright_first:
        state["agent_confidences"]["scrape_mode"] = 0.95
    else:
        state["agent_confidences"]["scrape_mode"] = 0.9
    return state
