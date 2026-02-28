from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import requests

try:
    from llm_client import call_llm_json, pop_warning
except ImportError:
    from .llm_client import call_llm_json, pop_warning


BLOCK_STATUSES = {401, 403, 407, 429, 503}
FILTER_MODEL = os.getenv("OLLAMA_MODEL_FILTER", os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
BLOCKED_HOST_KEYWORDS = {"github.com", "rth.dk", "reddit.com", "youtube.com"}
BLOCKED_PATH_KEYWORDS = {"/about", "/help", "/docs", "/faq", "/blog"}
NON_HTML_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".zip"}


def _keyword_score(text: str, query: str) -> float:
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", query) if len(t) > 2]
    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for term in unique_terms if term in lowered)
    return hits / len(unique_terms)


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _status_obstruction(url: str) -> str:
    try:
        response = requests.head(url, timeout=8, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code in BLOCK_STATUSES:
            return f"status-{response.status_code}"
    except Exception:
        return "head-failed"
    return ""


def _is_noise_or_nonpaper_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if any(keyword in host for keyword in BLOCKED_HOST_KEYWORDS):
        return True
    if any(path.startswith(prefix) for prefix in BLOCKED_PATH_KEYWORDS):
        return True
    if any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS):
        return True
    return False


def _llm_relevant(paper: dict, goal: str, fallback_score: float) -> tuple[bool, float]:
    prompt = f"""
User goal:
{goal}

Title:
{paper.get("title", "")}

Abstract/Snippet:
{paper.get("summary", "")}

Return JSON:
{{"relevant": true/false, "score": number between 0 and 1}}
"""
    parsed = call_llm_json(
        prompt=prompt,
        schema_hint="{relevant:boolean, score:number}",
        fallback={"relevant": fallback_score >= 0.2, "score": fallback_score},
        model_name=FILTER_MODEL,
    )

    score = float(parsed.get("score", fallback_score) or fallback_score)
    relevant = bool(parsed.get("relevant", score >= 0.2))
    return relevant, max(0.0, min(score, 1.0))


def filter_node(state):
    goal = state["goal"]
    query = state["search_query"]

    filtered = []
    audits = []
    semantic_scores = []

    for paper in state["candidate_papers"]:
        url = str(paper.get("html_link", "")).strip()
        title = str(paper.get("title", "")).strip()
        summary = str(paper.get("summary", "")).strip()

        audit = {
            "url": url,
            "title": title,
            "keyword_score": 0.0,
            "semantic_score": 0.0,
            "relevant": False,
            "obstruction": "",
        }

        if not _is_valid_url(url):
            audit["obstruction"] = "invalid-url"
            audits.append(audit)
            continue
        if _is_noise_or_nonpaper_url(url):
            audit["obstruction"] = "filtered-nonpaper-source"
            audits.append(audit)
            continue

        combined = f"{title} {summary}"
        keyword_score = _keyword_score(combined, query)
        audit["keyword_score"] = keyword_score

        if keyword_score < 0.08:
            audits.append(audit)
            continue

        relevant, semantic_score = _llm_relevant(paper=paper, goal=goal, fallback_score=keyword_score)
        warning = pop_warning()
        if warning:
            state["errors"].append(warning)
        audit["semantic_score"] = semantic_score

        if not relevant:
            audits.append(audit)
            continue

        obstruction = _status_obstruction(url)
        if obstruction:
            audit["obstruction"] = obstruction
            audits.append(audit)
            continue

        audit["relevant"] = True
        filtered.append(paper)
        semantic_scores.append(semantic_score)
        audits.append(audit)

    state["filtered_papers"] = filtered
    state["url_audit"] = audits

    pass_rate = (len(filtered) / len(state["candidate_papers"])) if state["candidate_papers"] else 0.0
    semantic_avg = (sum(semantic_scores) / len(semantic_scores)) if semantic_scores else 0.0
    state["agent_confidences"]["filter"] = max(0.0, min((0.6 * pass_rate) + (0.4 * semantic_avg), 1.0))

    return state
