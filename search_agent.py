from __future__ import annotations

import os
from typing import Dict, List

try:
    from duckduckgo_search import search_duckduckgo
except ImportError:
    from .duckduckgo_search import search_duckduckgo


MAX_PAPERS = max(1, int(os.getenv("MAX_PAPERS", "12")))


def _dedupe_by_title(papers: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()
    for paper in papers:
        title = str(paper.get("title", "")).strip().lower()
        if not title or title in seen:
            continue
        seen.add(title)
        deduped.append(paper)
    return deduped

def search_node(state):
    query = state["search_query"]

    web_results = search_duckduckgo(query, max_results=MAX_PAPERS)
    merged = _dedupe_by_title(web_results)[:MAX_PAPERS]

    state["candidate_papers"] = merged
    state["agent_confidences"]["search"] = min(len(merged) / float(MAX_PAPERS), 1.0)
    return state
