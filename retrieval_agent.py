from __future__ import annotations

import os
import re
from typing import Dict, List

try:
    from llm_client import call_llm_json, pop_warning
except ImportError:
    from .llm_client import call_llm_json, pop_warning

MAX_CHUNKS = max(5, int(os.getenv("MAX_CHUNKS", "30")))
RETRIEVAL_LLM_SCORING = os.getenv("RETRIEVAL_LLM_SCORING", "0") == "1"
MAX_LLM_CHUNKS_PER_DOC = max(1, int(os.getenv("MAX_LLM_CHUNKS_PER_DOC", "3")))
RETRIEVAL_MODEL = os.getenv("OLLAMA_MODEL_RETRIEVAL", os.getenv("OLLAMA_MODEL", "llama3.2:1b"))


def _chunk_text(text: str, chunk_size: int = 1800) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 <= chunk_size:
            current = f"{current}\n{line}".strip()
        else:
            if current:
                chunks.append(current)
            current = line

    if current:
        chunks.append(current)

    return chunks


def _keyword_score(chunk: str, goal: str) -> float:
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", goal) if len(t) > 2]
    terms = list(dict.fromkeys(terms))
    if not terms:
        return 0.0
    lowered = chunk.lower()
    hits = sum(1 for term in terms if term in lowered)
    return hits / len(terms)


def _semantic_score(chunk: str, goal: str, fallback: float) -> float:
    prompt = f"""
Task:
Score how useful this text chunk is for answering the user request.

User request:
{goal}

Chunk:
{chunk[:1200]}

Return JSON: {{"score": number between 0 and 1}}
"""
    parsed = call_llm_json(
        prompt=prompt,
        schema_hint="{score:number}",
        fallback={"score": fallback},
        model_name=RETRIEVAL_MODEL,
    )
    score = float(parsed.get("score", fallback) or fallback)
    return max(0.0, min(score, 1.0))


def retrieval_node(state):
    goal = state["goal"]
    retrieved: List[Dict] = []
    top_scores = []

    total_docs = len(state["scraped_docs"])
    for doc_idx, doc in enumerate(state["scraped_docs"], start=1):
        print(
            f"[RETRIEVAL] scoring doc {doc_idx}/{total_docs}: {doc.get('title', '')[:80]}",
            flush=True,
        )
        chunks = _chunk_text(doc["full_text"])
        scored = []
        llm_calls_used = 0

        for chunk in chunks:
            kscore = _keyword_score(chunk, goal)
            if kscore < 0.08:
                continue
            if RETRIEVAL_LLM_SCORING and llm_calls_used < MAX_LLM_CHUNKS_PER_DOC:
                sscore = _semantic_score(chunk, goal, fallback=kscore)
                llm_calls_used += 1
                warning = pop_warning()
                if warning:
                    state["errors"].append(warning)
            else:
                sscore = kscore
            score = 0.5 * kscore + 0.5 * sscore
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        for score, chunk in scored[:4]:
            retrieved.append(
                {
                    "paper_id": doc["paper_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "source": doc["source"],
                    "score": score,
                    "chunk": chunk,
                }
            )
            top_scores.append(score)

    retrieved.sort(key=lambda item: item["score"], reverse=True)
    state["retrieved_chunks"] = retrieved[:MAX_CHUNKS]

    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    state["agent_confidences"]["retrieval"] = avg_score
    return state
