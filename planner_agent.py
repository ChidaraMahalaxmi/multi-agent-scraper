from __future__ import annotations

import os
import re
from typing import List

try:
    from llm_client import call_llm_json, pop_warning
except ImportError:
    from .llm_client import call_llm_json, pop_warning


STOPWORDS = {
    "extract",
    "find",
    "from",
    "on",
    "in",
    "to",
    "for",
    "the",
    "and",
    "or",
    "with",
    "about",
    "papers",
    "paper",
    "latest",
    "recent",
    "information",
    "data",
    "table",
    "format",
    "source",
    "url",
    "urls",
}
PLANNER_MODEL = os.getenv("OLLAMA_MODEL_PLANNER", os.getenv("OLLAMA_MODEL", "llama3.2:1b"))


def _split_columns(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1

        if ch == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
            continue
        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _clean_column_name(value: str) -> str:
    cleaned = value.strip(" .;:")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+Use only.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*if\s+missing.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*fill\s+na.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _fallback_query(goal: str) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z0-9\-]+", goal) if t.lower() not in STOPWORDS]
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9\-]+", goal)
    return " ".join(tokens[:8])


def _infer_output_format(goal: str) -> str:
    lowered = goal.lower()
    if "markdown" in lowered:
        return "markdown"
    if "json" in lowered:
        return "json"
    if "list" in lowered:
        return "list"
    return "table"


def _infer_columns(goal: str) -> List[str]:
    match = re.search(r"columns?\s*:\s*(.+)", goal, flags=re.IGNORECASE)
    if match:
        parts = [_clean_column_name(part) for part in _split_columns(match.group(1))]
        return [p for p in parts if p]

    # If user says "extract X, Y, and Z", infer table columns from that intent.
    extract_match = re.search(
        r"extract\s+(.+?)(?:\s+in\s+table\s+format|\s+as\s+table|$)",
        goal,
        flags=re.IGNORECASE,
    )
    if not extract_match:
        return []

    raw = re.split(r"\.\s+", extract_match.group(1))[0]
    raw = re.split(r"\bif\s+missing\b", raw, flags=re.IGNORECASE)[0]
    raw = re.split(r"\buse\s+only\b", raw, flags=re.IGNORECASE)[0]
    raw = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
    raw = raw.replace("/", " / ")
    parts = [_clean_column_name(part) for part in _split_columns(raw)]
    return [p for p in parts if p and len(p) > 1]


def planner_node(state):
    goal = state["goal"]

    fallback = {
        "search_query": _fallback_query(goal),
        "output_format": _infer_output_format(goal),
        "table_columns": _infer_columns(goal),
    }

    prompt = f"""
User request:
{goal}

Create a research plan for paper search.
Return JSON with keys:
- search_query: concise web-search query string for DuckDuckGo
- output_format: one of table/json/list/markdown
- table_columns: array of column names for table output
"""

    planned = call_llm_json(
        prompt=prompt,
        schema_hint="{search_query:string, output_format:string, table_columns:string[]}",
        fallback=fallback,
        model_name=PLANNER_MODEL,
    )
    warning = pop_warning()
    if warning:
        state["errors"].append(warning)

    state["search_query"] = str(planned.get("search_query") or fallback["search_query"]).strip()

    output_format = str(planned.get("output_format") or fallback["output_format"]).strip().lower()
    if output_format not in {"table", "json", "list", "markdown"}:
        output_format = "table"
    state["output_format"] = output_format

    table_columns = planned.get("table_columns")
    if not isinstance(table_columns, list):
        table_columns = fallback["table_columns"]
    state["table_columns"] = [str(col).strip() for col in table_columns if str(col).strip()]

    state["agent_confidences"]["planner"] = 0.8 if state["search_query"] else 0.35
    return state
