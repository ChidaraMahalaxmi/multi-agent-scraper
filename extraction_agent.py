from __future__ import annotations

import os
import re
from typing import Any, Dict, List

try:
    from llm_client import call_llm_json, pop_warning
except ImportError:
    from .llm_client import call_llm_json, pop_warning

EXTRACTION_MODEL = os.getenv("OLLAMA_MODEL_EXTRACTOR", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))


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


def _normalize_columns(columns: List[str], goal: str) -> List[str]:
    if not columns:
        columns = _infer_columns_from_goal(goal)

    normalized: List[str] = []
    seen = set()
    for col in columns:
        name = _clean_column_name(str(col))
        if not name:
            continue
        lower = name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        normalized.append(name)

    has_source_url = any("source url" in c.lower() or c.lower() == "url" for c in normalized)
    if not has_source_url:
        normalized.append("Source URL")

    return normalized or ["Finding", "Source Title", "Source URL"]


def _normalize_rows(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = {}
        for col in columns:
            value = row.get(col, "NA")
            normalized[col] = value if value not in (None, "") else "NA"
        normalized_rows.append(normalized)
    return normalized_rows


def _infer_columns_from_goal(goal: str) -> List[str]:
    explicit = re.search(r"columns?\s*:\s*(.+)", goal, flags=re.IGNORECASE)
    if explicit:
        parts = [_clean_column_name(part) for part in _split_columns(explicit.group(1))]
        return [p for p in parts if p]

    extract_match = re.search(
        r"extract\s+(.+?)(?:\s+in\s+table\s+format|\s+as\s+table|$)",
        goal,
        flags=re.IGNORECASE,
    )
    if extract_match:
        raw = re.split(r"\.\s+", extract_match.group(1))[0]
        raw = re.split(r"\bif\s+missing\b", raw, flags=re.IGNORECASE)[0]
        raw = re.split(r"\buse\s+only\b", raw, flags=re.IGNORECASE)[0]
        raw = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
        parts = [_clean_column_name(part) for part in _split_columns(raw)]
        cols = [p for p in parts if p]
        if cols:
            return cols + ["Source URL"] if "source url" not in " ".join(c.lower() for c in cols) else cols

    return ["Finding", "Source Title", "Source URL"]


def _extract_signal(text: str, keywords: List[str]) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lowered = sentence.lower()
        if any(k in lowered for k in keywords):
            return sentence[:220]
    return "NA"


def _row_from_item(item: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    text = str(item.get("chunk", ""))
    row: Dict[str, Any] = {}
    for col in columns:
        key = col.lower().strip()
        if "source url" in key or key == "url":
            row[col] = item.get("url", "NA")
        elif "source title" in key or "title" == key:
            row[col] = item.get("title", "NA")
        elif "method" in key or "model" in key or "approach" in key:
            row[col] = _extract_signal(text, ["method", "model", "approach", "framework", "network", "algorithm"])
        elif "benchmark" in key or "dataset" in key:
            row[col] = _extract_signal(text, ["benchmark", "dataset", "guide-seq", "circle-seq", "site-seq"])
        elif "metric" in key or "performance" in key or "accuracy" in key:
            row[col] = _extract_signal(text, ["auc", "roc", "pr", "precision", "recall", "f1", "accuracy", "score"])
        elif "finding" in key:
            row[col] = text[:220] if text else "NA"
        else:
            row[col] = "NA"
    return row


def _default_rows(chunks: List[Dict[str, Any]], goal: str, columns: List[str]) -> Dict[str, Any]:
    resolved_columns = _normalize_columns(columns if columns else _infer_columns_from_goal(goal), goal)
    rows = []
    for item in chunks[:12]:
        rows.append(_row_from_item(item, resolved_columns))
    return {
        "columns": resolved_columns,
        "rows": rows,
    }


def extraction_node(state):
    goal = state["goal"]
    output_format = state["output_format"]
    columns = state["table_columns"]

    context_parts = []
    for item in state["retrieved_chunks"][:18]:
        context_parts.append(
            f"Source: {item['title']} ({item['url']})\n"
            f"Score: {item['score']:.2f}\n"
            f"Text: {item['chunk'][:900]}"
        )
    context_text = "\n\n".join(context_parts)

    if output_format == "table":
        fallback = _default_rows(state["retrieved_chunks"], goal=goal, columns=columns)
        prompt = f"""
User goal:
{goal}

Extract structured findings from the context.
Output must be a JSON object with:
- columns: string[]
- rows: object[]

Preferred columns: {fallback['columns']}
Use "NA" if a value is missing.

Context:
{context_text}
"""
        extracted = call_llm_json(
            prompt=prompt,
            schema_hint="{columns:string[], rows:object[]}",
            fallback=fallback,
            model_name=EXTRACTION_MODEL,
        )
        warning = pop_warning()
        if warning:
            state["errors"].append(warning)

        if not isinstance(extracted.get("rows"), list):
            extracted = fallback
        if not isinstance(extracted.get("columns"), list):
            extracted["columns"] = fallback["columns"]
        extracted["columns"] = _normalize_columns(extracted.get("columns", []), goal)
        extracted["rows"] = _normalize_rows(extracted.get("rows", []), extracted["columns"])

        state["extracted_output"] = extracted
        state["extracted_items"] = extracted.get("rows", [])
    else:
        fallback = {
            "format": output_format,
            "items": [
                {
                    "finding": item["chunk"][:220],
                    "source_title": item["title"],
                    "source_url": item["url"],
                }
                for item in state["retrieved_chunks"][:12]
            ],
        }

        prompt = f"""
User goal:
{goal}

Requested output format: {output_format}

Return JSON:
- format: string
- items: array of extracted findings

Context:
{context_text}
"""
        extracted = call_llm_json(
            prompt=prompt,
            schema_hint="{format:string, items:object[]}",
            fallback=fallback,
            model_name=EXTRACTION_MODEL,
        )
        warning = pop_warning()
        if warning:
            state["errors"].append(warning)

        items = extracted.get("items") if isinstance(extracted, dict) else []
        if not isinstance(items, list):
            extracted = fallback
            items = fallback["items"]

        state["extracted_output"] = extracted
        state["extracted_items"] = items

    count_score = min(len(state["extracted_items"]) / 10.0, 1.0)
    state["agent_confidences"]["extraction"] = count_score
    return state
