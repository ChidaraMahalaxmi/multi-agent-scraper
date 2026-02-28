from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiagentscraper locally from VS Code terminal.")
    parser.add_argument(
        "--goal",
        required=True,
        help="User extraction goal. Example: 'Extract PROTACs and linkers from 2025 in table format'",
    )
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("MAX_ITERATIONS", "1")))
    parser.add_argument("--max-papers", type=int, default=int(os.getenv("MAX_PAPERS", "8")))
    parser.add_argument("--max-chunks", type=int, default=int(os.getenv("MAX_CHUNKS", "20")))
    parser.add_argument(
        "--retrieval-llm-scoring",
        action="store_true",
        help="Enable LLM semantic scoring for retrieval chunks (slower).",
    )
    parser.add_argument("--max-llm-chunks-per-doc", type=int, default=int(os.getenv("MAX_LLM_CHUNKS_PER_DOC", "2")))
    parser.add_argument("--ollama-chat-timeout", type=int, default=int(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "35")))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--model-planner", default=os.getenv("OLLAMA_MODEL_PLANNER", "llama3.2:1b"))
    parser.add_argument("--model-filter", default=os.getenv("OLLAMA_MODEL_FILTER", "llama3.2:1b"))
    parser.add_argument("--model-retrieval", default=os.getenv("OLLAMA_MODEL_RETRIEVAL", "llama3.2:1b"))
    parser.add_argument("--model-extractor", default=os.getenv("OLLAMA_MODEL_EXTRACTOR", "llama3.2:3b"))
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Enable Playwright fallback scraping (optional, slower).",
    )
    parser.add_argument(
        "--playwright-first",
        action="store_true",
        help="Try Playwright first, then fallback to requests when blocked/empty.",
    )
    parser.add_argument("--save-json", default="result.json", help="Path to save full result JSON.")
    parser.add_argument(
        "--save-table",
        default="result_table.csv",
        help="Path to save extracted tabular output as CSV.",
    )
    return parser


def _extract_rows(result: dict) -> list:
    extracted = result.get("extracted_output")
    if isinstance(extracted, dict) and isinstance(extracted.get("rows"), list):
        return extracted["rows"]
    items = result.get("extracted_items")
    if isinstance(items, list):
        return [i for i in items if isinstance(i, dict)]
    return []


def _write_table_csv(path: str, rows: list, columns: list) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return

    if not columns:
        columns = sorted({k for row in rows for k in row.keys()})

    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()

    os.environ["MAX_ITERATIONS"] = str(max(1, args.max_iterations))
    os.environ["MAX_PAPERS"] = str(max(1, args.max_papers))
    os.environ["MAX_CHUNKS"] = str(max(5, args.max_chunks))
    os.environ["RETRIEVAL_LLM_SCORING"] = "1" if args.retrieval_llm_scoring else "0"
    os.environ["MAX_LLM_CHUNKS_PER_DOC"] = str(max(1, args.max_llm_chunks_per_doc))
    os.environ["OLLAMA_CHAT_TIMEOUT_SECONDS"] = str(max(10, args.ollama_chat_timeout))
    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_MODEL_PLANNER"] = args.model_planner
    os.environ["OLLAMA_MODEL_FILTER"] = args.model_filter
    os.environ["OLLAMA_MODEL_RETRIEVAL"] = args.model_retrieval
    os.environ["OLLAMA_MODEL_EXTRACTOR"] = args.model_extractor
    os.environ["OLLAMA_HOST"] = args.ollama_host
    os.environ["SCRAPE_USE_PLAYWRIGHT"] = "1" if args.use_playwright else "0"
    os.environ["SCRAPE_PLAYWRIGHT_FIRST"] = "1" if args.playwright_first else "0"

    try:
        from multi_agent_runner import run_pipeline
    except ImportError:
        from .multi_agent_runner import run_pipeline

    print("[INFO] Starting pipeline...", flush=True)
    print(
        f"[INFO] model={os.environ['OLLAMA_MODEL']} host={os.environ['OLLAMA_HOST']} "
        f"max_iterations={os.environ['MAX_ITERATIONS']} max_papers={os.environ['MAX_PAPERS']} "
        f"max_chunks={os.environ['MAX_CHUNKS']} retrieval_llm={os.environ['RETRIEVAL_LLM_SCORING']} "
        f"playwright={os.environ['SCRAPE_USE_PLAYWRIGHT']} playwright_first={os.environ['SCRAPE_PLAYWRIGHT_FIRST']} "
        f"chat_timeout={os.environ['OLLAMA_CHAT_TIMEOUT_SECONDS']}s",
        flush=True,
    )
    print(
        f"[INFO] models planner={os.environ['OLLAMA_MODEL_PLANNER']} "
        f"filter={os.environ['OLLAMA_MODEL_FILTER']} "
        f"retrieval={os.environ['OLLAMA_MODEL_RETRIEVAL']} "
        f"extractor={os.environ['OLLAMA_MODEL_EXTRACTOR']}",
        flush=True,
    )

    result = run_pipeline(goal=args.goal, max_iterations=max(1, args.max_iterations), verbose=True)

    extracted = result.get("extracted_output") if isinstance(result.get("extracted_output"), dict) else {}
    columns = extracted.get("columns") if isinstance(extracted.get("columns"), list) else []
    table_rows = _extract_rows(result)
    rows = len(table_rows)

    summary = {
        "goal": result.get("goal"),
        "search_query": result.get("search_query"),
        "global_confidence": result.get("global_confidence"),
        "rows": rows,
        "iterations_used": result.get("iteration"),
        "errors": result.get("errors", [])[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved full result to: {args.save_json}")

    if args.save_table:
        _write_table_csv(args.save_table, table_rows, columns)
        print(f"Saved table output to: {args.save_table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
