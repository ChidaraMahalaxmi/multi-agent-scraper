from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    goal: str
    search_query: str
    output_format: str
    table_columns: List[str]

    candidate_papers: List[Dict[str, Any]]
    filtered_papers: List[Dict[str, Any]]
    url_audit: List[Dict[str, Any]]
    scraped_docs: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]

    extracted_items: List[Dict[str, Any]]
    extracted_output: Any

    agent_confidences: Dict[str, float]
    diversity_score: float
    content_coverage: float
    global_confidence: float

    iteration: int
    max_iterations: int
    orchestrator_action: str
    errors: List[str]


def make_initial_state(goal: str, max_iterations: int = 2) -> AgentState:
    return {
        "goal": goal,
        "search_query": "",
        "output_format": "table",
        "table_columns": [],
        "candidate_papers": [],
        "filtered_papers": [],
        "url_audit": [],
        "scraped_docs": [],
        "retrieved_chunks": [],
        "extracted_items": [],
        "extracted_output": {"columns": [], "rows": []},
        "agent_confidences": {},
        "diversity_score": 0.0,
        "content_coverage": 0.0,
        "global_confidence": 0.0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "orchestrator_action": "",
        "errors": [],
    }
