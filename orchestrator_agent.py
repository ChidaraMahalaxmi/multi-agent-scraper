from __future__ import annotations

import re


def _refine_query(goal: str, query: str) -> str:
    goal_terms = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", goal) if len(t) > 3]
    stop = {"find", "from", "with", "table", "format", "source", "url", "recent", "papers", "paper"}
    missing = []
    for term in goal_terms:
        if term in stop:
            continue
        if term not in query.lower() and len(missing) < 2:
            missing.append(term)

    if not missing:
        return query

    additions = " ".join(missing)
    return f"{query} {additions}".strip() if query else additions


def orchestrator_node(state):
    threshold = 0.68
    max_iterations = state["max_iterations"]

    if state["global_confidence"] >= threshold:
        state["orchestrator_action"] = "stop"
        return state

    if state["iteration"] >= max_iterations:
        state["orchestrator_action"] = "stop"
        return state

    state["search_query"] = _refine_query(state["goal"], state["search_query"])
    state["iteration"] += 1
    state["orchestrator_action"] = "research_more"
    return state



def orchestrator_router(state):
    return "search" if state.get("orchestrator_action") == "research_more" else "__end__"
