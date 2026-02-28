from __future__ import annotations

import json
import os
import sys

from langgraph.graph import END, StateGraph

try:
    from evaluator_agent import evaluate_node
    from extraction_agent import extraction_node
    from filter_agent import filter_node
    from orchestrator_agent import orchestrator_node, orchestrator_router
    from planner_agent import planner_node
    from retrieval_agent import retrieval_node
    from scrape_agent import scrape_node
    from search_agent import search_node
    from state import AgentState, make_initial_state
except ImportError:
    from .evaluator_agent import evaluate_node
    from .extraction_agent import extraction_node
    from .filter_agent import filter_node
    from .orchestrator_agent import orchestrator_node, orchestrator_router
    from .planner_agent import planner_node
    from .retrieval_agent import retrieval_node
    from .scrape_agent import scrape_node
    from .search_agent import search_node
    from .state import AgentState, make_initial_state



def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("orchestrator", orchestrator_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "scrape")
    workflow.add_edge("scrape", "retrieval")
    workflow.add_edge("retrieval", "extraction")
    workflow.add_edge("extraction", "evaluate")
    workflow.add_edge("evaluate", "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "search": "search",
            "__end__": END,
        },
    )

    return workflow.compile()


def run_pipeline(goal: str, max_iterations: int = 2, verbose: bool = False):
    app = build_app()
    state = make_initial_state(goal=goal, max_iterations=max_iterations)
    if not verbose:
        return app.invoke(state)

    final_state = state
    for event in app.stream(state):
        if not isinstance(event, dict):
            continue
        for node_name, node_state in event.items():
            if node_name == "__end__":
                continue
            print(f"[NODE] {node_name} completed", flush=True)
            if isinstance(node_state, dict):
                final_state = node_state
    return final_state


if __name__ == "__main__":
    user_goal = "Extract PROTACs and linkers from 2025 in table format"
    if len(sys.argv) > 1:
        user_goal = " ".join(sys.argv[1:]).strip()

    max_iterations = int(os.getenv("MAX_ITERATIONS", "2"))
    result = run_pipeline(user_goal, max_iterations=max_iterations)

    print("FINAL_RESULT_START")
    print(
        json.dumps(
            {
                "goal": result["goal"],
                "search_query": result["search_query"],
                "output_format": result["output_format"],
                "result": result["extracted_output"],
                "agent_confidences": result["agent_confidences"],
                "diversity_score": result["diversity_score"],
                "content_coverage": result["content_coverage"],
                "global_confidence": result["global_confidence"],
                "iterations_used": result["iteration"],
            },
            ensure_ascii=False,
        )
    )
    print("FINAL_RESULT_END")
