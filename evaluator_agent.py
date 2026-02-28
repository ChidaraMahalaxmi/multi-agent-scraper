from __future__ import annotations


def evaluate_node(state):
    extracted = state["extracted_items"]
    retrieved = state["retrieved_chunks"]
    docs = state["scraped_docs"]

    unique_sources = len({item.get("url", "") for item in retrieved if item.get("url")})
    source_diversity = min(unique_sources / 6.0, 1.0)

    avg_chars = (sum(doc.get("char_count", 0) for doc in docs) / len(docs)) if docs else 0.0
    content_coverage = min(avg_chars / 5000.0, 1.0)

    extraction_volume = min(len(extracted) / 10.0, 1.0)

    state["diversity_score"] = source_diversity
    state["content_coverage"] = content_coverage

    confidences = state.get("agent_confidences", {})
    if confidences:
        agent_avg = sum(confidences.values()) / len(confidences)
    else:
        agent_avg = 0.0

    state["agent_confidences"]["evaluator"] = (0.4 * source_diversity) + (0.3 * content_coverage) + (0.3 * extraction_volume)
    state["global_confidence"] = (
        0.45 * agent_avg
        + 0.25 * source_diversity
        + 0.15 * content_coverage
        + 0.15 * extraction_volume
    )

    return state
