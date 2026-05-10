from pipeline.extraction.extractor import extract_event
from pipeline.grounding.normalizer import normalize_event
from pipeline.propagation.propagator import propagate

from graph.graph_engine import (
    load_causal_graph,
    get_all_source_nodes,
)


GRAPH, RULES = load_causal_graph()
AVAILABLE_NODES = get_all_source_nodes(RULES)


def analyze_article(article: str):
    extracted = extract_event(article)

    canonical_node = normalize_event(
        extracted["raw_event"],
        AVAILABLE_NODES,
    )

    causal_chain = propagate(GRAPH, canonical_node)

    if causal_chain:
        causal_chain[0]["domain"] = extracted["domain"]

    edges = []

    for i, node in enumerate(causal_chain):
        if node["parent"] is not None:
            edges.append(
                {
                    "data": {
                        "id": f"e{i}",
                        "source": node["parent"],
                        "target": node["id"],
                        "strength": node["edge_strength"],
                        "label": f"{int(node['probability'] * 100)}%",
                    }
                }
            )

    total_nodes = len(causal_chain)

    max_depth = max((n["hop"] for n in causal_chain), default=0)

    avg_confidence = (
        sum(n["score"] for n in causal_chain) / total_nodes
        if total_nodes
        else 0.0
    )

    return {
        "root_event": extracted["raw_event"],
        "canonical_node": canonical_node,
        "domain": extracted["domain"],
        "entities": extracted["entities"],
        "causal_chain": causal_chain,
        "edges": edges,
        "stats": {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "avg_confidence": round(avg_confidence, 4),
        },
    }