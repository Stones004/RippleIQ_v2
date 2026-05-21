from pipeline import grounding
from pipeline.extraction.extractor import extract_event
from pipeline.grounding.normalizer import normalize_event
from pipeline.propagation.propagator import propagate
from pipeline.grounding.semantic_grounder import SemanticGrounder


import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from memory.precedent_search import (
    PrecedentSearch
)

from graph.graph_engine import (
    load_causal_graph,
    get_all_source_nodes,
)


GRAPH, RULES = load_causal_graph()
AVAILABLE_NODES = get_all_source_nodes(RULES)


PRECEDENT_SEARCH = PrecedentSearch()


SEMANTIC_GROUNDER = SemanticGrounder(
    graph_path="data/graphs/causal_rules.json"
)


def analyze_article(article: str):
    extracted = extract_event(article)

    grounding = SEMANTIC_GROUNDER.ground_event(
        extracted["raw_event"]
    )

    if not grounding["matched"]:

        return {
            "error": "Unable to confidently ground event",
            "root_event": extracted["raw_event"],
            "grounding_score": grounding["score"]
        }
    
    precedents = PRECEDENT_SEARCH.search(
        extracted["raw_event"],
    )
    
    canonical_node = grounding["node"]
    
    if canonical_node is None:

        return {
            "error": "Unable to ground event",
            "root_event": extracted["raw_event"]
        }

    causal_chain = propagate(
        graph=GRAPH,
        root_node=canonical_node,
        precedents=precedents,
        context={
            "domain":
                extracted["domain"]
        }
    )

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

    print(
        f"[GROUNDING] "
        f"{extracted['raw_event']} "
        f"→ {canonical_node}"
    )

    return {
        "root_event": extracted["raw_event"],
        "canonical_node": canonical_node,
        "grounding": grounding,
        "precedents": precedents,
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