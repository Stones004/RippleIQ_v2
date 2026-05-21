# pyrefly: ignore [missing-import]
from networkx.generators import spectral_graph_forge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# pyrefly: ignore [missing-import]
import numpy as np
import json
from pathlib import Path
import os

from graph.graph_engine import (
            load_causal_graph,
            get_all_source_nodes,
        )


class SemanticGrounder:

    def __init__(self, graph_path: str):

        self.model = SentenceTransformer(
            "BAAI/bge-large-en-v1.5"
        )

        self.graph_path = graph_path
        self.embedding_dir = Path("data/embeddings")
        self.embedding_path = (
            self.embedding_dir / "node_embeddings.npy"
        )
        self.nodes_path = (
            self.embedding_dir / "nodes.json"
        )

        self.canonical_nodes = self.load_nodes()

        self.node_embeddings = self.load_or_create_embeddings()


    def normalize_node_text(self, node: str):

        return (
            node
            .replace("_", " ")
            .lower()
            .strip()
        )
        
    def load_nodes(self):

        _, rules = load_causal_graph()
        nodes = get_all_source_nodes(rules)

        return sorted(list(nodes))

    def generate_node_embeddings(self):

        normalized_nodes = [

            self.normalize_node_text(node)

            for node in self.canonical_nodes
        ]

        embeddings = self.model.encode(
            normalized_nodes,
            normalize_embeddings=True
        )

        return embeddings

    def ground_event(
        self,
        raw_event: str,
        threshold: float = 0.55
    ):

        normalized_query = self.normalize_node_text(raw_event)

        query_embedding = self.model.encode(
            [normalized_query],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.node_embeddings
        )[0]

        best_idx = np.argmax(similarities)

        best_score = similarities[best_idx]

        best_node = self.canonical_nodes[best_idx]

        print("\n===== SEMANTIC GROUNDING =====")
        print(f"RAW EVENT: {raw_event}")
        print(f"MATCHED NODE: {best_node}")
        print(f"SIMILARITY: {best_score:.4f}")
        print("==============================\n")

        if best_score < threshold:
            return {
                "matched": False,
                "node": None,
                "score": float(best_score)
            }

        return {
            "matched": True,
            "node": best_node,
            "score": float(best_score)
        }

    def save_embeddings(self):

        np.save(
            "data/embeddings/node_embeddings.npy",
            self.node_embeddings
        )

        with open(
            "data/embeddings/nodes.json",
            "w"
        ) as f:

            json.dump(self.canonical_nodes, f)
    
    def load_or_create_embeddings(self):

        self.embedding_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            self.embedding_path.exists()
            and self.nodes_path.exists()
        ):

            print("[CACHE] Loading cached embeddings...")

            self.node_embeddings = np.load(
                self.embedding_path
            )

            with open(self.nodes_path, "r") as f:

                self.canonical_nodes = json.load(f)

            return self.node_embeddings

        current_nodes = self.load_nodes()

        if len(current_nodes) != len(self.canonical_nodes):

            print("[CACHE] Node count mismatch.")
            print("[CACHE] Regenerating embeddings.")

            self.canonical_nodes = current_nodes

            embeddings = self.generate_node_embeddings()

            np.save(self.embedding_path, embeddings)

            with open(self.nodes_path, "w") as f:
                json.dump(
                    self.canonical_nodes,
                    f,
                    indent=2
                )

            return embeddings

        print("[CACHE] Generating new embeddings...")

        embeddings = self.generate_node_embeddings()

        np.save(
            self.embedding_path,
            embeddings
        )

        with open(self.nodes_path, "w") as f:

            json.dump(
                self.canonical_nodes,
                f,
                indent=2
            )

        print("[CACHE] Embeddings saved.")

        return embeddings