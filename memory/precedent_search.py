from sklearn.metrics.pairwise import (
    cosine_similarity
)

from sentence_transformers import (
    SentenceTransformer
)

#from vector_store import VectorStore
from memory.vector_store import VectorStore

class PrecedentSearch:

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-large-en-v1.5"
        )

        store = VectorStore()

        self.embeddings, self.events = (
            store.load_embeddings()
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = (
            similarities.argsort()[-top_k:][::-1]
        )

        results = []

        for idx in top_indices:

            results.append({

                "title":
                    self.events[idx]["title"],

                "canonical_node":
                    self.events[idx]["canonical_node"],

                "canonical_impacts":
                    self.events[idx].get(
                        "impacts",
                        []
                    ),

                "score":
                    float(similarities[idx]),

                "summary":
                    self.events[idx]["summary"],

                "confidence":
                    self.events[idx].get(
                        "confidence",
                        0.8
                    )
            })
        
        print("\n===== FIRST PRECEDENT =====")
        print(results[0])
        print("===========================\n")

        return results