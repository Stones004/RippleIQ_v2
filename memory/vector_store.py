import json
import numpy as np

from pathlib import Path


class VectorStore:

    def __init__(self):

        self.memory_dir = Path("memory")

        self.embedding_path = (
            self.memory_dir / "event_embeddings.npy"
        )

        self.metadata_path = (
            self.memory_dir / "event_metadata.json"
        )

    def save_embeddings(
        self,
        embeddings,
        metadata
    ):

        np.save(
            self.embedding_path,
            embeddings
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2
            )

        print(
            "[VECTOR STORE] Historical embeddings saved."
        )

    def load_embeddings(self):

        embeddings = np.load(
            self.embedding_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        return embeddings, metadata