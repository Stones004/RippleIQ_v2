# Run it like this 
# python -m memory.ingest_events


import json

from sentence_transformers import (
    SentenceTransformer
)

#from vector_store import VectorStore
from memory.vector_store import VectorStore


MODEL = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)


with open(
    "memory/historical_events.json",
    "r",
    encoding="utf-8"
) as f:

    events = json.load(f)


embedding_texts = [

    event["embedding_text"]

    for event in events
]


embeddings = MODEL.encode(
    embedding_texts,
    normalize_embeddings=True
)


store = VectorStore()

store.save_embeddings(
    embeddings,
    events
)

print(
    "[INGESTION] Historical event ingestion complete."
)