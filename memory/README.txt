PURPOSE
-------
Stores and retrieves historical precedent events.

RESPONSIBILITIES
----------------
- Historical event ingestion
- Vector embedding storage
- Similar event retrieval
- Macro state comparison

INPUT
-----
Canonical events

OUTPUT
------
Top-k historical analogues

TECH STACK
----------
- Qdrant
- PostgreSQL
- SentenceTransformers

FUTURE IMPROVEMENTS
-------------------
- Temporal retrieval
- Market regime clustering
- Adaptive weighting