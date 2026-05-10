PURPOSE
-------
Maps extracted events to canonical causal graph nodes.

RESPONSIBILITIES
----------------
- Semantic similarity search
- Ontology grounding
- Fuzzy matching fallback

INPUT
-----
Raw extracted event

OUTPUT
------
Canonical node ID

TECH STACK
----------
- SentenceTransformers
- BGE-large embeddings
- cosine similarity
- thefuzz

FUTURE IMPROVEMENTS
-------------------
- Ontology expansion
- Dynamic node creation
- Domain-aware grounding