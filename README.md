# RippleIQ v2 - causal intelligence engine

Takes a news article, extracts the events in it, grounds them to nodes of a causal graph, and propagates
their impact through the graph, adjusted by similar historical events.

```
article -> extraction (LLM, Groq) -> grounding (normaliser + semantic grounder)
        -> propagation over the causal graph (depth <= 6, hop decay 0.75, min score 0.05)
        -> precedent modifier from historical events (vector search)
```

## Layout
- `app/` - FastAPI service (`app/main.py`, `app/services/analysis_service.py`); `frontend/static/index.html`
  is the web page.
- `pipeline/extraction`, `pipeline/grounding`, `pipeline/propagation` - the three stages.
- `graph/graph_engine.py` + `data/graphs/causal_rules.json` - builds the causal graph.
- `memory/` - historical events, their embeddings and precedent search.
- `docs/` - the architecture and implementation blueprint (PDF).

## Run
```bash
pip install -r requirements.txt
# put GROQ_API_KEY in a local .env file (not committed)
uvicorn app.main:app --reload
```

## Project status (as of 2026-11-26)

**Last thing built - Phase 2 (2026-05-21, commit `c24700e`):** precedent memory (`memory/`: historical-event
ingestion, embeddings, similarity search), the semantic grounder, and precedent-weighted propagation.

**Not built yet (placeholder folders with a `README.txt`):** `pipeline/scenarios`, `pipeline/uncertainity`,
`retrieval`, and `pipeline/orchestration` (only an `__init__`).

**Recent dev changes:** no code changes since Phase 2. This push adds the architecture blueprint PDF, a
README, and ignore rules for virtualenvs, `.env` and generated files.
