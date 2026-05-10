from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    article: str


class AnalyzeStats(BaseModel):
    total_nodes: int
    max_depth: int
    avg_confidence: float


class AnalyzeResponse(BaseModel):
    root_event: str
    canonical_node: str
    domain: str
    entities: list[str]
    causal_chain: list[dict]
    edges: list[dict]
    stats: AnalyzeStats