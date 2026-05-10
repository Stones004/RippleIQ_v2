import json
import os
from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client

    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return _client


SYSTEM_PROMPT = """You are RippleIQ's precision event extraction engine.

Analyze the news content and extract:
1. raw_event — The single ROOT CAUSE triggering event.
2. entities — Named entities directly involved.
3. domain — The PRIMARY sector, exactly one of: agriculture, energy, economics, finance, geopolitics, health, technology, labor

Return ONLY valid JSON."""


def extract_event(article_text: str) -> dict:
    """Extract the primary triggering event from a raw news article."""
    response = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {}

    return {
        "raw_event": str(result.get("raw_event", article_text[:80])),
        "entities": list(result.get("entities", [])),
        "domain": str(result.get("domain", "economics")),
    }
