PURPOSE
-------
Extracts the primary triggering event from raw news articles.

RESPONSIBILITIES
----------------
- Identify root cause event
- Extract entities
- Detect primary domain
- Return structured JSON

INPUT
-----
Raw article text

OUTPUT
------
{
  raw_event,
  entities[],
  domain
}

TECH STACK
----------
- Groq API
- Llama 3.1
- Pydantic validation

FUTURE IMPROVEMENTS
-------------------
- Multi-event extraction
- Temporal event parsing
- Confidence scoring