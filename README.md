# Agentic AI Starter Projects

Three small projects built end to end while learning the practical side of GenAI and agentic AI. No frameworks, raw OpenAI API calls, so every piece is visible instead of hidden behind a library.

## Build 1 - Basic LLM call (`build1.py`)
A plain API call with a system prompt and user prompt. Used this to actually feel the difference between what the system prompt controls (tone/behavior) versus what temperature controls (predictability of word choice), by running the same question at temperature 0 vs 1.3 with different system prompts.

## Build 2 - RAG over my resume (`build2.py`)
Chunks my resume, embeds each chunk, and retrieves the most relevant chunks via cosine similarity before answering a question, instead of relying on the model's own (non-existent) knowledge of my resume.

**Bug I hit and fixed:** asked for my *most recent* job title, and retrieval returned an older job instead. Turns out semantic search ranks chunks by how similar their wording is to the question, it has no concept of chronological order. A resume written top to bottom newest-first means nothing to an embedding, it only measures meaning-similarity. Fixed by widening the number of retrieved chunks and explicitly telling the model in the system prompt that chunks aren't chronologically ordered and to use any dates present to judge recency.

## Build 3 - Agent with tool calling (`build3.py`)
An agent that chooses between two tools, a resume lookup (reusing Build 2's RAG) and a calculator, deciding on its own which to call and in what order based on the question. Asking "how many years of Power BI experience do I have, in months?" triggers a resume lookup followed automatically by a calculator call, chained with no hardcoded sequence telling it to do that.

Same recency bug from Build 2 resurfaced here since the tool reuses the same retrieval logic, confirming that an agent's reasoning is only as reliable as the tools it calls. Same fix applied.

## Setup
```bash
uv add openai numpy
```
Set your own `OPENAI_API_KEY` as an environment variable (never hardcoded in the files), then:
```bash
uv run build1.py
uv run build2.py
uv run build3.py
```