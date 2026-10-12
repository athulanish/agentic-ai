# Agentic AI Starter Projects

## Build 1-3 - GenAI Fundamentals (`fundamentals/`)

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
## Project 1 - HR Policy RAG Bot (`hr-policy-bot/`)

An upgrade of Build 2's RAG pattern using LangChain and Chroma instead of raw cosine similarity. Answers employee questions using only a company policy document (a realistic ~40-page fictional employee handbook, 17 sections, written specifically to include cross-references between sections so retrieval could be properly stress-tested).

**Stack:** LangChain (LCEL), `langchain-openai`, `langchain-chroma`, Chroma as a persisted vector store instead of an in-memory list, so embeddings are computed once and reused across runs instead of recalculated every time the script executes.

**What it does well:** single-topic questions, even ones needing multiple chunks from the same section (e.g. "what's the remote work policy"), get pulled together correctly and answered accurately.

**Bug I hit, and what I got wrong at first:** a question spanning two sections (probation and Performance Improvement Plans) returned "I don't know." I first blamed semantic search for not connecting unrelated topics, and added a MultiQueryRetriever. Later I found the real cause: `Chroma.from_texts()` appends chunks on every run, so repeated runs filled the vector store with duplicate copies, and the top-k results were the same few chunks repeated. After adding a guard so chunks are only inserted into an empty collection, plain `k=3` retrieval found both sections. Lesson: inspect what was retrieved (and what is in the store) before changing the retrieval strategy.

**Also learned:** LCEL chains are shape-sensitive, `chain.invoke(question)` works because the string gets passed to every branch of the chain's input dictionary at once, but `chain.invoke({"question": question})` breaks it, since the retriever branch then tries to run a similarity search on a dictionary instead of text.

## Project 2 - Guardrailed Support Bot (`nemo-guardrailed-bot/`)

A customer support bot for a fictional company, built two ways to show a real before-and-after: `basic_bot.py` relies on a system prompt alone, `guarded_bot.py` adds NVIDIA NeMo Guardrails as an independent enforcement layer on top.

**Adversarial testing on the unguarded version:** ran classic jailbreaks ("ignore previous instructions," DAN-style roleplay), fiction-framing attacks, indirect prompt injection (malicious instructions hidden inside simulated knowledge-base content), scope-drift probes (neutral off-topic questions like "capital of France"), and payload splitting across multiple messages. gpt-4o-mini's system prompt alone resisted all of them, a genuinely stronger result than expected, and a useful finding in itself: frontier models in 2026 are considerably harder to jailbreak with well-known attack patterns than commonly assumed.

**Why build guardrails anyway, given that:** production systems don't rely on "the model probably won't misbehave" as their only defense, since behavior can shift across model versions, temperatures, or untested edge cases. NeMo Guardrails adds an independent input/output check, a second model call judging "does this violate policy" as its only job, separate from the main conversational model.

**Bug I hit building it:** forgot to port the system prompt's factual grounding into the guardrails config, which only had input/output policy checks, no instructions about what the company actually does. Guardrails correctly blocked off-topic and manipulative messages, but the bot hallucinated vague, generic company info for legitimate questions, proof that safety enforcement and factual grounding are two separate jobs, passing every guardrail check doesn't stop a model from confidently making things up if it was never told the facts. Fixed by adding a `general` instructions block to `config.yml`.

**Interesting side effect after the fix:** the same off-topic question (the France one) started getting handled by the main model's own redirect instead of the guardrails' canned block message, two independent layers catching the same problem through different mechanisms depending on phrasing.

## Project 3 - Audio Meeting Notes Agent (`audio-notes-agent/`)

Transcribes a recorded meeting and extracts structured, schema-validated notes (action items, decisions, risks) using Pydantic. Deliberately stress-tested with a messy, multi-speaker recording and surfaced three distinct failure modes: transcription errors on spoken dollar amounts, a silently dropped action item (a recall gap invisible in valid-looking JSON), and a hallucinated year inside an otherwise correct date string.

[Full write-up, including the pipeline, model choice, and all three bugs](./audio-notes-agent/README.md)

## Project 4 - Data Analysis Crew (`data-analysis-crew/`)

A three-agent CrewAI pipeline (SQL analyst, insights analyst, executive communicator) run against a synthetic sales database with a planted anomaly. The first run found the right region and the right numbers but drew the wrong conclusion, because comparing quarterly totals hid that the dip had already recovered. Fixing the task granularity corrected the narrative.

[Full write-up, including framework comparison and the dependency conflict](./data-analysis-crew/README.md)

## Project 5 - HR Policy Bot in LangGraph (`hr-policy-bot-langgraph/`)

The Project 1 HR bot rebuilt in LangGraph for a like-for-like comparison with LangChain, using the same handbook, model, and prompt. Swapping frameworks changed nothing about answer quality, and both made the same unsupported claim on a question the handbook doesn't answer. Adding a grading node with a rewrite-and-retry loop, which a linear chain can't express, made the bot refuse that question and refer the employee to HR.

[Full write-up, including the comparison table, cost, and limitations](./hr-policy-bot-langgraph/README.md)