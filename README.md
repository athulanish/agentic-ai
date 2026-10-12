# Agentic AI Portfolio

Hands-on GenAI and agentic AI projects built by Athul Anish, a data/business analyst moving into AI engineering. Each project is built against a known answer, checked against it, and documented with the failures I hit, not just the version that worked.

## Projects

| # | Project | Stack | What I found |
|---|---|---|---|
| 1 | [HR Policy RAG Bot](./hr-policy-bot/) | LangChain, Chroma, OpenAI | Answers questions from a 17-section fictional employee handbook (81 chunks). A "no answer" bug I first blamed on semantic search was actually duplicate chunks: `Chroma.from_texts()` appended the whole document on every run. |
| 2 | [Guardrailed Support Bot](./nemo-guardrailed-bot/) | NVIDIA NeMo Guardrails, OpenAI | Tried 8 attack types on the unguarded bot (jailbreaks, roleplay, indirect prompt injection, payload splitting) and it resisted all of them. Added guardrails as defense in depth, then found that guardrails enforce policy but do not supply facts: with no grounding, the bot invented a company description. |
| 3 | [Audio Meeting Notes Agent](./audio-notes-agent/) | gpt-4o-mini-transcribe, Pydantic structured output | Turns a recorded meeting into validated JSON. Three silent failures: spoken dollar amounts misheard, a real action item dropped, and a year hallucinated inside a valid-looking date. |
| 4 | [Data Analysis Crew](./data-analysis-crew/) | CrewAI, SQLite, custom SQL tool | Three agents (SQL, interpretation, executive summary) over 2,172 rows of sales data with a planted anomaly. First run found the right region and correct numbers (39.6% drop, $53,251 lost revenue) but told the wrong story, because comparing quarterly totals hid that sales had already recovered. |
| 5 | [HR Bot in LangGraph](./hr-policy-bot-langgraph/) | LangGraph, Chroma, OpenAI | The Project 1 bot rebuilt for a like-for-like comparison. Same answers as LangChain on all three test questions, including the same unsupported claim. A grading node with a rewrite-and-retry loop made it refuse the question the handbook doesn't answer. |

Earlier practice scripts (a raw API call, RAG with hand-written cosine similarity, a tool-calling agent) are in [`fundamentals/`](./fundamentals/).

## Patterns that repeated across projects

- **Failures that look trustworthy.** A dropped action item, a wrong year in a valid date, and a confident wrong narrative all produced clean, well-formatted output. Format validity says nothing about correctness.
- **Check the input before blaming the model.** Several bugs were upstream: duplicate chunks in a vector store, misheard numbers in a transcript, a prompt that asked for the wrong aggregation level.
- **Known answers make evaluation possible.** Each project uses data or documents where I know the right answer, so outputs can be graded instead of just read.
- **A framework swap changed nothing; a design change did.** LangChain and LangGraph gave the same answers. Adding a verification step is what changed behavior.

## Tools used

OpenAI API (chat, embeddings, transcription, structured output), LangChain, LangGraph, CrewAI, NVIDIA NeMo Guardrails, Chroma, SQLite, Pydantic, uv.

## Running a project

Each project is self-contained with its own dependencies and a README with setup steps. Create a `.env` file in the project folder with `OPENAI_API_KEY=your-key-here`, then follow that project's README. API keys are never committed.
