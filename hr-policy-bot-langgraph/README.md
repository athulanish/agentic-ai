# HR Policy Bot in LangGraph (LangChain vs LangGraph)

The HR policy RAG bot from `hr-policy-bot/` (LangChain + Chroma), rebuilt in LangGraph, then extended with a self-checking loop. The goal was a fair, evidence-based comparison of the two frameworks on the same documents, model, and prompt, not a claim that one is better.

## Setup of the comparison

- **Document:** a ~40-page fictional employee handbook (17 sections), the same file in both projects.
- **Retrieval:** identical in all versions. `RecursiveCharacterTextSplitter` (400 chars, 50 overlap), `text-embedding-3-small`, Chroma, `k=3`.
- **Model and prompt:** `gpt-4o-mini` at temperature 0, with the same "answer only from the context, otherwise say you don't know" prompt.
- **Test questions (known answers):**
  1. *Multi-section, answerable:* resignation notice during probation and after probation as a Band 3 employee. Expected: 1 week (Section 3) and 4 weeks (Section 15.1).
  2. *Single-section, numeric:* meal allowance and standard-city hotel cap. Expected: $60/day and $250/night (Sections 7.3, 7.4).
  3. *Unanswerable:* which process applies to someone on a Performance Improvement Plan during probation. The handbook never says, so the correct answer is to refer the employee to HR.

## Three versions

| Version | Q1 | Q2 | Q3 (handbook is silent) |
|---|---|---|---|
| LangChain (LCEL chain) | Correct | Correct | Unsupported claim: asserted the PIP process takes precedence |
| LangGraph, basic port | Correct | Correct | Same unsupported claim |
| LangGraph, corrective loop | Correct | Correct | Refused and referred to HR (correct) |

Swapping frameworks changed nothing about answer quality. The two basic versions share the same retrieval, prompt, and model, and behaved identically, including the same mistake on Q3. The only version that behaved differently is the one with a different design.

## Code comparison

**LangChain (LCEL):**
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
chain.invoke(question)
```
Compact. Data flows through unnamed pipes, and the input shape has to match what the first step expects. A bug I hit: passing `{"question": q}` instead of a plain string made the retriever try to embed a dictionary.

**LangGraph (basic port):**
```python
class State(TypedDict):
    question: str
    context: str
    answer: str

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
```
More lines for the same behavior. Every step reads and writes named fields in one shared state, so the input-shape bug above does not occur, and the flow is declared separately from the logic, which makes it easy to inspect and change.

## The corrective graph

```
START -> retrieve -> generate -> grade --supported--> END
                         ^          |
                         |          +--unsupported, first attempt--> rewrite --+
                         +-------------------------------------------------- (back to retrieve)
                                    |
                                    +--unsupported, retries used--> fallback ("contact HR") -> END
```

- **`grade`:** a second LLM call with a structured output (`supported: bool`, `reason: str`) that checks whether every claim in the answer is directly stated in the retrieved context. It is told to reject conclusions, rankings, or rules of precedence the context does not explicitly state.
- **`rewrite`:** turns the question into a keyword-rich search query covering every topic mentioned, then retrieves again and merges the new chunks with the old ones.
- **`route`:** plain Python reading the state. Supported leads to the end, unsupported on the first attempt leads to a rewrite, unsupported after one retry leads to the fallback. The `attempts` counter in the state is the safety stop against infinite loops.

This control flow (a conditional branch plus a loop) is what a linear LCEL chain does not express naturally, and it is the real reason to reach for LangGraph here.

### Results

- Q1 and Q2: the grader returned `supported=True` on the first pass. No rewrite, no extra retrieval, and the answers were unchanged.
- Q3: the grader rejected the original answer, noting that the context never states that the PIP process takes precedence over probation. The rewrite produced a new query (`Performance Improvement Plan, probation period, process guidelines, employee handbook`). The second answer was rejected for the same reason, and the graph returned: "I couldn't find a clear answer to this in the company handbook. Please contact HR directly."

## A bug and a wrong conclusion I had to correct

In the original `hr-policy-bot`, a question spanning probation and PIPs returned "I don't know." I first blamed semantic search for failing to connect unrelated topics and added a MultiQueryRetriever. That was the wrong diagnosis. `Chroma.from_texts()` appends chunks on every run, so repeated runs filled the persisted store with duplicate copies and the top-k results were the same chunks repeated. After guarding the insert so chunks are only added to an empty collection, plain `k=3` retrieval found both sections.

A second correction came from testing the unanswerable question: the original "I don't know" had actually been the right answer. The handbook never says whether a PIP or probation takes precedence, so what looked like a retrieval failure was the bot being correct, and the later unsupported answer was the real failure. Lesson: check what is in the store, check what was retrieved, and check whether the document actually contains the answer, before changing the retrieval strategy.

## Cost of the extra safety

| Path | LLM calls per question |
|---|---|
| Basic chain or basic graph | 1 |
| Corrective graph, answer supported | 2 (generate, grade) |
| Corrective graph, rejected twice | 5 (generate, grade, rewrite, generate, grade) |

The grader adds latency and cost to every question, and the retry loop added three calls to Q3 without changing the outcome.

## Limitations (stated honestly)

- **Tiny test set.** Three questions, one run each. This shows the mechanism works, not that it is reliable. A real evaluation would use dozens of questions and repeat runs.
- **The grader is an LLM.** It can reject correct answers or approve wrong ones. It behaved correctly on all three questions here, but that is not a guarantee.
- **The retry did not rescue anything.** On Q3 the information genuinely is not in the handbook, so retrying could not help. The loop is designed for retrieval misses, and none of my tests demonstrated it recovering an answer.
- **Faithfulness is checked, completeness is not.** The grader asks whether claims are supported by the context. A bot that answers "I don't know" because retrieval missed the right section would pass this check, since refusing is trivially supported. Catching that would need a second check on whether the answer actually addresses the question.

## Which to use

- **LCEL chain:** a linear pipeline (retrieve, prompt, model, parse) with no branching. Less code, easy to read.
- **LangGraph:** anything with branching, loops, retries, validation steps, or human approval. More code up front, but each step and transition is explicit and inspectable.

## Repo structure
```
hr-policy-bot-langgraph/
├── hr_graph_basic.py       # faithful port of the LangChain bot
├── hr_graph_corrective.py  # grade, rewrite, fallback loop
├── eval_questions.py       # 3 questions with known answers (basic graph)
├── run_corrective.py       # same questions through the corrective graph
├── company_policy.txt
└── .env, .gitignore        # .env and chroma_db/ are ignored
```

## Setup
```bash
uv init --no-workspace
uv add langgraph langchain-openai langchain-chroma langchain-text-splitters python-dotenv
```
Create `.env` with `OPENAI_API_KEY=your-key-here`, then:
```bash
uv run eval_questions.py
uv run run_corrective.py
```
