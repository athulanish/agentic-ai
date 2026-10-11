# Data Analysis Crew (CrewAI)

A three-agent CrewAI pipeline that investigates a sales database the way a BI team would: one agent writes and runs SQL, one interprets the results, one writes an executive summary for a non-technical reader. Built against a synthetic dataset with a deliberately planted anomaly, so every claim the crew makes can be checked against a known answer.

## Pipeline

```
sales_data.csv (2,172 rows) -> sales.db (SQLite)
        |
   SQL Analyst  --(runs queries via custom tool)-->  raw results
        |  context
   Data Insights Analyst  -->  finding, timeline, likely cause
        |  context
   Business Communications Specialist  -->  3-4 sentence executive summary
```

Each agent has one narrow job. A single agent asked to query, interpret, and communicate in one prompt tends to do all three at a mediocre level. Splitting them makes each prompt shorter and each failure easier to locate.

## The dataset and the planted anomaly

Synthetic daily sales, January to June 2026: 4 regions x 3 products, with weekday/weekend seasonality and random noise. One pattern is buried on purpose: APAC sales of DataGuard collapse from April 10 to May 20 (a simulated stockout) and recover afterwards. Every other region/product combination moves by 3.5% or less between Q1 and Q2.

Because the ground truth is known, the crew's output can be graded, not just read and trusted.

## Why CrewAI, and what the alternatives are

| Framework | Strength | Tradeoff |
|---|---|---|
| **CrewAI** (used here) | Role-based agents and sequential handoffs, a working multi-agent prototype in hours | Less explicit control over state and branching |
| **LangGraph** | Stateful, auditable graph workflows, widely used for production agents | More upfront design work |
| **AutoGen** | Dynamic multi-agent conversation | Microsoft has moved active development to its Agent Framework, so it is not a good new investment |

This task is a linear handoff (query, interpret, summarise) with no branching, loops, or human approval step, which is exactly CrewAI's sweet spot. A workflow that needs checkpointing or human-in-the-loop pauses would be a better fit for LangGraph.

## Code walkthrough

**The tool.** `@tool("Run SQL Query")` turns a plain Python function into something an agent can choose to call. The docstring doubles as the tool description, so it includes the table schema and the valid region and product names. The agent cannot see the database, so anything not in the docstring is unknown to it. Errors are returned as strings (`"SQL error: ..."`) instead of raised, which lets the agent read the error and retry with a corrected query.

**The agents.** Each has a `role`, `goal`, and `backstory`. Only the SQL analyst gets a tool. The interpreter and communicator work purely from the text handed to them.

**The handoff.** `context=[query_task]` on the second task is the mechanism that makes this a pipeline. CrewAI injects the earlier task's output into the later agent's input. Without it, the three agents would run with no knowledge of each other.

**The process.** `Process.sequential` runs tasks in listed order. `Process.hierarchical` (a manager agent deciding order dynamically) exists but trades predictability for autonomy, and was not needed here.

## Results: run 1 vs run 2

### Run 1: right numbers, wrong story
The crew correctly isolated APAC DataGuard and reported a drop of nearly 40% and over $53,000 in lost revenue. Checked against the database: the true figures are a 39.6% unit drop and $53,251 in lost revenue. It also correctly noted the drop was unique to that region and product.

But it recommended acting urgently "before the situation worsens." The monthly data shows APAC DataGuard did about 560 units a month in Q1, dropped to 203 in April and 242 in May, and was back to 575 by June. The problem had already resolved.

**Root cause:** the task told the SQL agent to compare Q2 against Q1, so it compared two quarterly totals. A quarterly total flattens the time shape of an event. The agents did exactly what the prompt asked, and the prompt asked for the wrong granularity.

### Run 2: after the fix
Two prompt changes: the SQL task now requires a month-by-month breakdown, and the interpreter task must state when the issue started and ended and whether it is ongoing, based on the latest month. The new summary correctly reported a drop of about 60% across April and May (verified: 64% and 57%), a full recovery by June, and recommended investigating the cause to prevent recurrence rather than treating it as live.

## Failures and lessons

**1. Uv workspace dependency conflict.** Running `uv init` in each project folder registered them all as members of one workspace with a shared lockfile. The audio project required `openai>=3.23` and `pydantic>=2.13`, while CrewAI pinned older ranges of both, so the resolver reported the workspace unsatisfiable. Fixed by excluding this project from the workspace (`exclude = ["data-analysis-crew"]` in the root `pyproject.toml`), pinning Python 3.12 for it, and giving it its own environment and lockfile. Lesson: a monorepo of independent apps should not share one dependency set when frameworks pin conflicting versions.

**2. Correct numbers hid a wrong conclusion.** The most useful failure. Spot-checking the headline figures would have passed run 1. Only checking the narrative against the monthly shape exposed the problem. Aggregation level is an analytical decision, and an agent will not question it unless told to.

**3. Honest uncertainty about cause.** The dataset contains no inventory or logistics data, so the crew cannot actually know why sales dropped. Its "supply chain or market disruption" is a hypothesis, and the correct response was to recommend investigation rather than assert a cause.

## Known limitations
- Verified by hand against a synthetic dataset, not by an automated evaluation. A production version would assert expected findings in a test suite.
- Single run per configuration. LLM output varies run to run, so one pass is not proof of reliability.
- The SQL tool executes any query the agent writes against a local SQLite file. Acceptable here, but a real deployment would need a read-only connection and query restrictions.

## Setup
```bash
uv python pin 3.12
uv add crewai pandas python-dotenv
```
Create `.env` with `OPENAI_API_KEY=your-key-here`, then:
```bash
uv run setup_db.py   # loads sales_data.csv into sales.db
uv run crew.py
```
