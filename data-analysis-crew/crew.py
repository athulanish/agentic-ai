from dotenv import load_dotenv
load_dotenv()

import sqlite3
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# --- the tool the SQL agent is allowed to use ---
@tool("Run SQL Query")
def run_sql_query(query: str) -> str:
    """Executes a SQL query against the 'sales' table and returns the results.
    The sales table has columns: date (YYYY-MM-DD), region, product, units_sold, revenue.
    Regions are: North America, Europe, APAC, Latin America.
    Products are: CloudSync Pro, DataGuard, FlowBoard."""
    conn = sqlite3.connect("sales.db")
    try:
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result = ", ".join(columns) + "\n"
        result += "\n".join(", ".join(str(v) for v in row) for row in rows)
        return result if rows else "Query returned no rows."
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()

# --- agents ---
sql_analyst = Agent(
    role="Senior SQL Analyst",
    goal="Write and run precise SQL queries to answer business questions about sales performance",
    backstory="You're a meticulous data analyst who always checks totals and comparisons across time periods, regions, and products before drawing conclusions. You never guess, you query.",
    tools=[run_sql_query],
    verbose=True
)

data_interpreter = Agent(
    role="Data Insights Analyst",
    goal="Interpret raw query results and identify meaningful patterns, anomalies, and likely causes",
    backstory="You're skilled at spotting what's actually significant in a dataset versus what's just noise, and you always ask 'why' before reporting a finding.",
    verbose=True
)

communicator = Agent(
    role="Business Communications Specialist",
    goal="Translate data findings into a clear, concise summary for a non-technical executive",
    backstory="You write the way a sharp consultant briefs a busy VP, no jargon, straight to what matters and what to do about it.",
    verbose=True
)

# --- tasks, each one feeding into the next via `context` ---
query_task = Task(
    description=(
        "Investigate sales performance across all regions and products for January through June 2026. "
        "Break results down MONTH BY MONTH, not just by quarter, and compare each region/product combination's "
        "monthly trend to identify any significant drops or anomalies. Run whatever SQL queries you need, "
        "don't stop at one query."
    ),
    expected_output="Monthly SQL results by region and product, with any anomaly clearly visible in the numbers.",
    agent=sql_analyst
)

interpret_task = Task(
    description=(
        "Using the SQL results from the previous task, identify the most significant finding. "
        "Explain what happened and when it started and ended. Critically, state whether the issue is still "
        "ongoing or has already recovered, based on the most recent month's numbers."
    ),
    expected_output="A clear explanation of the finding, its timeline, and whether it is ongoing or resolved.",
    agent=data_interpreter,
    context=[query_task]
)

summary_task = Task(
    description=(
        "Write a brief executive summary of this finding for the VP of Sales, who has no interest in SQL or technical detail. "
        "State what happened, the business impact, and one clear recommended next step."
    ),
    expected_output="A 3-4 sentence executive summary, plain English, no jargon.",
    agent=communicator,
    context=[interpret_task]
)

crew = Crew(
    agents=[sql_analyst, data_interpreter, communicator],
    tasks=[query_task, interpret_task, summary_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print("\n\n=== FINAL OUTPUT ===")
print(result)