# Voice BI Agent

Senior leaders speak a business question; an agent queries the company's Excel data and answers out loud.

**Stack:** Python, pandas, OpenAI (speech-to-text, tool-calling LLM, text-to-speech), Excel as the data source.

## How it works
mic -> `gpt-4o-mini-transcribe` -> tool-calling agent -> pandas tools -> answer -> `gpt-4o-mini-tts` -> speaker

Design choice: the LLM never writes code or does maths. It picks from 4 typed tools
(`describe_data`, `query_metrics`, `compare_periods`, `vs_target`) and pandas computes the numbers.
That removes arbitrary code execution and makes every figure traceable to a tool call (printed in the console).

## Run it
    pip install -r requirements.txt
    export OPENAI_API_KEY=...
    python generate_data.py     # builds data/business_data.xlsx (~8.9k orders, 21 months, 6 regions)
    python test_tools.py        # no API needed: checks tool output against raw pandas
    python voice_agent.py       # or --text / --no-speak

## The dataset
Sheets: Orders, Products, Targets, Data_Dictionary, Summary (live SUMIFS check).
Four business stories are planted in the data so you can test whether the agent finds the cause, not just the number:
1. Dubai online discounts jumped to about 24% in Q2 2026, squeezing margin.
2. Riyadh Smart Home return rate spiked in Jul-Aug 2026.
3. Kochi delivery times climbed from mid-Aug 2026.
4. Bengaluru beat its revenue target all through 2026.

## Demo questions
- "How did we do last quarter versus the one before?"
- "Why did margin drop in Dubai?"
- "Which category has a returns problem and where?"
- "Are any regions missing target this year?"
- "Is delivery getting slower anywhere?"

## Known limitations (fill in as you test)
- Speech-to-text can still mangle spoken numbers and dates; the console prints what it heard.
- Synthetic data: findings are illustrative.
