# Audio Meeting Notes Agent

Transcribes a recorded meeting and extracts structured, schema-validated notes from it: action items with owners and resolved deadlines, decisions, open risks, and attendees. Built specifically to surface where an audio-to-structured-data pipeline breaks, not just to show a happy-path demo.

## Pipeline

```
audio file (.m4a)
   → gpt-4o-mini-transcribe (speech-to-text)
   → raw transcript (plain text)
   → gpt-4o-mini + Pydantic schema (structured extraction)
   → validated JSON (MeetingNotes)
```

Two separate model calls doing two separate jobs. The first job is "turn sound into text." The second is "turn unstructured text into a specific, typed shape." Keeping them separate (rather than trying to go straight from audio to JSON in one call) made it possible to isolate which stage was actually responsible for each bug below, since errors in a combined pipeline are much harder to attribute to a specific cause.

## Why `gpt-4o-mini-transcribe` over `whisper-1`

OpenAI's newer GPT-4o-based transcription models post meaningfully better word-error-rates than the original Whisper models, and `gpt-4o-mini-transcribe` is the cheapest of the current options while still outperforming `whisper-1` on accuracy, a reasonable default for anything that isn't extremely latency- or cost-sensitive at scale. `gpt-4o-transcribe-diarize` (speaker-labeled transcription) was considered but not used here, the test recording was read by one person doing multiple "voices," so diarization would have had nothing real to separate, worth revisiting if this project is extended with an actual multi-person recording.

## Why Pydantic + `response_format` instead of prompting for JSON

Early GenAI projects (including this one's own Build 1-3) got structured-ish output by asking nicely in the prompt, "please respond in JSON." That's a request, not a guarantee, the model can and does drift from it under real-world phrasing variance. Defining a `pydantic.BaseModel` and passing it as `response_format` to `client.beta.chat.completions.parse()` constrains the output at the API level, the response is validated against the schema before it ever reaches your code, including nested structures (a list of typed `ActionItem` objects inside `MeetingNotes`, not a loose list of strings). This is the difference between hoping for structure and enforcing it, and it's what makes the output actually usable downstream, e.g. looping through `notes.action_items` to programmatically create tasks in a real tool, which is not realistically doable from unstructured paragraph text.

## The test recording

Deliberately built to be harder than a clean, simple meeting: three speakers, overlapping action items phrased three different ways ("by this Thursday," "by tomorrow morning," "first thing tomorrow," "I'll reach out today"), a pricing decision where the final outcome differs from the initially proposed number, specific dollar figures, a genuinely unresolved/vague risk (no fixed date given), and a next-meeting reference given only as "next Tuesday," never a calendar date. The goal was to stress-test extraction against real meeting messiness, not a scripted, clean example.

## Three failure modes found, and what each one teaches

### 1. Transcription errors on spoken numbers
The transcript came back with "$49 a month" correctly in one place but the same Finance proposal elsewhere (on a different read of the same script) came through as "$14 a month," and "$12,000 total" came through as "$12 total." Spoken numbers are a known weak spot for transcription models, "forty-nine" and "fourteen" are phonetically close, and "twelve thousand" can lose its scale word entirely in compressed audio.

**Lesson:** a structured-extraction pipeline is only as trustworthy as what came before it. A perfectly valid, schema-checked JSON object can faithfully encode a transcription error with zero indication anything is wrong, the JSON's validity says nothing about the underlying facts' correctness. For anything high-stakes, numeric values extracted from audio need a human spot-check or a secondary verification pass, not blind trust because the pipeline didn't throw an error.

### 2. A real action item silently dropped (recall failure)
Marcus's own stated deadline for the main deliverable, the API integration itself, was missing from the first extraction pass entirely, while three smaller, more clearly-announced tasks were caught correctly. The missed item was stated casually, in passing, as part of a status update, not flagged as a formal "action item" the way another speaker's task explicitly was.

**Lesson:** this is more dangerous than a formatting error, because nothing about the output looked incomplete. Clean, validated JSON with three correct action items looks exactly as trustworthy as clean, validated JSON with four, there is no structural signal that something is missing. Fixed by explicitly instructing the model to re-read the transcript specifically for casually-stated commitments, not just clearly flagged ones:
```
"Before finalizing, re-read the transcript once more specifically looking for
any task, commitment, or deadline mentioned by any speaker about their own
work, these are easy to miss when they're stated briefly in passing rather
than announced as a formal action item."
```

### 3. Hallucinated year inside a valid-looking date
The transcript only ever said "March 3rd," no year anywhere. Asked to resolve relative dates ("this Thursday," "tomorrow," "next Tuesday") into real calendar dates, the model correctly did the day-of-week arithmetic (confirmed against March 3rd actually being a Tuesday) but silently invented a year, `2021`, that appears nowhere in the source material, with no indication in the output that it was guessed rather than known.

**Lesson:** the most dangerous hallucinations are the ones wearing a trustworthy format. A model inventing a plausible-sounding paragraph is relatively easy to spot on a careful read; a model inventing one field inside an otherwise-correct, schema-validated ISO date string (`"2021-03-05"`) is not, the string's correctness as a *date format* gives no signal about its correctness as a *fact*. Fixed by removing the ambiguity entirely rather than hoping the model would flag its own uncertainty:
```
"Today's date, for resolving relative dates, is Tuesday, March 3rd, 2026.
...If the transcript does not specify a year for any date, use 2026."
```
Also fixed the day-of-week math itself the same way, by giving the model the answer inline (`"March 3rd is a Tuesday, so 'this Thursday' is March 5th"`) rather than trusting it to compute that reliably from scratch, since language models are generally unreliable at silent calendar arithmetic, the same way they're unreliable at silent regular math.

## Code structure

```python
class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: str

class Decision(BaseModel):
    topic: str
    outcome: str

class MeetingNotes(BaseModel):
    meeting_date: str
    attendees: List[str]
    action_items: List[ActionItem]
    decisions: List[Decision]
    open_risks: List[str]
    next_meeting: str
```
Nested Pydantic models, not one flat structure, so each action item carries its own owner/task/deadline as separate typed fields rather than one unstructured string that would need further parsing downstream.

```python
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[...],
    response_format=MeetingNotes
)
notes = completion.choices[0].message.parsed
```
`.parse()` (as opposed to the standard `.create()`) is what actually enforces the schema and returns an already-validated Python object, `notes.action_items[0].owner` is a real attribute access, not a dictionary key guessed from a string.

## Known remaining limitations (not fixed, worth stating honestly)
- No verification layer on numeric values extracted from audio, a wrong dollar figure would currently pass through silently.
- Single-pass extraction; a production version would likely benefit from a second model call cross-checking the first extraction against the transcript specifically for recall gaps, rather than relying on one instruction in the prompt to catch every missed item.
- No diarization, the pipeline trusts speaker labels as spoken in the transcript ("Marcus," "Priya," "Diane") rather than actually detecting distinct voices, which would misattribute an action item if a speaker were introduced inconsistently.

## Setup
```bash
uv add openai pydantic python-dotenv
```
`.env` with `OPENAI_API_KEY=your-key-here`, drop an audio file in as `meeting.m4a` (or update the filename in `notes_agent.py`), then:
```bash
uv run notes_agent.py
```
