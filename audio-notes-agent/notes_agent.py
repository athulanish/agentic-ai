from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from openai import OpenAI

client = OpenAI()

# Step 1 transcribe the audio file to plain text
with open ("meeting.m4a", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
    )

from pydantic import BaseModel
from typing import List

# Step 2: define the exact shape we want the data extracted into
class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: str

class Decision(BaseModel):
    topic: str
    outcome: str

class MeetingNotes(BaseModel):
    meeting_date:str
    attendees: List[str]
    action_items: List[ActionItem]
    decisions: List[Decision]
    open_risks: List[str]
    next_meeting: str

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "Extract structured meeting notes from this transcript. Be precise, don't invent information that isn't in the transcript.\n\n"
                "Today's date, for resolving relative dates, is March 3rd. Convert every relative date reference (e.g. 'tomorrow,' 'this Thursday,' 'next Tuesday') into an actual calendar date based on that anchor. Show your date reasoning is correct: March 3rd is a Tuesday, so 'this Thursday' is March 5th, 'tomorrow' is March 4th, 'next Tuesday' is March 10th.\n\n"
                "Before finalizing, re-read the transcript once more specifically looking for any task, commitment, or deadline mentioned by any speaker about their own work, these are easy to miss when they're stated briefly in passing rather than announced as a formal action item."
            )
        },
        {"role": "user", "content": transcript.text},
    ],
    response_format=MeetingNotes,
)

if not completion.choices:
    raise ValueError("No response choices returned for meeting notes.")

notes = completion.choices[0].message.parsed
if notes is None:
    raise ValueError("No structured notes were returned by the model.")

print("---Transcript---")
print(transcript.text)

print("---Structured Notes---")
print(notes.model_dump_json(indent=2))