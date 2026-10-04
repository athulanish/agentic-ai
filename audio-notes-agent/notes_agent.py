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

print("---Transcript---")
print(transcript.text)

