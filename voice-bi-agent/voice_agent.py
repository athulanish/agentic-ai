"""Voice-driven BI agent: speak a question, get a spoken answer grounded in the Excel data.

    python voice_agent.py            # voice in, voice out
    python voice_agent.py --text     # type instead of speak (no mic needed)
    python voice_agent.py --no-speak # voice in, text out
"""
from dotenv import load_dotenv
load_dotenv()
import argparse, io, json
from openai import OpenAI
import bi_tools as bt

client = OpenAI()  # reads OPENAI_API_KEY
CHAT_MODEL = "gpt-4o-mini"
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
MAX_STEPS = 6

info = bt.describe_data()
SYSTEM = f"""You are a business intelligence assistant for senior leaders. Your answers are SPOKEN aloud.

Data: {info['rows']:,} retail orders from {info['date_range'][0]} to {info['date_range'][1]}, in USD.
Treat {info['date_range'][1]} as "today". Resolve phrases like "last quarter", "this month", "year to date"
against that date, and say the exact dates you used.

Rules:
- Get every number from the tools. Never estimate, never do your own arithmetic on large figures.
- If the question is vague, make the most sensible assumption, state it in one short clause, and answer.
- When a metric moved, check at least one breakdown (region, channel or category) to say WHY, not just what.
- Format for the ear: 2 to 4 short sentences, round numbers ("1.4 million", "about 26 percent"),
  no bullet points, no markdown, no tables. Lead with the answer, then the driver, then one suggested action.
- If the tools cannot answer it, say so plainly instead of guessing.
Valid filter values: {json.dumps(info['filter_values'])}"""

history = [{"role": "system", "content": SYSTEM}]


def ask(question: str, verbose: bool = True) -> str:
    history.append({"role": "user", "content": question})
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=history,
                                              tools=bt.SCHEMAS, temperature=0)
        msg = resp.choices[0].message
        history.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            if verbose:
                print(f"   [tool] {call.function.name}({json.dumps(args)})")
            try:
                result = bt.REGISTRY[call.function.name](**args)
            except Exception as e:  # let the model see the error and self-correct
                result = {"error": str(e)}
            history.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    return "I could not finish that analysis within my step limit. Try narrowing the question."


# ---------- voice I/O ----------
def listen(seconds_hint=True) -> str:
    import sounddevice as sd, soundfile as sf, numpy as np
    sr, chunks = 16000, []
    input("\nPress Enter, then speak your question...")
    stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            callback=lambda indata, *_: chunks.append(indata.copy()))
    with stream:
        input("Recording... press Enter to stop.")
    audio = np.concatenate(chunks)
    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV"); buf.seek(0); buf.name = "q.wav"
    # Priming the transcriber with domain vocabulary reduces mis-heard region and product names
    vocab = ("Dubai, Abu Dhabi, Riyadh, Jeddah, Kochi, Bengaluru, revenue, margin, CSAT, "
             "Smart Home, Laptops, Phones, Audio, Accessories, B2B, Q1 Q2 Q3 Q4 2026")
    return client.audio.transcriptions.create(model=STT_MODEL, file=buf, prompt=vocab).text.strip()


def speak(text: str):
    import sounddevice as sd, soundfile as sf
    audio = client.audio.speech.create(model=TTS_MODEL, voice="alloy", input=text, response_format="wav",
                                       instructions="Calm, confident executive briefing tone.")
    data, sr = sf.read(io.BytesIO(audio.content))
    sd.play(data, sr); sd.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true", help="type questions instead of speaking")
    p.add_argument("--no-speak", action="store_true", help="print answers, do not read them aloud")
    a = p.parse_args()
    print("Ask about revenue, margin, returns, delivery, targets... (say or type 'quit' to exit)")
    while True:
        q = input("\nYou: ").strip() if a.text else listen()
        if not a.text:
            print(f"You (heard): {q}")
        if q.lower().strip(" .!") in {"quit", "exit", "stop"}:
            break
        if not q:
            continue
        answer = ask(q)
        print(f"\nAgent: {answer}")
        if not (a.text or a.no_speak):
            speak(answer)


if __name__ == "__main__":
    main()
