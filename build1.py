from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a successful career coach. give practical advice to help people navigate their career paths."},
        {"role": "user", "content": "I'm feeling stuck in my current job and don't know what to do next."}
    ], temperature=1.3
)

print(response.choices[0].message.content)