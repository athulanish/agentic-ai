from urllib import response

from openai import OpenAI
import numpy as np
import json 
from typing import cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

client = OpenAI()

with open("resume.txt", "r", encoding="cp1252") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print(f"Split resume into {len(chunks)} chunks")

def embed(text_list):
    result = client.embeddings.create(model="text-embedding-3-small", input=text_list)
    return [item.embedding for item in result.data]

embed_chunks = embed(chunks)

def cosine_similarity(a,b):
    a,b = np.array(a), np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

# The tools AI is allowed to use

def lookup_resume(query: str) -> str:
    query_embedding = embed([query])[0]
    scores = [cosine_similarity(query_embedding, ce) for ce in embed_chunks]
    top_indices = np.argsort(scores)[::-1][:5]  # Get indices of top 5 relevant chunks
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Tool schemas: this is how the model knows these exist and how to call them
tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_resume",
            "description": "Search the user's resume for relevant experience, skills, or job history",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for in the resume. Returned chunks are not in chronological order, so check any dates present in the text to determine what is most recent."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "A math expression like '4 * 3.5'"}},
                "required": ["expression"]
            }
        }
    }
]

available_functions = {"lookup_resume": lookup_resume, "calculator": calculator}

def run_agent(question):
    messages = cast(list[ChatCompletionMessageParam], [
        {"role": "user", "content": question}
    ])
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0.7
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        messages.append(cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True)))
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue
            function_name = tool_call.function.name
            function = available_functions.get(function_name)
            if function is None:
                result = f"Unknown function: {function_name}"
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = function(**arguments)
                except Exception as exc:
                    result = f"Error calling {function_name}: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })


if __name__ == "__main__":
    question = input("Ask a question about the resume: ")
    print(run_agent(question))