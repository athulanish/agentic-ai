from openai import OpenAI
import numpy as np

client = OpenAI()

#Step 1: Load resume and split it into chunks 
with open("resume.txt", "r", encoding="cp1252") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print(f"Split resume into {len(chunks)} chunks")

# Step 2: Create embeddings for each chunk
def embed(text_list):
    result = client.embeddings.create(model="text-embedding-3-small", input=text_list)
    return [item.embedding for item in result.data]

chunk_embeddings = embed(chunks)

# Step 3: Create a function to find the most relevant chunk based on user query
def cosine_similarity(a,b):
    a,b = np.array(a), np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))    
question = " What is my most recent job title? How many years experiance do I have with Excel?"
question_embedding = embed([question])[0]

scores = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]
top_indices = np.argsort(scores)[::-1][:3]  # Get indices of top 3 relevant chunks
# print("--- Retrieved chunks ---")
# for i in top_indices:
#     print(f"Score: {scores[i]:.3f} | {chunks[i][:100]}")
# print("------------------------")
context = "\n\n".join([chunks[i] for i in top_indices])

#Step 4: give the model only the relevant chunks, and make it stick to them
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Only use the context to answer the question. If you dont know, dont guess, say you dont know"},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ],temperature=0.7
)
print(response.choices[0].message.content)
