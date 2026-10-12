## Project 1 - HR Policy RAG Bot (`hr-policy-bot/`)

An upgrade of Build 2's RAG pattern using LangChain and Chroma instead of raw cosine similarity. Answers employee questions using only a company policy document (a realistic ~40-page fictional employee handbook, 17 sections, written specifically to include cross-references between sections so retrieval could be properly stress-tested).

**Stack:** LangChain (LCEL), `langchain-openai`, `langchain-chroma`, Chroma as a persisted vector store instead of an in-memory list, so embeddings are computed once and reused across runs instead of recalculated every time the script executes.

**What it does well:** single-topic questions, even ones needing multiple chunks from the same section (e.g. "what's the remote work policy"), get pulled together correctly and answered accurately.

**Bug I hit, and what I got wrong at first:** a question spanning two sections (probation and Performance Improvement Plans) returned "I don't know." I first blamed semantic search for not connecting unrelated topics, and added a MultiQueryRetriever. Later I found the real cause: `Chroma.from_texts()` appends chunks on every run, so repeated runs filled the vector store with duplicate copies, and the top-k results were the same few chunks repeated. After adding a guard so chunks are only inserted into an empty collection, plain `k=3` retrieval found both sections. Lesson: inspect what was retrieved (and what is in the store) before changing the retrieval strategy. 

**Also learned:** LCEL chains are shape-sensitive, `chain.invoke(question)` works because the string gets passed to every branch of the chain's input dictionary at once, but `chain.invoke({"question": question})` breaks it, since the retriever branch then tries to run a similarity search on a dictionary instead of text.