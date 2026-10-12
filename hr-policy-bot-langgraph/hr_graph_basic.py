from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- same setup as the LangChain version ---
with open("company_policy.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="hr_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
if len(vectorstore.get()["ids"]) == 0:
    vectorstore.add_texts(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant. Answer the question using ONLY the context below, which is pulled from the company's official policy documents.
If the answer isn't in the context, say you don't know and suggest the employee contact HR directly. Do not guess, do not make up numbers.

Context:
{context}

Question: {question}
""")

# --- the state: one shared dictionary every step reads from and writes to ---
class State(TypedDict):
    question: str
    context: str
    answer: str

# --- nodes: plain functions that take the state and return the fields they changed ---
def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}

def generate(state: State) -> dict:
    messages = prompt.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": llm.invoke(messages).content}

# --- the graph: declare the steps, then the order ---
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

if __name__ == "__main__":
    question = "If I'm on a Performance Improvement Plan and I'm also in my probation period, which process applies?"
    result = graph.invoke(State(question=question, context="", answer=""))
    print("--- Retrieved context ---")
    print(result["context"][:800])
    print("\n--- Answer ---")
    print(result["answer"])