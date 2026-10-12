from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load the policy document
with open("company_policy.txt", "r") as f:
    raw_text = f.read()

# Step 2: Split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_text(raw_text)
print(f"Number of chunks: {len(chunks)}")

# Step 3: Create embeddings for each chunk and store in chroma vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory="./chroma_db")
retrieve = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 4: the prompt, deliberatively restrictive, this is an HR bot, it should refuse to not go off script
prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant. Answer the question using ONLY the context below, which is pulled from the company's official policy documents.
If the answer isn't in the context, say you don't know and suggest the employee contact HR directly. Do not guess, do not make up numbers.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retrieve | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "I'm in my probation period and want to resign. How much notice do I need to give, and would that change if I did it after probation as a Band 3 employee?",
    "How much can I claim for meals on a business trip, and what is the hotel cap in a standard city?",
    "If I'm on a Performance Improvement Plan and also in my probation period, which process applies?",
]
for q in questions:
    print("Q:", q)
    print("GOT:", chain.invoke(q))
    print("-" * 60)