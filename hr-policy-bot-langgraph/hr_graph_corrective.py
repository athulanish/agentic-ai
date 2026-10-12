from typing import TypedDict, List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# reuse the retriever, llm and answer prompt from the basic version
from hr_graph_basic import retriever, llm, prompt


class State(TypedDict):
    question: str
    search_query: str
    docs: List[str]
    answer: str
    supported: bool
    reason: str
    attempts: int


class Grade(BaseModel):
    supported: bool
    reason: str


grader = llm.with_structured_output(Grade)

grade_prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Below are a CONTEXT from a company handbook, a QUESTION, and an ANSWER.
Decide whether every claim in the ANSWER is directly stated in the CONTEXT.
Mark it unsupported if the answer draws a conclusion, ranking, or rule of precedence that the CONTEXT does not explicitly state, even if it sounds reasonable.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite this employee question as a short, keyword-rich search query for finding the relevant handbook sections.
Cover every distinct topic the question mentions. Return only the query.

Question: {question}
""")


def retrieve(state: State) -> dict:
    query = state.get("search_query") or state["question"]
    found = [d.page_content for d in retriever.invoke(query)]
    merged = list(dict.fromkeys(state.get("docs", []) + found))  # dedupe, keep order
    return {"docs": merged}


def generate(state: State) -> dict:
    context = "\n\n".join(state["docs"])
    messages = prompt.invoke({"context": context, "question": state["question"]})
    return {"answer": llm.invoke(messages).content}


def grade(state: State) -> dict:
    context = "\n\n".join(state["docs"])
    result = grader.invoke(grade_prompt.invoke(
        {"context": context, "question": state["question"], "answer": state["answer"]}
    ))
    if isinstance(result, dict):
        result = Grade.model_validate(result)
    print(f"[grade] supported={result.supported} | {result.reason}")
    return {"supported": result.supported, "reason": result.reason}


def rewrite(state: State) -> dict:
    new_query = llm.invoke(rewrite_prompt.invoke({"question": state["question"]})).content
    print(f"[rewrite] {new_query}")
    return {"search_query": new_query, "attempts": state["attempts"] + 1}


def fallback(state: State) -> dict:
    return {"answer": "I couldn't find a clear answer to this in the company handbook. Please contact HR directly."}


def route(state: State) -> str:
    if state["supported"]:
        return "done"
    if state["attempts"] < 1:
        return "retry"
    return "fallback"


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("fallback", fallback)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "grade")
builder.add_conditional_edges("grade", route, {"done": END, "retry": "rewrite", "fallback": "fallback"})
builder.add_edge("rewrite", "retrieve")   # the loop
builder.add_edge("fallback", END)
corrective_graph = builder.compile()


def ask(question: str) -> str:
    initial_state: State = {
        "question": question,
        "search_query": "",
        "docs": [],
        "answer": "",
        "supported": False,
        "reason": "",
        "attempts": 0,
    }
    result = corrective_graph.invoke(initial_state)
    return result["answer"]