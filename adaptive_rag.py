from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_chain import format_docs, make_chain
from retriever import build_retriever
from settings import settings


class RAGState(TypedDict):
    question: str
    route: str
    route_reasoning: str
    documents: list[Document]
    answer: str

class RouteDecision(BaseModel):
    route: Literal["simple", "complex"] = Field(
        description="'simple' for single-fact/definitional/keyword-lookup questions "
        "answerable from one passage; 'complex' for questions needing cross-passage "
        "comparison, relationships, or multi-hop reasoning"
    )
    reasoning: str = Field(description="one short sentence")

ROUTER_PROMPT = """Route a question to a retrieval strategy for a corpus of papers on \
retrieval-augmented generation and dense retrieval.

- 'simple': asks for one fact, definition, or specific value found in a single passage \
(e.g. "What encoder does HyDE use?"). Single-passage lookup suffices.
- 'complex': needs connecting information across passages — comparisons ("how does X \
compare to Y"), relationships, trade-offs, or multi-hop synthesis.

Question: {question}"""

def build_adaptive_graph():
    # Build once and reuse the compiled graph across invokes
    router = ChatOpenAI(model=settings.llm_model, temperature=0).with_structured_output(RouteDecision)
    simple_retriever = build_retriever(use_graph=False, use_rerank=True, weights=(0.5, 0.5))
    complex_retriever = build_retriever(use_graph=True, use_rerank=True, weights=(0.4, 0.3, 0.3))
    chain = make_chain()

    def route_question(state: RAGState) -> dict:
        d = router.invoke(ROUTER_PROMPT.format(question=state["question"]))
        print(f" -> route = {d.route} :: {d.reasoning}")
        return {"route": d.route, "route_reasoning": d.reasoning}

    def retrieve_simple(state: RAGState) -> dict:
        return {"documents": simple_retriever.invoke(state["question"])}

    def retrieve_complex(state: RAGState) -> dict:
        return {"documents": complex_retriever.invoke(state["question"])}

    def generate(state: RAGState) -> dict:
        context = format_docs(state["documents"])
        return {"answer": chain.invoke({"context": context, "question": state["question"]})}

    def pick_route(state: RAGState) -> str:
        return state["route"]

    g = StateGraph(RAGState)
    g.add_node("route_question", route_question)
    g.add_node("retrieve_simple", retrieve_simple)
    g.add_node("retrieve_complex", retrieve_complex)
    g.add_node("generate", generate)

    g.add_edge(START, "route_question")
    g.add_conditional_edges(
        "route_question", 
        pick_route,
        path_map={"simple": "retrieve_simple", "complex": "retrieve_complex"},
    )
    g.add_edge("retrieve_simple", "generate")
    g.add_edge("retrieve_complex", "generate")
    g.add_edge("generate", END)
    return g.compile()

def make_adaptive_answer_fn():
    app = build_adaptive_graph()

    def answer_fn(question: str):
        final = app.invoke({"question": question})
        return final["answer"], [d.page_content for d in final["documents"]]

    return answer_fn