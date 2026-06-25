from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from retriever import build_reranking_retriever
from settings import settings

SYSTEM_PROMPT = """You are a research assistant answering questions about a corpus of \
papers on retrieval-augmented generation and dense retrieval.

Answer the question using ONLY the provided context. You may and should SYNTHESIZE \
information across multiple context blocks to form a complete answer — relevant facts \
are often spread across several blocks. Connect them.

Only say the context is insufficient if the answer genuinely cannot be derived from any \
combination of the provided blocks. Do NOT refuse just because no single block states \
the answer verbatim. If the context partially answers the question, give the partial \
answer and state specifically what is missing — do not dismiss the whole question.

Cite sources inline as [source: filename p.X]. Be precise and concise; do not use \
outside knowledge or guess."""

PROMPT = ChatPromptTemplate([
    ("system",  SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def format_docs(docs: list[Document]) -> str:
    """Format retrieved docs into context blocks with source tags."""
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "knowledge-graph")
        page = d.metadata.get("page")
        tag = f"{src} p.{page + 1}" if page is not None else src
        blocks.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def build_rag_chain(fetch_k: int =5, top_n: int = 4):
    retriever = build_reranking_retriever(fetch_k=fetch_k, top_n=top_n)
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )

def make_chain():
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    return PROMPT | llm | StrOutputParser()

def answer_with_contexts(query: str, retriever_or_fn, chain):
    if callable(retriever_or_fn) and not hasattr(retriever_or_fn, "invoke"):
        return retriever_or_fn(query)
    docs = retriever_or_fn.invoke(query)
    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": query})
    return answer, [d.page_content for d in docs]
