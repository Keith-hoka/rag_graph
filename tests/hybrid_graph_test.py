from dotenv import load_dotenv

load_dotenv()

from rag.graph_retriever import build_graph_retriever
from rag.retriever import build_hybrid_graph_retriever
from rag.vectorstore import chunks_from_vectorstore, load_vectorstore
from langchain_community.retrievers import BM25Retriever

def show(title, docs):
    print(f"\n===== {title} ({len(docs)} docs) =====")
    for i, d in enumerate(docs, 1):
        tag = d.metadata.get("retriever", "vec/bm25")
        src = d.metadata.get("source", d.metadata.get("type", "?"))
        preview = d.page_content[:90].strip().replace("\n", " ")
        print(f"[{i}] <{tag}> {src} | {preview}")


if __name__ == "__main__":
    k = 5
    query = "What encoder and datasets does HyDE use for dense retrieval?"

    vs = load_vectorstore()
    show("Vector only", vs.as_retriever(search_kwargs={"k": k}).invoke(query))

    bm25 = BM25Retriever.from_documents(chunks_from_vectorstore(vs))
    bm25.k = k
    show("BM25 only", bm25.invoke(query))

    show("Graph only", build_graph_retriever(k=k).invoke(query))

    show("Fused (vector + BM25 + graph, RRF)", build_hybrid_graph_retriever(k=k).invoke(query))    