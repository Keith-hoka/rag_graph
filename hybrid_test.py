from dotenv import load_dotenv

load_dotenv()

from retriever import build_hybrid_retriever
from vectorstore import load_vectorstore

def show(title, docs):
    print(f"\n=== {title} ===")
    for i, d in enumerate(docs, 1):
        preview = d.page_content[:120].strip().replace("\n", " ")
        print(f"[{i}] {d.metadata.get('source')} p.{d.metadata.get('page')} | {preview}")


if __name__ == "__main__":
    k =5
    query = "BM25 lexical keyword matching baseline for sparse retrieval"

    vector_only = load_vectorstore().as_retriever(search_kwargs={"k": k})
    show("Vector only (dense)", vector_only.invoke(query))

    hybrid = build_hybrid_retriever(k=k)
    show("Hybrid (dense + BM25, RRF)", hybrid.invoke(query))

    