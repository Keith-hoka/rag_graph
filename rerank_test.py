from dotenv import load_dotenv

load_dotenv()

from retriever import build_hybrid_graph_retriever, build_reranking_retriever


def show(title, docs):
    print(f"\n===== {title} ({len(docs)} docs) =====")
    for i, d in enumerate(docs, 1):
        tag = d.metadata.get("retriever", "vec/bm25")
        score = d.metadata.get("relevance_score")
        score_str = f" score={score:.3f}" if score is not None else ""
        preview = d.page_content[:90].strip().replace("\n", " ")
        print(f"[{i}] <{tag}>{score_str} | {preview}")


if __name__ == "__main__":
    query = "What encoder and datasets does HyDE use for dense retrieval?"

    show("Before rerank (fused, 11 docs)", build_hybrid_graph_retriever(k=5).invoke(query))
    show("After FlashRank rerank (top 4)", build_reranking_retriever(fetch_k=5, top_n=4).invoke(query))