from dotenv import load_dotenv

load_dotenv()

from graph_retriever import build_graph_retriever

if __name__ == "__main__":
    retriever = build_graph_retriever(k=4)
    query = "What encoder and datasets does HyDE use for dense retrieval?"
    docs = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Returned {len(docs)} document\n")
    for i, d in enumerate(docs, 1):
        tag = d.metadata.get("type", d.metadata.get("source", "?"))
        print(f"[{i}] ({tag})")
        print(d.page_content[:280].strip())
        print("-" * 60)
