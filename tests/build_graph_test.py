from dotenv import load_dotenv

load_dotenv()

from rag.graph_store import build_graph, get_graph, reset_graph
from rag.ingestion import chunk_documents, load_documents

SAMPLE_SIZE = 30

if __name__ == "__main__":
    all_chunks = chunk_documents(load_documents())
    stride = max(1, len(all_chunks) // 30)
    chunks = all_chunks[::stride]

    graph = get_graph()
    print("Clear old pictures (the last round of data contaminated by bibliography)...")
    reset_graph(graph)


    print(f"Extract entities and relationships from {len(chunks)} chunks")
    n = build_graph(chunks, graph)
    print(f"{n} graph documents are writen.\n")

    print("=== Number of nodes for each label ===")
    for row in graph.query(
        "MATCH (n) RETURN labels(n) AS labels, count(*) AS c ORDER BY c DESC"
        ):
        print(f"  {row['labels']}: {row['c']}")

    print("\n=== Quantity of each relationship ===")
    for row in graph.query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC"
        ):
        print(f"  {row['rel']}: {row['c']}")        

    print("\n=== Sample 15 triples ===")
    for row in graph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE NOT a:Document AND NOT b:Document
        RETURN a.id AS src, type(r) AS rel, b.id AS dst
        LIMIT 15
        """
    ):
        print(f" ({row['src']}) - [{row['rel']}] -> ({row['dst']})")
