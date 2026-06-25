from dotenv import load_dotenv

load_dotenv()

from graph_store import build_graph, get_graph, reset_graph
from ingestion import chunk_documents, load_documents

if __name__ == "__main__":
    chunks = chunk_documents(load_documents())
    graph = get_graph()

    print(f"Clear the old pictures and prepare to build the full {len(chunks)} chunks...")
    reset_graph(graph)

    print(f"Use LLM to extract all {len(chunks)} chunks, it will take couple minutes ({len(chunks)} calls...)")
    n = build_graph(chunks, graph)
    print(f"Done, {n} graph documents are written")

    print("\n=== Number of nodes for each label ===")
    for row in graph.query("MATCH (n) RETURN labels(n) AS labels, count(*) AS c ORDER BY c DESC"):
        print(f"  {row['labels']}: {row['c']}")
