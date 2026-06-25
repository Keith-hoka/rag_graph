from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_classic.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from rag.graph_retriever import build_graph_retriever
from rag.vectorstore import chunks_from_vectorstore, load_vectorstore, get_embeddings

FlashrankRerank.model_rebuild()

def build_hybrid_retriever(k: int = 5) -> BaseRetriever:
    """Dense (Chroma) + sparse (BM25), fused with Reciprocal Rank Fusion."""
    vectorstore = load_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    chunks = chunks_from_vectorstore(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

def build_hybrid_graph_retriever(
    k: int = 5,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> BaseRetriever:
    """Vector (dense) + BM25 (sparse) + Graph, three-way RRF fusion."""
    vectorstore = load_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    chunks = chunks_from_vectorstore(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    graph_retriever = build_graph_retriever(k=k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever, graph_retriever],
        weights=list(weights),  
    )

def build_reranking_retriever(
    fetch_k: int = 5,
    top_n: int = 4,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> BaseRetriever:
    base = build_hybrid_graph_retriever(k=fetch_k, weights=weights)

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

def build_retriever(
    use_graph: bool = True,
    use_rerank: bool = True,
    weights: tuple[float, ...] | None = None,
    fetch_k: int = 5,
    top_n: int = 4,
) -> BaseRetriever:
    vectorstore = load_vectorstore()
    chunks = chunks_from_vectorstore(vectorstore)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = fetch_k

    retrievers = [vector_retriever, bm25_retriever]
    if use_graph:
        retrievers.append(build_graph_retriever(k=fetch_k))

    if weights is None:
        weights = tuple(1 / len(retrievers) for _ in retrievers)
    assert len(weights) == len(retrievers), f"weights length {len(weights)} does not match retriever quantity {len(retrievers)}"

    base = EnsembleRetriever(retrievers=retrievers, weights=list(weights))
    if not use_rerank:
        return base

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

def build_inmemory_retriever(chunks, weights = (0.5, 0.5), fetch_k = 5, top_n = 4):
    """Build a '2-way + rerank' search engine from arbitrary chunks, entirely in-memory, 
    not written to the pre-built Chroma/Neo4j database—for real-time indexing of uploaded files, no graph."""
    vector_store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": fetch_k})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = fetch_k

    base = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=list(weights))
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
    