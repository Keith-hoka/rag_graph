import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from settings import settings

CHROMA_DIR = Path(".chroma")
COLLECTION = "rag_graph"

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)

def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Re-embed and write to disk (clear the old collection first to avoid repeated insertion)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

def load_vectorstore() -> Chroma:
    """Connect to the collection that has been created on the disk and do not re-embed it."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"{CHROMA_DIR} does not exist, please run index.py first")

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),  # Note: constructor uses embedding_function
        persist_directory=str(CHROMA_DIR),
    )

def chunks_from_vectorstore(vs: Chroma) -> list[Document]:
    """Get the chunk text + metadata back from Chroma and rebuild the Document (for BM25 indexing).
Let Chroma be the single source of truth for the corpus, and BM25 just re-indexes the same chunks in memory. """
    data = vs.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
