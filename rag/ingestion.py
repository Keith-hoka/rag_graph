from pathlib import Path
from random import sample
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load all PDFs under data/, one document per page."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Could not find any PDF under {data_dir}")

    docs: list[Document] = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        docs.extend(pages)
        print(f"  {path.name}: {len(pages)} pages")
    return docs

def chunk_documents(
    docs: list[Document], 
    chunk_size: int = 500, 
    chunk_overlap: int = 60,
    ) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # Remember the starting position of each chunk in the original text, and use it for reference tracing later.
    )
    return splitter.split_documents(docs)

def load_full_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Merge all pages of each PDF into "one complete document" and perform headline segmentation for RAGAS.
Avoid page-level feeds that cause HeadlineSplitter to explode due to lack of headlines on pages that are too short. """
    pages = load_documents(data_dir)
    by_source: dict[str, list[Document]] = defaultdict(list)
    for p in pages:
        by_source[p.metadata.get("source", "unknown")].append(p)

    full_docs = []
    for source, ps in by_source.items():
        ps.sort(key=lambda d: d.metadata.get("page", 0))
        text = "\n\n".join(p.page_content for p in ps)
        full_docs.append(Document(
            page_content=text,
            metadata={"source": source, "filename": source}
        ))
    return full_docs


if __name__ == "__main__":
    print("Loading PDF...")
    documents = load_documents()
    print(f"{len(documents)} pages\n")

    print("chunking...")
    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks are chunked\n")

    sample = chunks[0]
    print("--- The first chunk ---")
    print("metadata: ", sample.metadata)
    print("The first 300 characters: ")
    print(sample.page_content[:300])
