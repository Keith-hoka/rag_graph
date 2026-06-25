from dotenv import load_dotenv

load_dotenv()

from ingestion import chunk_documents, load_documents
from vectorstore import build_vectorstore


if __name__ == "__main__":
    chunks = chunk_documents(load_documents())
    print(f" {len(chunks)} are embed to Chroma...")

    vs = build_vectorstore(chunks)
    print(f"Done, {vs._collection.count()} vectors in total.")

