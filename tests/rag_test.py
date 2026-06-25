from dotenv import load_dotenv

load_dotenv()

from rag.rag_chain import build_rag_chain

QUESTIONS = [
    "What encoder and datasets does HyDE use for dense retrieval?",
    "How does HyDE compare to the Contriever baseline?",
    "What are common failure modes discussed across these RAG papers?",
]

if __name__ == "__main__":
    chain = build_rag_chain()
    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {q}\n{'-' * 70}")
        print(chain.invoke(q))
