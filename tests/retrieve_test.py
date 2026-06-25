from dotenv import load_dotenv

load_dotenv()

from rag.vectorstore import load_vectorstore

if __name__ == "__main__":
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    query = "What is HyDE and how does it generate hypothetical documents?"
    docs = retriever.invoke(query)

    print(f"Query: {query}\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}")
        print(d.page_content[:200].strip())
        print("-" * 60)

        