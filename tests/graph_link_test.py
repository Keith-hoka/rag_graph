from dotenv import load_dotenv

load_dotenv()

from rag.graph_retriever import create_entity_fulltext_index, extract_entities, link_entities

QUESTIONS = [
    "What encoder and datasets does HyDE use for dense retrieval?",
    "How does HyDE compare to the Contriever baseline?",
]

if __name__ == "__main__":
    print("Create entity full-text index...")
    create_entity_fulltext_index()

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQuestion: {q}")
        entities = extract_entities(q)
        print(f"Extract entities: {entities}")
        for item in link_entities(entities):
            print(f"\n[{item['mention']}]")
            for m in item["matches"]:
                print(f"  {m['score']:.2f}  {m['id']}  {m['labels']}")

