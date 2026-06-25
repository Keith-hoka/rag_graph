import json

from dotenv import load_dotenv

load_dotenv()

from rag.rag_chain import answer_with_contexts, make_chain
from rag.retriever import build_reranking_retriever

IDX = 8 

if __name__ == "__main__":
    item = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")][IDX]
    retriever = build_reranking_retriever(fetch_k=5, top_n=4)
    chain = make_chain()

    answer, contexts = answer_with_contexts(item["user_input"], retriever, chain)

    print(f"Q: {item['user_input']}\n")
    print(f"=== your pipeline answer ===\n{answer}\n")
    print(f"=== gold reference ===\n{item['reference']}\n")
    print(f"=== found {len(contexts)} contexts (the first 100 charactors）===")
    for i, c in enumerate(contexts, 1):
        print(f"[{i}] {c[:100].strip()}")