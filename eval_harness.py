import sys, types
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.__getattr__ = lambda name: object
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import asyncio, json, math
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from rag_chain import answer_with_contexts, make_chain
from retriever import build_reranking_retriever
from settings import settings

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

def load_eval_set(path="eval_set.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_metrics():
    client = AsyncOpenAI()
    llm = llm_factory(settings.llm_model, client=client, max_tokens=4096)
    emb = embedding_factory("openai", model=settings.embedding_model, client=client)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),        
    }

async def safe_score(metric, retries: int = 5, **kwargs) -> float:
    """Failures are always recorded as NaN, never as 0.0—otherwise, 
    a harness failure will silently pull the aggregate low."""
    for attempt in range(retries + 1):
        try:
            result = await metric.ascore(**kwargs)
            v = float(result.value)
            return v if not math.isnan(v) else float("nan")
        except Exception as e:
            if attempt == retries:
                print(f"  {metric.__class__.__name__} (failed after {retries} times of retries): {type(e).__name__}: {e}")
            else:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
        if attempt < retries:
            await asyncio.sleep(1.0)
    return float("nan")

async def evaluate_pipeline(retriever, eval_set, metrics, label):
    chain = make_chain()
    rows = []
    for i, item in enumerate(eval_set):
        q, ref = item["user_input"], item["reference"]
        has_ref_ctx = bool(item.get("reference_contexts"))
        print(f" [{i + 1}/{len(eval_set)}] {q[:60]}...(ctx_metrics={'on' if has_ref_ctx else 'off'})")
        answer, contexts = answer_with_contexts(q, retriever, chain)

        rows.append({
            "faithfulness": await safe_score(
                metrics["faithfulness"], user_input=q, response=answer, retrieved_contexts=contexts),
            "answer_relevancy": await safe_score(
                metrics["answer_relevancy"], user_input=q, response=answer),
            "context_precision": await safe_score(
                metrics["context_precision"], user_input=q, reference=ref, retrieved_contexts=contexts) 
                if has_ref_ctx else float("nan"),
            "context_recall": await safe_score(
                metrics["context_recall"], user_input=q, reference=ref, retrieved_contexts=contexts)
                if has_ref_ctx else float("nan"),            
        })
    return {"label": label, "rows": rows}

def nanmean(values):
    clean = [v for v in values if not math.isnan(v)]
    return mean(clean) if clean else float("nan")

def report(result):
    rows = result["rows"]
    print(f"\n{'=' * 78}\n== {result['label']} ==\n{'=' * 78}")
    header = f"{'#':<3}" + "".join(f"{m[:15]:>16}" for m in METRIC_NAMES)
    print(header)
    for i, r in enumerate(rows):  
        print(f"{i + 1:<3}" + "".join(f"{r[m]:>16.3f}" for m in METRIC_NAMES))
    print("-" * len(header))
    print(f"{'avg':<3}" + "".join(f"{nanmean([r[m] for r in rows]):>16.3f}" for m in METRIC_NAMES))
    print(f"{'NaN':<3}" + "".join(
        f"{sum(1 for r in rows if math.isnan(r[m])):>16}" for m in METRIC_NAMES))

async def main():
    eval_set = load_eval_set()
    metrics = build_metrics()

    print("Baseline: Complete pipeline (three-way + FlashRank rerank)...")
    retriever = build_reranking_retriever(fetch_k=5, top_n=4)
    report(await evaluate_pipeline(retriever, eval_set, metrics, "Full pipeline (3-way + rerank)"))


if __name__ == "__main__":
    asyncio.run(main())
