import asyncio, math

from dotenv import load_dotenv

load_dotenv()

from rag.adaptive_rag import make_adaptive_answer_fn
from evaluation.eval_harness import (
    METRIC_NAMES, build_metrics, evaluate_pipeline, load_eval_set, nanmean,
)
from rag.retriever import build_retriever


async def main():
    eval_set = load_eval_set()
    metrics = build_metrics()

    targets = [
        ("H: adaptive (router)", make_adaptive_answer_fn()),
        ("A: always 3way+rerank", build_retriever(use_graph=True, use_rerank=True, weights=(0.4, 0.3, 0.3))),
        ("D: always 2way no-rerank", build_retriever(use_graph=False, use_rerank=False, weights=(0.5, 0.5))),
    ]

    results = []
    for label, target in targets:
        print(f"\n{'#' * 78}\n# {label}\n{'#' * 78}")
        results.append(await evaluate_pipeline(target, eval_set, metrics, label))

    print(f"\n\n{'=' * 92}\n== ADAPTIVE comparison ==\n{'=' * 92}")
    print(f"{'config':<28}" + "".join(f"{m[:13]:>16}" for m in METRIC_NAMES))
    print("-" * 92)
    for r in results:
        avgs = [nanmean([row[m] for row in r["rows"]]) for m in METRIC_NAMES]
        print(f"{r['label']:<28}" + "".join(f"{a:>16.3f}" for a in avgs))

    # per-question：adaptive(H) vs always-A，看 simple 那 2 題（第 9、10）H 是否 ≈ A
    print(f"\n=== H - A, per-question (重點看第 9、10 simple 題）===")
    h, a = results[0], results[1]
    print(f"{'#':<3}" + "".join(f"{m[:12]:>14}" for m in METRIC_NAMES))
    for i in range(len(h["rows"])):
        deltas = []
        for m in METRIC_NAMES:
            vh, va = h["rows"][i][m], a["rows"][i][m]
            d = vh - va if not (math.isnan(vh) or math.isnan(va)) else float("nan")
            deltas.append(d)
        print(f"{i + 1:<3}" + "".join(f"{d:>+14.3f}" for d in deltas))


if __name__ == "__main__":
    asyncio.run(main())
