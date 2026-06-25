import asyncio
import json
import math

from dotenv import load_dotenv

load_dotenv()

from eval_harness import (
    METRIC_NAMES, build_metrics, evaluate_pipeline, load_eval_set, nanmean,
)
from retriever import build_retriever

CONFIGS = [
    {"label": "A: 3way+rerank w.4/.3/.3", "use_graph": True,  "use_rerank": True,  "weights": (0.4, 0.3, 0.3)},
    {"label": "B: 3way no-rerank",        "use_graph": True,  "use_rerank": False, "weights": (0.4, 0.3, 0.3)},
    {"label": "C: 2way+rerank",           "use_graph": False, "use_rerank": True,  "weights": (0.5, 0.5)},
    {"label": "D: 2way no-rerank",        "use_graph": False, "use_rerank": False, "weights": (0.5, 0.5)},
    {"label": "E: 3way+rerank equal",     "use_graph": True,  "use_rerank": True,  "weights": (0.34, 0.33, 0.33)},
    {"label": "F: 3way+rerank graph-hi",  "use_graph": True,  "use_rerank": True,  "weights": (0.3, 0.3, 0.4)},
    {"label": "G: 3way+rerank dense-hi",  "use_graph": True,  "use_rerank": True,  "weights": (0.5, 0.3, 0.2)},
]


async def main():
    eval_set = load_eval_set()
    metrics = build_metrics()
    results = []

    for cfg in CONFIGS:
        print(f"\n{'#' * 78}\n# {cfg['label']}\n{'#' * 78}")
        retriever = build_retriever(
            use_graph=cfg["use_graph"], use_rerank=cfg["use_rerank"],
            weights=cfg["weights"], fetch_k=5, top_n=4,
        )
        result = await evaluate_pipeline(retriever, eval_set, metrics, cfg["label"])
        results.append(result)

    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump([{"label": r["label"], "rows": r["rows"]} for r in results],
                  f, ensure_ascii=False, indent=2)

    # === Summary table: config × Four-indicator average ===
    print(f"\n\n{'=' * 92}\n== ABLATION 摘要 ==\n{'=' * 92}")
    print(f"{'config':<28}" + "".join(f"{m[:13]:>16}" for m in METRIC_NAMES))
    print("-" * 92)
    for r in results:
        avgs = [nanmean([row[m] for row in r["rows"]]) for m in METRIC_NAMES]
        print(f"{r['label']:<28}" + "".join(f"{a:>16.3f}" for a in avgs))

    # === per-question delta（redistribution，not covered by average）===
    def delta(a_prefix, b_prefix, metric):
        ra = next(r for r in results if r["label"].startswith(a_prefix))
        rb = next(r for r in results if r["label"].startswith(b_prefix))
        print(f"\n{metric}: [{a_prefix.strip()}] − [{b_prefix.strip()}]")
        print(f"{'#':<3}{'A':>10}{'B':>10}{'Δ':>10}")
        for i, (xa, xb) in enumerate(zip(ra["rows"], rb["rows"])):
            va, vb = xa[metric], xb[metric]
            d = va - vb if not (math.isnan(va) or math.isnan(vb)) else float("nan")
            print(f"{i + 1:<3}{va:>10.3f}{vb:>10.3f}{d:>+10.3f}")

    print(f"\n{'=' * 60}\n== graph contribution:A - C (rerank with or without graph) ==\n{'=' * 60}")
    delta("A:", "C:", "context_recall")
    delta("A:", "C:", "faithfulness")

    print(f"\n{'=' * 60}\n== rerank contribution:A - B (three way with or without rerank) ==\n{'=' * 60}")
    delta("A:", "B:", "context_recall")
    delta("A:", "B:", "answer_relevancy")


if __name__ == "__main__":
    asyncio.run(main())