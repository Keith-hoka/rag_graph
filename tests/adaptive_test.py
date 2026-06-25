import json

from dotenv import load_dotenv

load_dotenv()

from rag.adaptive_rag import build_adaptive_graph

if __name__ == "__main__":
    app = build_adaptive_graph()
    eval_set = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")]

    print("=== Routing decisions (Compare synthesizer type)===\n")
    counts = {"simple": 0, "complex": 0}
    for i, item in enumerate(eval_set):
        q, syn = item["user_input"], item.get("synthesizer", "?")
        # synthesizer is just proxy：single_hop should be simple, multi_hop should be complex
        expect = "simple" if "single_hop" in syn else "complex"
        final = app.invoke({"question": q})
        route = final["route"]
        counts[route] += 1
        flag = "✓" if route == expect else "✗ Inconsistent"
        print(f"[{i+1}] syn={syn[:22]:<22} route={route:<8}(expect {expect}) {flag}")
        print(f"     Q: {q[:62]}\n")

    print(f"=== Distribution: {counts} ===")    