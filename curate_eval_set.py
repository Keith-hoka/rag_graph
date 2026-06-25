import json

KEEP = [0, 4, 5, 6, 7, 8, 9, 11]

rows = [json.loads(l) for l in open("testset_raw.jsonl", encoding="utf-8")]

with open("eval_set.jsonl", "w", encoding="utf-8") as f:
    for i in KEEP:
        r = rows[i]
        f.write(json.dumps({
            "user_input": r["user_input"],
            "reference": r["reference"],
            "reference_contexts": r["reference_contexts"],
            "synthesizer": r["synthesizer_name"],            
        }, ensure_ascii=False) + "\n")

print(f"Done writing eval_set.jsonl, {len(KEEP)} questions in total.")
