# debug_retrieval.py
from dotenv import load_dotenv

load_dotenv()

from vectorstore import chunks_from_vectorstore, load_vectorstore

# 每道候選 simple 題的「答案關鍵字串」——這些字串必須真的出現在某個 chunk 裡，
# 題目才算「答案在語料、可被檢索」，否則就是無解題（像第 9 題的全稱）
CANDIDATES = {
    "Q_encoder": "Contriever",                          # HyDE 用什麼 encoder
    "Q_generative_model": "InstructGPT",                # HyDE 用什麼生成模型
    "Q_trec_dl": "DL19",                                # 評測用哪個 TREC track
    "Q_lost_middle": "U-shaped",                        # Lost-in-the-Middle 的曲線形狀
    "Q_msmarco": "MS-MARCO",                            # 主要評測 collection
}

if __name__ == "__main__":
    vs = load_vectorstore()
    all_chunks = chunks_from_vectorstore(vs)
    print(f"語料共 {len(all_chunks)} 個 chunk\n")

    for tag, phrase in CANDIDATES.items():
        hits = [(i, c) for i, c in enumerate(all_chunks)
                if phrase.lower() in c.page_content.lower()]
        status = f"✓ 存在於 {len(hits)} 個 chunk" if hits else "✗ 不在語料（無解題，棄用）"
        print(f"[{tag}] '{phrase}' → {status}")
        if hits:
            i, c = hits[0]
            pos = c.page_content.lower().find(phrase.lower())
            snippet = c.page_content[max(0, pos - 35):pos + 65].replace("\n", " ")
            print(f"     首見 {c.metadata.get('source')} p.{c.metadata.get('page')}: …{snippet}…")
        print()