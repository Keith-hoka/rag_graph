import sys, types
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.__getattr__ = lambda name: object
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
import httpx

from ingestion import load_full_documents
from settings import settings

TESTSET_SIZE = 12

if __name__ == "__main__":
    # Feed page-level Document (not your 500-token chunks): RAGAS 0.4 will build its own knowledge graph
    # And do summary/headline transform. Feeding chunks that are too broken will trigger ValueError because the content is too short.
    docs = load_full_documents()

    print(f"Loaded {len(docs)} pages, RAGAS will generate {TESTSET_SIZE} questions")

    client = AsyncOpenAI(
        timeout=httpx.Timeout(120.0, connect=10.0, read=100.0, write=20.0),
        max_retries=6,
    )
    gen_llm = llm_factory(settings.llm_model, client=client)
    gen_emb = OpenAIEmbeddings(client=client, model=settings.embedding_model)
    
    # Reduce concurrency + extend retry to avoid the 200K TPM upper limit
    run_config = RunConfig(
        max_workers=2,
        timeout=300,
        max_retries=15,
        max_wait=120,
    )

    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb)
    testset = generator.generate_with_langchain_docs(
        docs, 
        testset_size=TESTSET_SIZE,
        run_config=run_config,
    )

    df = testset.to_pandas()
    df.to_json("testset_raw.jsonl", orient="records", lines=True, force_ascii=False)

    for i, row in df.iterrows():
        print(f"{'=' * 70}\n[{i}] synthesizer: {row['synthesizer_name']}")
        print(f"Q: {row['user_input']}")
        print(f"A(gold): {str(row['reference'])[:180]}")
        print(f"# reference_contexts: {len(row['reference_contexts'])}")
