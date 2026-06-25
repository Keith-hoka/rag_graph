import re

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from rag.graph_store import get_graph
from rag.settings import settings

FULLTEXT_INDEX = "entity_id_index"

def create_entity_fulltext_index() -> None:
    """Create a full-text index for the IDs of all __Entity__ entities, supporting fuzzy matching."""
    get_graph().query(
        f"""
        CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS
        FOR (e:`__Entity__`) ON EACH [e.id]
        """        
    )

class Entities(BaseModel):
    names: list[str] = Field(
        description="specific NAMED entities (proper names of methods, models, "
        "datasets, or systems) explicitly mentioned in the question"
    )

def extract_entities(question: str) ->list[str]:
    """Use LLM to extract entity names from the question 
    (avoid searching the index for the entire sentence containing stopword)."""
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    structured = llm.with_structured_output(Entities)
    prompt = (
        "Extract only SPECIFIC, NAMED entities from the question — proper names "
        "of methods, models, datasets, or systems (e.g. 'HyDE', 'Contriever', "
        "'MS-MARCO'). Do NOT extract generic category words that describe the "
        "TYPE of answer wanted, such as 'encoder', 'dataset', 'model', 'method', "
        "'metric', or 'retrieval' — those are what we discover by traversing the "
        "graph, not entities to match.\n"
        "Example: 'What encoder and datasets does HyDE use?' -> ['HyDE']\n\n"
        f"Question: {question}"        
    )
    return structured.invoke(prompt).names

def _fulltext_query(text: str) -> str:
    """Remove Lucene special characters, add blur ~2 to each token and then OR to overcome spelling differences."""
    cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", text)
    tokens = [t for t in cleaned.split() if t]
    return " OR ".join(f"{t}~2" for t in tokens) if tokens else text

def link_entities(names: list[str], limit_per_name: int = 3) -> list[dict]:
    """Each entity name fuzzy finds the closest node in the graph. 
    Only retain scores greater than or equal to the highest score * score_ratio."""
    graph = get_graph()
    results = []
    for name in names:
        rows = graph.query(
            """
            CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
            RETURN node.id AS id, labels(node) AS labels, score
            ORDER BY score DESC LIMIT $limit
            """,
            params={"index": FULLTEXT_INDEX, "q": _fulltext_query(name), "limit": limit_per_name},            
        )
        results.append({"mention": name, "matches": rows})
    return results

def get_entity_neighborhood(entity_ids: list[str], max_rels: int = 30) -> list[str]:
    """Grab neighbors along a relationship and return a human-readable string of relationship facts."""
    rows = get_graph().query(
        """
        MATCH (e:`__Entity__`)-[r]-(n:`__Entity__`)
        WHERE e.id IN $ids
        RETURN e.id AS src, type(r) AS rel, n.id AS dst
        LIMIT $max_rels
        """,
        params={"ids": entity_ids, "max_rels": max_rels},        
    )
    return [f"{r['src']} {r['rel']} {r["dst"]}"for r in rows]

def get_source_chunks(entity_ids: list[str], max_chunks: int = 4) -> list[Document]:
    """Retrieve the source chunks that mention these entities in the reverse direction of MENTIONS, 
    sorted by the number of mentions."""
    rows = get_graph().query(
        """
        MATCH (d:Document)-[:MENTIONS]->(e:`__Entity__`)
        WHERE e.id IN $ids
        WITH d, count(DISTINCT e) AS hits
        RETURN d.text AS text, d.source AS source, d.page AS page
        ORDER BY hits DESC LIMIT $max_chunks
        """,
        params={"ids": entity_ids, "max_chunks": max_chunks},        
    )
    return [Document(
        page_content=r["text"],
        metadata={"source": r["source"], "page": r["page"], "retriever": "graph"},
    ) for r in rows]

class GraphRetriever(BaseRetriever):
    """Anchor entity → neighbor relationship fact + source chunk, packaged into Document."""
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        names = extract_entities(query)
        if not names:
            return []  # No named entity → graph This path does not contribute; it is handled by hybrid.

        linked = link_entities(names)
        entity_ids = [m["id"] for item in linked for m in item["matches"]]
        if not entity_ids:
            return []

        docs = get_source_chunks(entity_ids, max_chunks=self.k)
        facts = get_entity_neighborhood(entity_ids)
        if facts:
            docs.append(Document(
                page_content="Knowledge graph facts:\n" + "\n".join(facts),
                metadata={"retriever": "graph", "type": "graph_facts"},
            ))
        return docs

def build_graph_retriever(k: int =4) -> GraphRetriever:
    create_entity_fulltext_index()
    return GraphRetriever(k=k)

