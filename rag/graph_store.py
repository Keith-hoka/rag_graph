from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from rag.settings import settings

ALLOWED_NODES = ["Method", "Model", "Dataset", "Metric", "Task", "Concept", "Paper"]
ALLOWED_RELATIONSHIPS = ["USES", "EVALUATED_ON", "PROPOSED_IN", "COMPARES_WITH", "IMPROVES", "PART_OF"]
ADDITIONAL_INSTRUCTIONS = (
    "Focus ONLY on technical research entities: methods, models, datasets, "
    "metrics, tasks, and core concepts, and the single paper being described. "
    "Do NOT extract author names, person names, affiliations, institutions, "
    "universities, or conference/journal venues. Do NOT extract entries from "
    "reference or citation lists. Create a Paper node only for the work being "
    "described in the text, never for cited works."
)

def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

def get_graph_transformer() -> LLMGraphTransformer:
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        additional_instructions=ADDITIONAL_INSTRUCTIONS,
    )

def build_graph(chunks: list[Document], graph: Neo4jGraph | None = None) -> int:
    """Extract the chunks into entities and relationships, write them into Neo4j, 
    and return the number of graph documents."""
    graph = graph or get_graph()
    transformer = get_graph_transformer()

    graph_documents = transformer.convert_to_graph_documents(chunks)

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )
    return len(graph_documents)

def reset_graph(graph: Neo4jGraph | None = None) -> None:
    graph = graph or get_graph()
    graph.query("MATCH (n) DETACH DELETE n")
