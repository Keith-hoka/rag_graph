from dotenv import load_dotenv

load_dotenv()

from langchain_neo4j import Neo4jGraph

from settings import settings

graph = Neo4jGraph(
    url=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password,
)

print("Neo4j conection OK: ", graph.query("RETURN 1 AS ok"))
print("APOC version: ", graph.query("RETURN apoc.version() AS v"))
