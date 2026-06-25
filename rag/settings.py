from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    open_api_key: str
    langsmith_project: str = "rag-graph"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "please-change-me"

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

settings = Settings()
