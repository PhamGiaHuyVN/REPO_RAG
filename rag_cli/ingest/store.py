"""Vector store (Chroma + Ollama embedding), tạo lười và dùng chung."""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_ollama import OllamaEmbeddings
    from rag_cli.settings import EMBED_MODEL
    return OllamaEmbeddings(model=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma
    from rag_cli.settings import COLLECTION, DB_DIR
    return Chroma(collection_name=COLLECTION,
                  embedding_function=get_embeddings(),
                  persist_directory=str(DB_DIR))
