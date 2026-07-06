import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama


# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_embeddings():
    return OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )


# -----------------------------
# CHAT LLM
# -----------------------------
def get_chat_model():
    return ChatOllama(
        model=os.getenv("CHAT_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        temperature=float(os.getenv("TEMPERATURE", 0.2))
    )