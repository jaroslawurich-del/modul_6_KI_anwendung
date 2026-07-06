from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import (
    OLLAMA_HOST,
    CHAT_MODEL,
    EMBED_MODEL,
)


# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_HOST,
    )


# -----------------------------
# CHAT LLM
# -----------------------------
def get_chat_model(temperature: float = 0.2):
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
    )