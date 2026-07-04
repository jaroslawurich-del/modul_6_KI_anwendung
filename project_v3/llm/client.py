from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from config.settings import (
    OLLAMA_HOST,
    CHAT_MODEL,
    EMBED_MODEL,
    TEMPERATURE,
)


def get_llm(
    temperature: float | None = None,
):
    """
    Erstellt das Chatmodell.
    """

    if temperature is None:
        temperature = TEMPERATURE

    return ChatOllama(
        base_url=OLLAMA_HOST,
        model=CHAT_MODEL,
        temperature=temperature,
    )


def get_embeddings():
    """
    Embedding-Modell für Chroma.
    """

    return OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=EMBED_MODEL,
    )