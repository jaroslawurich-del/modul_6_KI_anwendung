from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL, OLLAMA_HOST, TEMPERATURE

def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=TEMPERATURE
    )