from llm.client import get_llm
from llm.embeddings import get_embeddings

class ModelFactory:

    @staticmethod
    def llm():
        return get_llm()

    @staticmethod
    def embeddings():
        return get_embeddings()