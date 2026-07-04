from llm.client import get_llm


class ModelFactory:

    @staticmethod
    def chat(
        temperature: float = 0.2
    ):
        return get_llm(temperature)