from typing import TypedDict

from rag.retrieval import retrieve_context
from rag.vectorstore import get_vectorstore
from rag.bm25 import BM25Retriever
from llm.client import get_chat_model  # <-- falls du sowas hast


class AgentState(TypedDict):
    question: str
    answer: str


def rag_node(state: AgentState):

    question = state["question"]

    # 1. Load RAG components
    vectorstore = get_vectorstore()

    # BM25 braucht Dokumente → aus DB oder Cache laden
    # (hier minimal Beispiel)
    docs = vectorstore.similarity_search(question, k=50)
    bm25 = BM25Retriever(docs)

    # 2. Retrieve context (HYBRID + RERANK)
    context = retrieve_context(vectorstore, bm25, question)

    # 3. Prompt bauen
    prompt = f"""
Nutze ausschließlich den folgenden Kontext:

{context}

Frage:
{question}
"""

    # 4. LLM call (Ollama / Chat Model)
    llm = get_chat_model()

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content
    }