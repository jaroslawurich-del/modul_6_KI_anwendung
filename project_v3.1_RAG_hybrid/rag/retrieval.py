from rag.search import hybrid_search
from rag.reranker import rerank


def retrieve_context(vectorstore, bm25, query):

    docs = hybrid_search(vectorstore, bm25, query, k=10)
    docs = rerank(query, docs, top_k=4)

    return "\n\n".join([d.page_content for d in docs])