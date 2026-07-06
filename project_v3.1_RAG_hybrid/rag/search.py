from rag.bm25 import BM25Retriever


def chroma_search(vectorstore, query: str, k: int = 5):
    """
    Dense retrieval über Chroma (Embeddings)
    """
    return vectorstore.similarity_search(query, k=k)


def hybrid_search(
    vectorstore,
    bm25: BM25Retriever,
    query: str,
    k: int = 8
):
    """
    Hybrid Retrieval:
    - Dense (Chroma / embeddings)
    - Sparse (BM25 / keyword)
    - Fusion via dedup + merge
    """

    # 1. Dense retrieval
    dense_docs = chroma_search(vectorstore, query, k)

    # 2. Sparse retrieval
    sparse_docs = bm25.search(query, k)

    # 3. Fusion (RRF-light / dedup merge)
    seen = set()
    merged = []

    for doc in dense_docs + sparse_docs:

        # stabile ID (beste Option: chunk_hash)
        chunk_id = (
            doc.metadata.get("chunk_hash")
            or hash(doc.page_content[:200])
        )

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        merged.append(doc)

    return merged


def rerank_ready_results(docs):
    """
    Optional helper:
    vorbereitet für reranker (nur Struktur, kein Model hier!)
    """
    return docs