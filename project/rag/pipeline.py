from rag.retriever import get_retriever

def search(query):

    retriever = get_retriever()
    docs = retriever.invoke(query)

    context = []
    sources = []

    for d in docs:
        context.append(d.page_content)
        sources.append(f"{d.metadata.get('source')} | {d.metadata.get('page')}")

    return {
        "context": "\n\n".join(context),
        "sources": sources
    }