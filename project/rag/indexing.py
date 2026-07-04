from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore

def rebuild_index():

    docs = load_documents()

    if not docs:
        print("⚠️ keine PDFs")
        return

    chunks = split_documents(docs)

    if not chunks:
        print("⚠️ keine chunks")
        return

    create_vectorstore(chunks)

    print(f"✔ index erstellt: {len(chunks)}")