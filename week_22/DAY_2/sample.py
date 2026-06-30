from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://127.0.0.1:11434"
)

# Testdaten
documents = [
    Document(page_content="Hallo Welt"),
    Document(page_content="Ollama embeddings funktionieren"),
]

# Vektordatenbank
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

print("OK")