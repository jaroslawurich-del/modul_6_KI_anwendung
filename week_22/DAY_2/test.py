from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://127.0.0.1:11434"
)

print(emb.embed_query("test"))