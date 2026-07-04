import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "llama3.2"
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-embed-text"
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.2")
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

VECTOR_DB_DIR = os.getenv(
    "VECTOR_DB_DIR",
    "/app/chroma_db"
)

UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    "/app/uploads"
)

MAX_RESULTS = int(
    os.getenv("MAX_RESULTS", "4")
)