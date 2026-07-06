import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# -----------------------------
# RAG
# -----------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 4))

# -----------------------------
# UI
# -----------------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

# -----------------------------
# Storage
# -----------------------------
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "/data/chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")