from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

TOP_K = int(os.getenv("TOP_K", 4))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))