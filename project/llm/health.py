import requests
from config.settings import OLLAMA_HOST

def check_ollama():
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).status_code == 200
    except:
        return False