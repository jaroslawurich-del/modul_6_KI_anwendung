import requests

from config.settings import OLLAMA_HOST


def check_ollama():

    try:

        r = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=3,
        )

        return r.status_code == 200

    except Exception:
        return False