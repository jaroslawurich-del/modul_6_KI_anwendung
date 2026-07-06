from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path


def load_file(file_path: str):

    path = Path(file_path)
    suffix = path.suffix.lower()

    # -------------------------
    # PDF LOADER
    # -------------------------
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

    # -------------------------
    # TEXT LOADER
    # -------------------------
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # -------------------------
    # METADATA FIX
    # -------------------------
    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["path"] = str(path)

    return documents