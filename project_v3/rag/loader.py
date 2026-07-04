from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_documents(upload_dir: str):

    documents = []

    upload_path = Path(upload_dir)

    if not upload_path.exists():
        upload_path.mkdir(parents=True, exist_ok=True)
        return documents

    for file in upload_path.glob("*"):

        if not file.is_file():
            continue

        suffix = file.suffix.lower()

        try:

            # -------------------------
            # PDF SUPPORT
            # -------------------------
            if suffix == ".pdf":

                reader = PdfReader(str(file))
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

            # -------------------------
            # TEXT FILES
            # -------------------------
            else:
                text = file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "type": suffix
                        }
                    )
                )

        except Exception as ex:
            print(f"Fehler beim Laden {file}: {ex}")

    return documents