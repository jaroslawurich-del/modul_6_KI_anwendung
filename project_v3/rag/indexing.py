import shutil
from pathlib import Path

from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore

from config.settings import VECTOR_DB_DIR
from config.settings import DELETE_DB


def rebuild_index(documents):

    db = Path(VECTOR_DB_DIR)

    #
    # vorhandene DB löschen
    #

    if DELETE_DB and db.exists():

        try:

            shutil.rmtree(db)

        except PermissionError:

            # Windows-Dateisperre
            pass

        except OSError:

            # Docker Volume belegt
            pass

    db.mkdir(
        parents=True,
        exist_ok=True
    )

    #
    # Dokumente splitten
    #

    chunks = split_documents(documents)

    if len(chunks) == 0:
        return

    #
    # Chroma erzeugen
    #

    create_vectorstore(chunks)