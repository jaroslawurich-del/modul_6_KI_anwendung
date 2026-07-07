from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ---------------------------------------------------
# 1. PDF laden
# ---------------------------------------------------

PDF_PATH = "./docker-de.pdf"  # <-- Pfad zu deiner PDF anpassen

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"PDF erfolgreich geladen.")
print(f"Seiten: {len(documents)}")

print("\nErste Seite:")
print(documents[0].page_content[:500])

# ---------------------------------------------------
# 2. Dokumente aufteilen
# ---------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"\nChunks erstellt: {len(chunks)}")

# ---------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------

embeddings = OllamaEmbeddings(
    model="all-minilm:33m"
)

# ---------------------------------------------------
# 4. Chroma Vector Store
# ---------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Chroma-Datenbank erstellt.")

# ---------------------------------------------------
# 5. Retriever
# ---------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

# ---------------------------------------------------
# 6. Prompt
# ---------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Du bist ein hilfreicher Assistent.

Beantworte Fragen ausschließlich anhand des bereitgestellten Kontextes.

Falls die Antwort nicht im Dokument enthalten ist, antworte:

"Ich weiß es anhand des Dokuments nicht."

Die Antwort soll maximal drei Sätze lang sein.

Kontext:
{context}
"""
        ),
        (
            "human",
            "{input}"
        )
    ]
)

# ---------------------------------------------------
# 7. LLM
# ---------------------------------------------------

llm = ChatOllama(
    model="mistral",
    temperature=0
)

# ---------------------------------------------------
# 8. RAG Chain
# ---------------------------------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("\n========================================")
print("RAG-System gestartet")
print("Zum Beenden: exit, quit oder q")
print("========================================")

# ---------------------------------------------------
# 9. Chat
# ---------------------------------------------------

while True:

    question = input("\nFrage: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Programm beendet.")
        break

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    print("\nAntwort:")
    print(response["answer"])