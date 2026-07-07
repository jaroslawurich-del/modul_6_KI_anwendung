from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------
# 1. PDF laden
# ---------------------------------------------------

PDF_PATH = "./docker-de.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"PDF geladen: {len(documents)} Seiten")

print("\nAuszug:")
print(documents[0].page_content[:500])


# ---------------------------------------------------
# 2. Text in Chunks teilen
# ---------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"\nChunks erstellt: {len(chunks)}")


# ---------------------------------------------------
# 3. Embeddings mit Ollama
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

print("Chroma Vector Store erstellt")


# ---------------------------------------------------
# 5. Retriever Top-6
# ---------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 6
    }
)


# ---------------------------------------------------
# 6. LLM
# ---------------------------------------------------

llm = ChatOllama(
    model="mistral",
    temperature=0
)


# ---------------------------------------------------
# 7. Prompt
# ---------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
Du bist ein hilfreicher Assistent.

Beantworte die Frage ausschließlich mit Informationen
aus dem bereitgestellten Kontext.

Wenn die Antwort nicht im Kontext steht,
sage:
"Ich weiß es anhand des Dokuments nicht."

Antwort maximal drei Sätze.

Kontext:
{context}

Frage:
{question}
"""
)


# ---------------------------------------------------
# 8. RAG Pipeline
# ---------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ---------------------------------------------------
# 9. Chat Loop
# ---------------------------------------------------

print("\n============================")
print("RAG System gestartet")
print("exit / quit / q beendet")
print("============================")


while True:

    question = input("\nFrage: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Beendet.")
        break

    answer = rag_chain.invoke(question)

    print("\nAntwort:")
    print(answer.content)