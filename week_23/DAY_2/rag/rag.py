from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("./docker-de.pdf")
documents = loader.load()

print("Pages:", len(documents))

# -----------------------------
# Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Chunks:", len(chunks))

# -----------------------------
# Embeddings
# -----------------------------
embeddings = OllamaEmbeddings(
    model="all-minilm:33m"
)

# -----------------------------
# Chroma Vector Store
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# -----------------------------
# Retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.

Answer using ONLY the provided context.

If the answer is not in the document, say:
"I don't know based on the provided document."

Limit your answer to three sentences.

Context:
{context}
"""
        ),
        ("human", "{input}")
    ]
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatOllama(
    model="mistral",
    temperature=0
)

# -----------------------------
# Build RAG Chain
# -----------------------------
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# -----------------------------
# Chat Loop
# -----------------------------
print("\nRAG System Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("Question: ")

    if question.lower() in ["exit", "quit", "q"]:
        break

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    print("\nAnswer:")
    print(response["answer"])
    print()