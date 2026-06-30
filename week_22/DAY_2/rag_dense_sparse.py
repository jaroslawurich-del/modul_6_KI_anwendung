# -*- coding: utf-8 -*-
"""
Hybrid RAG Chain — Dense (Chroma) + Sparse (BM25) with Manual RRF Fusion
=========================================================================
Architecture:
  - Dense retriever  : Chroma vector store + Ollama nomic-embed-text embeddings
                       (semantic similarity search over embedded document chunks)
  - Sparse retriever : BM25 (term-frequency keyword search, no embeddings needed)
  - Fusion           : Reciprocal Rank Fusion (RRF) manually implemented —
                       combines ranked results from both retrievers using
                       weighted inverse-rank scores, then re-ranks and top-k selects
  - LLM              : Llama 3.1 via Ollama (local inference, no API key required)

Requires:
    pip install rank_bm25 langchain langchain-community langchain-chroma langchain-ollama
"""


#%% 1. IMPORTS
print("1. Importing libraries...")

# PDF loading: reads pages from a PDF file and yields LangChain Document objects
from langchain_community.document_loaders import PyPDFLoader

# Vector store: stores dense embeddings in an in-memory (or persistent) Chroma DB
from langchain_chroma import Chroma

# Text splitting: breaks large documents into overlapping chunks for better retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prompt template: structures system + human messages for the chat-style LLM input
from langchain_core.prompts import ChatPromptTemplate

# Ollama integrations: local LLM runner (OllamaLLM) and embedding model (OllamaEmbeddings)
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Output parser: strips the raw LLM message object down to a plain string
from langchain_core.output_parsers import StrOutputParser

# Chain primitives:
#   RunnablePassthrough — passes the input unchanged to the next step
#   RunnableLambda      — wraps any Python callable as a chain step
from langchain_core.runnables import RunnableLambda

# BM25 sparse retriever: keyword-based ranking (no embeddings, fast, good for exact terms)
from langchain_community.retrievers import BM25Retriever

import langchain
print(langchain.__version__)   # sanity-check installed version


#%% 2. LOAD PDF FILE
print("2. Loading PDF file...")

# PyPDFLoader reads the PDF page-by-page and returns a list of Document objects.
# Each Document has: .page_content (str) and .metadata (dict with source, page num).
#file_path = "C:\\Users\\milos\\Desktop\\DATA\\Educx\\Modul_6_KI\\KI_Lernplan\\KI_W2_T7\\RAG_chain\\nke-10k-2023.pdf"
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()   # docs is a list[Document], one per PDF page

print(len(docs))                        # total number of pages loaded
print(docs[1].page_content[0:1000])     # first 1000 chars of page 2 (0-indexed)
print(docs[1].metadata)                 # e.g. {'source': '...pdf', 'page': 1}


#%% 3. SPLIT DOCUMENTS INTO CHUNKS
print("3. Splitting documents into chunks...")

# RecursiveCharacterTextSplitter splits text by paragraph, then sentence, then word
# to keep semantically related content together.
#
#   chunk_size=800    — max characters per chunk (a trade-off: larger = more context,
#                       smaller = more precise retrieval)
#   chunk_overlap=200 — characters shared between adjacent chunks so context
#                       at chunk boundaries is not lost
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(docs)   # splits is a list[Document]


#%% 4. DENSE RETRIEVER — Chroma + Ollama Embeddings
print("4. Creating dense retriever...")


# OllamaEmbeddings calls the local Ollama server to embed each chunk into a
# fixed-size vector. nomic-embed-text is a small, fast embedding model.
local_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    #base_url="http://127.0.0.1:11434"
    )

print(local_embeddings.base_url)

print(local_embeddings.embed_query("Hallo Welt"))
print(local_embeddings)

# Chroma.from_documents: embeds all splits and stores vectors in an in-memory DB.
# On subsequent runs you can persist it to disk with persist_directory="./chroma_db".
vectorstore = Chroma.from_documents(documents=splits, embedding=local_embeddings)

# as_retriever exposes the vector store as a LangChain Retriever interface.
# search_type="similarity" → cosine / inner-product nearest-neighbour search.
# k=6 → return the 6 most similar chunks per query.
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# k = [doc1, doc2, ..., doc6]

#%% 5. SPARSE RETRIEVER — BM25 (keyword-based)
print("5. Creating sparse retriever...")
# BM25Retriever builds an inverted index over the text of every split chunk.
# It scores documents by term frequency (TF) and inverse document frequency (IDF),
# so it excels at matching exact keywords that embedding models might miss.
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 6   # return the top 6 BM25-ranked chunks per query

# k = [doc1, doc2, ..., doc6]

#%% 6. HYBRID RETRIEVER — Manual Reciprocal Rank Fusion (RRF)
print("6. Creating hybrid retriever with RRF fusion...")

# RRF merges two ranked lists by rewarding documents that appear high in BOTH lists.
# Score formula per document per list:  weight * 1 / (rank + c)
#   rank  — 1-based position in the ranked list
#   c=60  — smoothing constant (standard RRF default) that dampens the importance
#            of rank-1 vs rank-2, preventing a single top result from dominating
#   weight_bm25 — share of the BM25 signal (0.5 = equal contribution from both)
#
# Final score for a document = sum of its RRF scores across both lists.
# Documents appearing in only one list still get a partial score.


def hybrid_retrieve(query: str, k: int = 6, weight_bm25: float = 0.5) -> list:
    # Step 7a: get top-k results from each retriever independently
    bm25_docs  = bm25_retriever.invoke(query)
    dense_docs = dense_retriever.invoke(query)

    c = 60        # RRF smoothing constant (see explanation above)
    scores   = {} # maps chunk text → accumulated RRF score
    all_docs = {} # maps chunk text → Document object (for final lookup)

    # Step 7b: accumulate BM25 RRF scores (weighted by weight_bm25)
    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key]   = scores.get(key, 0) + weight_bm25 * (1 / (rank + c))
        all_docs[key] = doc

    # Step 7c: accumulate dense RRF scores (weighted by 1 - weight_bm25)
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content
        scores[key]   = scores.get(key, 0) + (1 - weight_bm25) * (1 / (rank + c))
        all_docs[key] = doc

    # Step 7d: sort all seen documents by descending fused score, keep top-k
    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:k]
    return [all_docs[k] for k in sorted_keys]


#%% 7. PROMPT TEMPLATE
print("7. Creating prompt template...")

# The system message instructs the LLM to act as a Q&A assistant and to
# ground its answer strictly in the retrieved {context} chunks.
# {context} and {input} are placeholder variables filled at chain run-time.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# ChatPromptTemplate.from_messages builds a prompt with a 'system' role message
# (instructions + context) and a 'human' role message (the user's question).
# LangChain injects {context} and {input} values before calling the LLM.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # grounding instructions with retrieved context
        ("human", "{input}"),       # the user's question
    ]
)


#%% 8. LOAD LOCAL LLM
print("8. Loading local LLM...")

# OllamaLLM sends inference requests to the locally running Ollama server.
# llama3.1:latest is a strong general-purpose model; swap for a smaller model
# (e.g. "mistral:latest") if hardware is constrained.
llm = OllamaLLM(model="llama3.1:latest")


#%% 9. BUILD THE HYBRID RAG CHAIN
print("9. Building the hybrid RAG chain...")

# Concatenates the .page_content of all retrieved Document objects into a single
# string separated by blank lines. This string is injected into {context}.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LangChain Expression Language (LCEL) pipes steps together using the | operator.
# Chain steps (left to right):
#
#   Step A — Build the prompt inputs dict:
#     "context" → run hybrid_retrieve(query) → format_docs() → plain-text context string
#     "input"   → RunnablePassthrough() passes the raw query string through unchanged
#
#   Step B — prompt: fills {context} and {input} into the ChatPromptTemplate
#
#   Step C — llm: sends the formatted prompt to Llama 3.1 and returns a message object
#
#   Step D — StrOutputParser: extracts the text content from the LLM message object

def prepare_inputs(query: str) -> dict:
    retrieved_docs = hybrid_retrieve(query)
    return {
        "context": format_docs(retrieved_docs),
        "input": query
    }

rag_chain = (
    RunnableLambda(prepare_inputs)
    | prompt
    | llm
    | StrOutputParser()
)


#%% 10. RUN QUERIES
print("10. Running queries...")

# rag_chain.invoke(query) runs the full pipeline end-to-end:
#   query → hybrid_retrieve → format_docs → prompt → llm → str answer
print(rag_chain.invoke("What was Nike's revenue in 2023?"))
print(rag_chain.invoke("What are the Risks Related to Operating a Global Business"))
