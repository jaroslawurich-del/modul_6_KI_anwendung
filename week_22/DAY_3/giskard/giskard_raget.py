# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:41:10 2025

@author: milos
"""

import os
import warnings
import pandas as pd

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from giskard import Model, scan
from giskard.rag import (
    AgentAnswer,
    KnowledgeBase,
    QATestset,
    RAGReport,
    evaluate,
    generate_testset,
)


### Section 1: Giskard and Ollama Configuration

import giskard
from langchain_ollama import OllamaLLM

# Set the base URL for local Ollama API endpoint
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

# Configure Giskard to use Ollama's Mistral model for LLM operations
giskard.llm.set_llm_model("ollama/mistral:instruct")

# Configure Giskard to use Ollama's nomic-embed-text for embeddings
giskard.llm.set_embedding_model("ollama/nomic-embed-text:latest")

# Initialize the Ollama LLM with Mistral model
# temperature=0 ensures deterministic responses for testing
llm = OllamaLLM(
    model="mistral:instruct", 
    base_url='http://localhost:11434',
    temperature=0
)

# Prevents errors when Ollama doesn't support certain LiteLLM parameters
# Drops unsupported function calling parameters
os.environ["LITELLM_DROP_PARAMS"] = "true"

### Section 2: RAG System Setup

# Configure pandas to display full column content
pd.set_option("display.max_colwidth", 400)
warnings.filterwarnings("ignore")

# Load the PDF document containing banking supervision information
loader = PyPDFLoader(file_path="C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/banking_supervision_report.pdf")
documents = loader.load()  # Extracts all pages as Document objects

# Initialize embedding model for converting text to vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Split documents into smaller chunks for better retrieval
# chunk_size=500: Each chunk contains ~500 characters
# chunk_overlap=50: Adjacent chunks share 50 characters to maintain context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
text_chunks = text_splitter.split_documents(documents)

# Create FAISS vector database from text chunks
# Converts chunks to embeddings and stores them for similarity search
vectorstore = FAISS.from_documents(text_chunks, embeddings)

# Create retriever that fetches the 4 most relevant chunks per query
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize conversation memory to track chat history
# return_messages=True: Returns messages in LangChain format
# output_key="answer": Specifies which output to store in memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational RAG chain combining LLM, retriever, and memory
# This enables context-aware question answering with conversation history
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True  # Returns source chunks with answers
)

### Section 3: Giskard Model Scanning

# Wrapper function for Giskard to evaluate the RAG model
# Takes a DataFrame with 'question' column and returns predictions
def model_predict(df: pd.DataFrame):
    results = []
    for question in df["question"]:
        # Generate answer using the QA chain
        response = qa_chain({"question": question})
        results.append(response["answer"])
    return results

# Wrap the model for Giskard's testing framework
giskard_model = Model(
    model=model_predict,
    model_type="text_generation",
    name="Banking Supervision Question Answering",
    description="A model that answers questions about ECB Banking Supervision report",
    feature_names=["question"],
)

# Scan the model for vulnerabilities and issues
# Checks for bias, robustness, performance issues, etc.
scan_report = scan(giskard_model)

# Export scan results to HTML for review
scan_report.to_html("C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/scan_report.html")

### Section 4: Test Set Generation

# Extract text content from chunks for knowledge base
text_chunks_content = [chunk.page_content for chunk in text_chunks]

# Create DataFrame for Giskard knowledge base
knowledge_base_df = pd.DataFrame(text_chunks_content, columns=["text"])
knowledge_base = KnowledgeBase(knowledge_base_df)

# Automatically generate 100 test questions from the knowledge base
# Uses LLM to create diverse questions about banking supervision
testset = generate_testset(
    knowledge_base=knowledge_base,
    num_questions=100,
    agent_description="A chatbot answering questions about banking supervision procedures and methodologies.",
    language="en",
)

# Save generated testset to JSONL file for later use
testset.save("C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/banking_supervision_testset.jsonl")

# Reload testset from file
testset = QATestset.load("C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/banking_supervision_testset.jsonl")
testset.to_pandas().head(5)  # Display first 5 questions

### Section 5: RAG Evaluation

# Function that answers questions and returns structured responses
# Required format for Giskard's RAG evaluation
def answer_fn(question: str, history: list[dict] = None) -> AgentAnswer:
    if history:
        # Convert conversation history to LangChain format
        # Pairs user questions with assistant responses
        chat_history = [
            (msg["content"], history[i+1]["content"]) 
            for i, msg in enumerate(history[:-1]) 
            if i % 2 == 0 and msg["role"] == "user"
        ]
        response = qa_chain({"question": question, "chat_history": chat_history})
    else:
        # Single-turn question without history
        response = qa_chain({"question": question})
    
    # Extract retrieved source documents
    source_docs = [doc.page_content for doc in response.get("source_documents", [])]
    
    # Return structured answer with sources
    return AgentAnswer(
        message=response["answer"],
        documents=source_docs
    )

# Evaluate RAG system using RAGAS metrics
# Measures context recall (retrieval quality) and precision (relevance)
rag_report = evaluate(
    answer_fn,
    testset=testset,
    knowledge_base=knowledge_base,
)

# Save evaluation results
rag_report.save("C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/banking_supervision_report")

# Reload the saved report for analysis
rag_report = RAGReport.load("C:/DATEN/educs/Modul 6 KI-Anwendungen/modul_6_KI_anwendung/week_22/DAY_3/giskard/banking_supervision_report")

