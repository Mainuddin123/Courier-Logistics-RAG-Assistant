
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. Load environment variables
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# ==========================================
# 2. Configuration
# ==========================================

DOCUMENTS_PATH = BASE_DIR / "data" / "documents"
CHROMA_PATH = BASE_DIR / "vectorstores" / "chroma"
COLLECTION_NAME = "courier_logistics_policy"

EMBEDDING_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


# ==========================================
# 3. Initialize embeddings and LLM
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.2,
    google_api_key=API_KEY,
)


# ==========================================
# 4. RAG prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a Courier and Logistics Policy Assistant.

Answer the user's question using ONLY the provided policy context.

If the answer cannot be found in the context, say:

"I could not find this information in the provided policies."

Do not invent, assume, or add policy information.

Policy Context:
{context}

User Question:
{question}

Answer clearly and concisely.
"""
)

chain = prompt | llm | StrOutputParser()


# ==========================================
# 5. Load and split policy PDFs
# ==========================================

def load_policy_chunks():
    """Load PDFs and split them into reusable document chunks."""

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_PATH}"
        )

    pdf_files = sorted(DOCUMENTS_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_PATH}"
        )

    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()
        documents.extend(pdf_documents)

        print(
            f"Loaded: {pdf_file.name} | "
            f"Pages: {len(pdf_documents)}"
        )

    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# ==========================================
# 6. Initialize ChromaDB
# ==========================================

def initialize_vectorstore():
    """
    Reuse an existing ChromaDB collection when available.
    Build the collection from PDFs if it is not available.
    """

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    existing_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    if existing_store._collection.count() > 0:
        print(
            "Loaded existing ChromaDB collection: "
            f"{COLLECTION_NAME}"
        )
        print(
            "Existing collection documents: "
            f"{existing_store._collection.count()}"
        )
        return existing_store

    print("No existing ChromaDB documents found. Building index...")

    chunks = load_policy_chunks()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )

    print("ChromaDB index created successfully.")

    return vectorstore


# Initialize once when this module is imported.
vectorstore = initialize_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)


# ==========================================
# 7. Format retrieved documents
# ==========================================

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ==========================================
# 8. RAG question-answering function
# ==========================================

def ask_question(question):
    """Return the generated answer and retrieved documents."""

    retrieved_docs = retriever.invoke(question)

    context = format_documents(retrieved_docs)

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer, retrieved_docs


# ==========================================
# 9. Evaluation interface
# ==========================================

def evaluate_question(question):
    """
    Reusable interface for the comparison evaluator.

    Returns:
        answer: Generated answer string.
        retrieved_docs: List of retrieved LangChain Documents.
    """
    return ask_question(question)


# ==========================================
# 10. Interactive mode
# ==========================================

if __name__ == "__main__":
    question = input(
        "\nAsk your courier/logistics question: "
    ).strip()

    if not question:
        print("Please enter a question.")
    else:
        answer, retrieved_docs = ask_question(question)

        print("\n==============================")
        print("RAG ANSWER")
        print("==============================")
        print(answer)

        print("\n==============================")
        print("SOURCES")
        print("==============================")

        seen_sources = set()

        for document in retrieved_docs:
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get("page")

            source_key = (source, page)

            if source_key not in seen_sources:
                display_page = (
                    page + 1 if isinstance(page, int)
                    else "Unknown"
                )

                print(f"Document: {source}")
                print(f"Page: {display_page}")
                print()

                seen_sources.add(source_key)
