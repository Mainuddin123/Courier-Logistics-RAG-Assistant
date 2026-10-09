
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. Project paths and environment
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in the project .env file.")

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
FAISS_DIR = BASE_DIR / "vectorstores" / "faiss"


# ==========================================
# 2. Create embeddings
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
)


# ==========================================
# 3. Load existing FAISS or create it
# ==========================================

def initialize_vectorstore():
    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    # Reuse the saved index whenever both files exist.
    if index_file.exists() and metadata_file.exists():
        print("Loading existing FAISS vector store...")

        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Existing FAISS index loaded successfully.")
        return vectorstore

    # Build a new index only if no complete saved index exists.
    print("Saved FAISS index not found. Creating a new index...")

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    documents = []

    for pdf_file in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()
        documents.extend(pdf_documents)

        print(
            f"Loaded: {pdf_file.name} | "
            f"Pages: {len(pdf_documents)}"
        )

    if not documents:
        raise ValueError(
            f"No PDF documents found in {DOCUMENTS_DIR}"
        )

    print(f"Total pages loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print("New FAISS vector store created and saved.")

    return vectorstore


vectorstore = initialize_vectorstore()


# ==========================================
# 4. Create retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================================
# 5. Create Gemini LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    google_api_key=api_key,
)


# ==========================================
# 6. Create RAG prompt
# ==========================================

FALLBACK_ANSWER = (
    "I could not find this information in the provided policies."
)

prompt = ChatPromptTemplate.from_template(
    """
You are a Courier and Logistics Policy Assistant.

Answer the user's question using ONLY the provided policy context.

If the answer cannot be found in the context, respond exactly:
"I could not find this information in the provided policies."

Do not invent or assume policy information.

Policy Context:
{context}

User Question:
{question}

Answer clearly and concisely.
"""
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
# 8. Ask a question
# ==========================================

def ask_question(question):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return FALLBACK_ANSWER, []

    context = format_documents(retrieved_docs)

    chain = prompt | llm | StrOutputParser()

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
    Reusable interface for comparing V2 with V3 and V4.

    Returns:
        answer: Generated answer.
        retrieved_docs: Retrieved LangChain documents.
    """
    return ask_question(question)


# ==========================================
# 10. Interactive terminal mode
# ==========================================

def main():
    print("\nCourier & Logistics RAG Assistant — V2 (FAISS)")

    while True:
        question = input(
            "\nAsk a policy question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting V2.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, retrieved_docs = ask_question(question)

            print("\n" + "=" * 40)
            print("RAG ANSWER")
            print("=" * 40)
            print(answer)

            print("\n" + "=" * 40)
            print("SOURCES")
            print("=" * 40)

            seen_sources = set()

            for document in retrieved_docs:
                source = document.metadata.get(
                    "source", "Unknown source"
                )
                page = document.metadata.get("page")

                source_key = (source, page)

                if source_key not in seen_sources:
                    page_number = (
                        page + 1 if page is not None else "Unknown"
                    )

                    print(f"Document: {Path(source).name}")
                    print(f"Page: {page_number}\n")

                    seen_sources.add(source_key)

        except Exception as exc:
            print(f"Error while answering question: {exc}")


if __name__ == "__main__":
    main()
