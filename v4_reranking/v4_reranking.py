
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from flashrank import Ranker, RerankRequest


# ==========================================
# 1. Environment and configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

CHROMA_PATH = BASE_DIR / "vectorstores" / "chroma"
COLLECTION_NAME = "courier_logistics_policy"

EMBEDDING_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = "gemini-3.5-flash-lite"

RETRIEVAL_K = 10
FINAL_K = 3
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

FALLBACK_ANSWER = (
    "I could not find this information in the provided policies."
)


# ==========================================
# 2. Embeddings and existing ChromaDB
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=API_KEY,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
)

if vectorstore._collection.count() == 0:
    raise ValueError(
        "ChromaDB collection is empty. Build the V3 index first."
    )


# ==========================================
# 3. Reranker and LLM
# ==========================================

ranker = Ranker(model_name=RERANKER_MODEL)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=API_KEY,
    temperature=0.2,
)


# ==========================================
# 4. Prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a Courier and Logistics Policy Assistant.

Answer the user's question using ONLY the provided policy context.

If the answer is not available in the context, say exactly:

"I could not find this information in the provided policies."

Do not invent, assume, or add policy information.

Policy Context:
{context}

User Question:
{question}

Give a clear and concise answer.
"""
)

chain = prompt | llm | StrOutputParser()


# ==========================================
# 5. Reranking
# ==========================================

def rerank_documents(question, documents):
    """Rerank LangChain Documents using FlashRank."""

    passages = [
        {
            "id": str(index),
            "text": document.page_content,
            "meta": document.metadata,
        }
        for index, document in enumerate(documents)
    ]

    if not passages:
        return []

    request = RerankRequest(
        query=question,
        passages=passages,
    )

    return ranker.rerank(request)


# ==========================================
# 6. Reusable RAG pipeline
# ==========================================

def ask_question(question):
    """Return answer and top reranked passages."""

    retrieved_documents = vectorstore.similarity_search(
        question,
        k=RETRIEVAL_K,
    )

    reranked_documents = rerank_documents(
        question,
        retrieved_documents,
    )

    top_documents = reranked_documents[:FINAL_K]

    if not top_documents:
        return FALLBACK_ANSWER, []

    context = "\n\n".join(
        document["text"]
        for document in top_documents
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer, top_documents


# ==========================================
# 7. Evaluation interface
# ==========================================

def evaluate_question(question):
    """Reusable interface for the comparison evaluator."""
    return ask_question(question)


# ==========================================
# 8. Interactive mode
# ==========================================

if __name__ == "__main__":
    question = input(
        "\nAsk your courier/logistics question: "
    ).strip()

    if not question:
        print("Please enter a question.")
    else:
        answer, top_documents = ask_question(question)

        print("\n========================================")
        print("RERANKED RAG ANSWER")
        print("========================================")
        print(answer)

        print("\n========================================")
        print("SOURCES")
        print("========================================")

        seen = set()

        for document in top_documents:
            metadata = document.get("meta", {})

            source = metadata.get("source", "Unknown")
            page = metadata.get("page")

            key = (source, page)

            if key not in seen:
                display_page = (
                    page + 1
                    if isinstance(page, int)
                    else "Unknown"
                )

                print(f"Document: {source}")
                print(f"Page: {display_page}")
                print(f"Reranker score: {document.get('score', 'N/A')}")
                print()

                seen.add(key)
