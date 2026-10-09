import os

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

from flashrank import Ranker, RerankRequest


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATA_DIR = "data/documents"
CHROMA_DIR = "vectorstores/chroma"


# ============================================================
# 1. LOAD POLICY DOCUMENTS
# ============================================================

documents = []

for filename in os.listdir(DATA_DIR):

    if filename.lower().endswith(".pdf"):

        path = os.path.join(DATA_DIR, filename)

        loader = PyPDFLoader(path)

        docs = loader.load()

        documents.extend(docs)

        print(
            f"Loaded: {filename} | Pages: {len(docs)}"
        )


print(f"\nTotal pages loaded: {len(documents)}")


# ============================================================
# 2. CHUNK DOCUMENTS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ============================================================
# 3. CREATE BM25 INDEX
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created.")


# ============================================================
# 4. LOAD CHROMADB
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

print("ChromaDB loaded.")


# ============================================================
# 5. FLASHRANK
# ============================================================

ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir="./flashrank_cache"
)

print("FlashRank initialized.")


# ============================================================
# 6. HYBRID RETRIEVAL
# ============================================================

def hybrid_search(query, k=5):

    # -------------------------------
    # Semantic / Vector Search
    # -------------------------------

    vector_docs = vectorstore.similarity_search(
        query,
        k=k
    )

    # -------------------------------
    # Keyword / BM25 Search
    # -------------------------------

    tokenized_query = query.lower().split()

    bm25_docs = bm25.get_top_n(
        tokenized_query,
        chunks,
        n=k
    )

    # -------------------------------
    # Combine Results
    # -------------------------------

    combined = []

    seen = set()

    for doc in vector_docs + bm25_docs:

        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )

        if key not in seen:

            seen.add(key)

            combined.append(doc)

    return combined


# ============================================================
# 7. FLASHRANK RERANKING
# ============================================================

def rerank_documents(query, documents):

    passages = []

    for i, doc in enumerate(documents):

        passages.append({
            "id": str(i),
            "text": doc.page_content,
            "meta": doc.metadata
        })

    request = RerankRequest(
    query=query,
    passages=passages
)

    results = ranker.rerank(request)

    return results[:3]


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk your courier/logistics question: "
    )

    print("\n" + "=" * 50)
    print("HYBRID RETRIEVAL")
    print("=" * 50)

    retrieved = hybrid_search(
        question,
        k=5
    )

    print(
        f"\nCombined documents retrieved: "
        f"{len(retrieved)}"
    )

    # -------------------------------
    # Reranking
    # -------------------------------

    ranked = rerank_documents(
        question,
        retrieved
    )

    print("\nAFTER FLASHRANK")
    print("-" * 30)

    for i, result in enumerate(
        ranked,
        start=1
    ):

        print(
            f"\n{i}. Score: "
            f"{result.get('score', 0):.4f}"
        )

        print(
            result["text"][:300]
        )

        metadata = result.get(
            "meta",
            {}
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown"
            )
        )

        print(
            "Page:",
            metadata.get(
                "page",
                0
            ) + 1
        )

    print("\n" + "=" * 50)
    print("HYBRID RETRIEVAL COMPLETED")
    print("=" * 50)