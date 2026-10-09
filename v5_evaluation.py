import os
import csv
import time

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from flashrank import Ranker, RerankRequest


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in .env file")


print("=" * 70)
print("COURIER & LOGISTICS RAG - FINAL EVALUATION")
print("=" * 70)


# ============================================================
# 2. EVALUATION QUESTIONS
# ============================================================

questions = [

    # Damaged / Lost Parcel
    "What should I do if my parcel is damaged?",
    "How should I document damage to a parcel?",
    "When should I report a damaged parcel?",
    "What evidence may be required for a damage claim?",

    "What should I do if my parcel is lost?",
    "What information should I provide for a lost parcel investigation?",

    # Address
    "Can I change my delivery address?",
    "Can I correct my address before dispatch?",
    "Can I change my address after dispatch?",
    "What happens if my delivery address is incorrect?",

    # Cancellation / Delivery
    "How can I cancel my shipment?",
    "Can I cancel my order before dispatch?",
    "What should I do if I want to cancel a delivery?",
    "Is shipment cancellation always possible?",

    "What should I do if there is a delivery problem?",
    "What happens if a delivery attempt fails?",
    "What information should I provide when contacting support about delivery?",

    # Prohibited Items
    "What items are prohibited?",
    "Can I send a prohibited item through the courier?",
    "What should I check before shipping an item?",

    # Return / Refund
    "How can I request a refund?",
    "What is the process for returning a parcel?",
    "What condition should I keep the parcel in for a return?",
    "When is a refund issued?",

    # Out of domain
    "What is the company's employee leave policy?"
]


# ============================================================
# 3. GEMINI EMBEDDINGS
# ============================================================

print("\nInitializing Gemini embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

print("Embeddings initialized.")


# ============================================================
# 4. LOAD V2 - FAISS
# ============================================================

print("\nLoading V2 FAISS...")

faiss_db = None

try:

    faiss_db = FAISS.load_local(
        "vectorstores/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("V2 FAISS loaded successfully.")

except Exception as e:

    print("V2 FAISS error:")
    print(e)


# ============================================================
# 5. LOAD V3 - CHROMADB
# ============================================================

print("\nLoading V3 ChromaDB...")

chroma_db = None

try:

    chroma_db = Chroma(
        persist_directory="vectorstores/chroma",
        embedding_function=embeddings,

        # IMPORTANT:
        # Your actual collection contains 9 documents
        collection_name="courier_logistics_policy"
    )

    count = chroma_db._collection.count()

    print("V3 ChromaDB loaded successfully.")
    print("Chroma documents:", count)

except Exception as e:

    print("V3 ChromaDB error:")
    print(e)


# ============================================================
# 6. FLASHRANK
# ============================================================

print("\nInitializing FlashRank...")

ranker = None

try:

    ranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir="./.cache"
    )

    print("FlashRank initialized.")

except Exception as e:

    print("FlashRank error:")
    print(e)


# ============================================================
# 7. GEMINI LLM
# ============================================================

print("\nInitializing Gemini LLM...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

print("Gemini LLM initialized.")


# ============================================================
# 8. HELPER - RESPONSE TEXT
# ============================================================

def get_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 9. RERANK DOCUMENTS
# ============================================================

def rerank_documents(question, documents):

    if not documents:
        return []

    if ranker is None:
        return documents[:3]

    passages = []

    for index, doc in enumerate(documents):

        passages.append({
            "id": str(index),
            "text": doc.page_content,
            "meta": doc.metadata
        })

    try:

        rerank_request = RerankRequest(
            query=question,
            passages=passages
        )

        results = ranker.rerank(rerank_request)

        final_docs = []

        for result in results[:3]:

            index = int(result["id"])

            final_docs.append(
                documents[index]
            )

        return final_docs

    except Exception as e:

        print("FlashRank error:", e)

        return documents[:3]


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    if not documents:

        return (
            "Sorry, I could not find this information "
            "in the available courier policy documents."
        )

    context_parts = []

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a Courier and Logistics Policy Assistant.

Answer the user's question ONLY using the policy context below.

Rules:

1. Use only the provided policy information.
2. Do not invent or assume policies.
3. Do not use outside knowledge.
4. If the answer is not available in the policy context,
   say:

"Sorry, I could not find this information in the available courier policy documents."

5. Keep the answer clear and concise.

User Question:
{question}

Policy Context:
{context}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        return get_response_text(response)

    except Exception as e:

        return f"LLM error: {e}"


# ============================================================
# 11. EVALUATION
# ============================================================

results = []

print("\n")
print("=" * 70)
print("STARTING 25-QUESTION RAG EVALUATION")
print("=" * 70)


for number, question in enumerate(
    questions,
    start=1
):

    print("\n" + "=" * 70)
    print(f"QUESTION {number}/{len(questions)}")
    print("=" * 70)

    print(question)


    # ========================================================
    # V2 - FAISS
    # ========================================================

    v2_docs = []
    v2_time = 0

    if faiss_db:

        start = time.perf_counter()

        try:

            v2_docs = faiss_db.similarity_search(
                question,
                k=3
            )

        except Exception as e:

            print("V2 error:", e)

        v2_time = time.perf_counter() - start


    # ========================================================
    # V3 - CHROMADB
    # ========================================================

    v3_docs = []
    v3_time = 0

    if chroma_db:

        start = time.perf_counter()

        try:

            v3_docs = chroma_db.similarity_search(
                question,
                k=5
            )

        except Exception as e:

            print("V3 error:", e)

        v3_time = time.perf_counter() - start


    # ========================================================
    # V4 - CHROMADB + FLASHRANK
    # ========================================================

    start = time.perf_counter()

    # Use ChromaDB as the retrieval source
    # and FlashRank as the reranker.

    v4_initial_docs = v3_docs

    v4_docs = rerank_documents(
        question,
        v4_initial_docs
    )

    v4_time = time.perf_counter() - start


    # ========================================================
    # DISPLAY RETRIEVAL
    # ========================================================

    print("\nRETRIEVAL")

    print("-" * 40)

    print(
        f"V2 FAISS documents     : {len(v2_docs)}"
    )

    print(
        f"V3 ChromaDB documents  : {len(v3_docs)}"
    )

    print(
        f"V4 Reranked documents  : {len(v4_docs)}"
    )


    # ========================================================
    # GENERATE V4 ANSWER
    # ========================================================

    print("\nGenerating V4 answer...")

    v4_answer = generate_answer(
        question,
        v4_docs
    )


    print("\nV4 ANSWER")
    print("-" * 40)

    print(v4_answer)


    # ========================================================
    # SOURCES
    # ========================================================

    sources = []

    for doc in v4_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        sources.append(
            f"{source} | Page {page}"
        )


    print("\nSOURCES")
    print("-" * 40)

    if sources:

        for source in sources:
            print(source)

    else:

        print("No sources found.")


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "question": question,

        "v2_documents": len(v2_docs),
        "v2_time": round(v2_time, 4),

        "v3_documents": len(v3_docs),
        "v3_time": round(v3_time, 4),

        "v4_documents": len(v4_docs),
        "v4_time": round(v4_time, 4),

        "v4_answer": v4_answer,

        "v4_sources": " || ".join(sources)

    })


# ============================================================
# 12. SAVE CSV
# ============================================================

output_file = "evaluation_results.csv"

fieldnames = [

    "question",

    "v2_documents",
    "v2_time",

    "v3_documents",
    "v3_time",

    "v4_documents",
    "v4_time",

    "v4_answer",
    "v4_sources"
]


with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"Questions evaluated : {len(results)}"
)

print(
    f"Results saved to    : {output_file}"
)

print("\nPipeline:")

print("V1 = LangChain + Gemini")
print("V2 = FAISS RAG")
print("V3 = ChromaDB RAG")
print("V4 = ChromaDB + FlashRank")
print("Evaluation = 25 questions")

print("\nEvaluation file:")
print(output_file)

print("=" * 70)