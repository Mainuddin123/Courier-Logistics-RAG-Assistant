import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is not set in .env"
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


# ============================================================
# QUERY TRANSFORMATION
# ============================================================

def transform_query(question):

    prompt = f"""
You are a search-query optimizer for a
Courier and Logistics Policy Assistant.

Rewrite the user's question into a clear,
specific search query suitable for retrieving
information from courier policy documents.

Rules:

1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not add information that was not implied.
4. Use important courier/logistics terminology.
5. Return ONLY the rewritten search query.

User question:
{question}

Rewritten search query:
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    return item.get(
                        "text",
                        ""
                    ).strip()

    return str(content).strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    questions = [
        "What if it got damaged?",
        "Can I change where it is delivered?",
        "How do I get my money back?"
    ]

    for question in questions:

        print("\nOriginal:")
        print(question)

        rewritten = transform_query(
            question
        )

        print("\nRewritten:")
        print(rewritten)

        print("-" * 60)

        