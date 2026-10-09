import pandas as pd

FILE = "evaluation_results.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("RAG EVALUATION METRICS")
print("=" * 70)

print(f"\nTotal Questions : {len(df)}")

# ------------------------------------------------------------
# Retrieval statistics
# ------------------------------------------------------------

print("\nRETRIEVAL STATISTICS")
print("-" * 40)

print(
    f"Average V2 documents : "
    f"{df['v2_documents'].mean():.2f}"
)

print(
    f"Average V3 documents : "
    f"{df['v3_documents'].mean():.2f}"
)

print(
    f"Average V4 documents : "
    f"{df['v4_documents'].mean():.2f}"
)

# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------

print("\nLATENCY")
print("-" * 40)

print(
    f"Average V2 time : "
    f"{df['v2_time'].mean():.4f} sec"
)

print(
    f"Average V3 time : "
    f"{df['v3_time'].mean():.4f} sec"
)

print(
    f"Average V4 time : "
    f"{df['v4_time'].mean():.4f} sec"
)

# ------------------------------------------------------------
# Empty / failed answers
# ------------------------------------------------------------

answers = df["v4_answer"].fillna("").astype(str)

no_info = answers.str.contains(
    "could not find this information",
    case=False,
    na=False
)

print("\nANSWER STATISTICS")
print("-" * 40)

print(
    f"Answers generated : "
    f"{len(df) - no_info.sum()}"
)

print(
    f"No-information answers : "
    f"{no_info.sum()}"
)

# ------------------------------------------------------------
# Source statistics
# ------------------------------------------------------------

sources = df["v4_sources"].fillna("").astype(str)

with_sources = (sources.str.strip() != "").sum()

print(
    f"Answers with sources : "
    f"{with_sources}/{len(df)}"
)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("V1 = LangChain + Gemini")
print("V2 = FAISS RAG")
print("V3 = ChromaDB RAG")
print("V4 = ChromaDB + FlashRank")
print(f"Evaluation questions = {len(df)}")

print("\nMetrics calculated successfully.")
print("=" * 70)