
import csv
import importlib.util
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

JUDGE_MODEL = "gemini-3.5-flash-lite"
FALLBACK_ANSWER = (
    "I could not find this information in the provided policies."
)
OUTPUT_JSON = BASE_DIR / "comparison_results.json"
OUTPUT_CSV = BASE_DIR / "comparison_results.csv"


# ==========================================
# 2. Same 20 questions used by V4 evaluation
# ==========================================

QUESTIONS = [
    {
        "question": "Can I change my delivery address after dispatch?",
        "expected": "After dispatch, an address change may not always be possible. Contact support immediately with the tracking number.",
        "category": "address_change",
    },
    {
        "question": "Can I change my delivery address before dispatch?",
        "expected": "An address correction may be possible before dispatch, subject to verification.",
        "category": "address_change",
    },
    {
        "question": "Can I cancel my shipment?",
        "expected": "Cancellation depends on shipment stage, status, and applicable service terms.",
        "category": "cancellation",
    },
    {
        "question": "What should I do if my parcel is lost?",
        "expected": "Provide the tracking number and contact customer support to initiate an investigation.",
        "category": "lost_parcel",
    },
    {
        "question": "What should I do if my parcel is damaged?",
        "expected": "Photograph the packaging and damaged item, report the damage promptly, and check the claim time limit.",
        "category": "damaged_parcel",
    },
    {
        "question": "How many delivery attempts are made?",
        "expected": "A delivery agent may make up to three delivery attempts.",
        "category": "delivery",
    },
    {
        "question": "What happens after a failed delivery attempt?",
        "expected": "After the final unsuccessful attempt, the shipment may be returned to the sender.",
        "category": "delivery",
    },
    {
        "question": "Can I track my shipment?",
        "expected": "Use the shipment tracking number to view the latest recorded delivery status.",
        "category": "tracking",
    },
    {
        "question": "What items are prohibited from shipping?",
        "expected": "Illegal, hazardous, or restricted items must not be sent through the service.",
        "category": "prohibited_items",
    },
    {
        "question": "How do I report a missing parcel?",
        "expected": "Provide the tracking number and contact customer support to initiate an investigation.",
        "category": "lost_parcel",
    },
    {
        "question": "Can I cancel an order after it has been dispatched?",
        "expected": "Once a shipment is in transit, cancellation may not be available; contact support for options.",
        "category": "cancellation",
    },
    {
        "question": "Who should I contact about an address correction?",
        "expected": "Before dispatch, correction may be possible subject to verification; after dispatch, contact support with the tracking number.",
        "category": "address_change",
    },
    {
        "question": "What if my tracking status says delivered but I received nothing?",
        "expected": "If the provided policy context does not explain this situation, do not invent a procedure; use the required fallback.",
        "category": "delivery",
    },
    {
        "question": "What information should I provide when contacting support?",
        "expected": "For a suspected lost parcel or post-dispatch address change, provide the tracking number. Claims may require supporting evidence.",
        "category": "support",
    },
    {
        "question": "Can every parcel be redirected after dispatch?",
        "expected": "No. An address change after dispatch may not always be possible; contact support immediately with the tracking number.",
        "category": "address_change",
    },
    {
        "question": "What is the courier company's CEO's birthday?",
        "expected": FALLBACK_ANSWER,
        "category": "out_of_scope",
    },
    {
        "question": "What is the capital of France?",
        "expected": FALLBACK_ANSWER,
        "category": "out_of_scope",
    },
    {
        "question": "What should I do if delivery is delayed?",
        "expected": "Check tracking status first and contact customer support if the status does not change within the applicable service window.",
        "category": "delivery",
    },
    {
        "question": "Can I ship restricted or hazardous items?",
        "expected": "Illegal, hazardous, or restricted items must not be sent through the service.",
        "category": "prohibited_items",
    },
    {
        "question": "What should I do if my parcel is both delayed and damaged?",
        "expected": "Check tracking status and contact support if it does not change within the applicable service window; photograph damage, report it promptly, and check the claim time limit.",
        "category": "multi_policy",
    },
]


# ==========================================
# 3. Load each version from its Python file
# ==========================================

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipeline_versions():
    v2 = load_module(
        "v2_faiss_comparison",
        BASE_DIR / "v2" / "v2_faiss.py",
    )

    v3 = load_module(
        "v3_chroma_comparison",
        BASE_DIR / "v3" / "v3_chroma.py",
    )

    v4 = load_module(
        "v4_reranking_comparison",
        BASE_DIR / "v4_reranking" / "v4_reranking.py",
    )

    return {
        "V2_FAISS": v2.evaluate_question,
        "V3_ChromaDB": v3.evaluate_question,
        "V4_Reranking": v4.evaluate_question,
    }


# ==========================================
# 4. Convert documents to context text
# ==========================================

def extract_context(documents):
    parts = []

    for document in documents or []:
        if isinstance(document, dict):
            content = document.get("text", "")
        else:
            content = getattr(document, "page_content", "")

        if isinstance(content, str) and content.strip():
            parts.append(content.strip())

    return "\n\n".join(parts)


def count_documents(documents):
    return len(documents or [])


# ==========================================
# 5. Groundedness judge
# ==========================================

def get_judge():
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Google API key not found in the project .env file."
        )

    return ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


def parse_judge_json(raw):
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end < start:
        raise ValueError("Judge response did not contain JSON.")

    return json.loads(cleaned[start:end + 1])


def judge_groundedness(judge, question, context, answer):
    if not context.strip():
        passed = answer.strip() == FALLBACK_ANSWER
        return passed, (
            "No context; required fallback used."
            if passed
            else "No context; required fallback not used."
        )

    prompt_text = f"""
You are a strict evaluator for a Courier and Logistics Policy RAG Assistant.
Use ONLY the supplied retrieved policy context.

QUESTION:
{question}

RETRIEVED POLICY CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Rules:
- grounded=true only when the answer addresses the question and its
  factual/policy claims are supported by the context.
- grounded=false for invented, contradictory, or unsupported claims.
- If context is insufficient, the required fallback is exactly:
  "{FALLBACK_ANSWER}"
- If context is insufficient and that exact fallback is used, score true.
- Do not require exact wording if policy meaning is preserved.

Return ONLY JSON:
{{"grounded": true, "reason": "brief explanation"}}
"""

    response = judge.invoke(prompt_text)
    content = getattr(response, "content", response)

    if not isinstance(content, str):
        content = str(content)

    data = parse_judge_json(content)
    grounded = data.get("grounded")

    if not isinstance(grounded, bool):
        raise ValueError("Judge did not return a Boolean score.")

    return grounded, str(data.get("reason", ""))


# ==========================================
# 6. Evaluate one question on one version
# ==========================================

def evaluate_one(version_name, evaluate_function, item, judge):
    question = item["question"]
    started = time.perf_counter()

    result = {
        "version": version_name,
        "question": question,
        "category": item["category"],
        "expected_answer": item["expected"],
        "generated_answer": "",
        "retrieved_count": 0,
        "context_found": False,
        "latency_seconds": None,
        "groundedness_score": None,
        "groundedness_reason": "",
        "error": "",
        "judge_error": "",
    }

    try:
        answer, documents = evaluate_function(question)

        result["latency_seconds"] = round(
            time.perf_counter() - started, 3
        )

        result["generated_answer"] = (
            answer.strip() if isinstance(answer, str) else str(answer)
        )
        result["retrieved_count"] = count_documents(documents)

        context = extract_context(documents)
        result["context_found"] = bool(context.strip())

        grounded, reason = judge_groundedness(
            judge,
            question,
            context,
            result["generated_answer"],
        )

        result["groundedness_score"] = grounded
        result["groundedness_reason"] = reason

    except Exception as exc:
        result["latency_seconds"] = round(
            time.perf_counter() - started, 3
        )
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ==========================================
# 7. Summary metrics
# ==========================================

def summarize_results(results):
    summaries = {}

    for version in sorted({r["version"] for r in results}):
        rows = [r for r in results if r["version"] == version]
        scored = [
            r for r in rows
            if isinstance(r["groundedness_score"], bool)
        ]

        passes = sum(
            r["groundedness_score"] is True for r in scored
        )
        failures = sum(
            r["groundedness_score"] is False for r in scored
        )

        latencies = [
            r["latency_seconds"]
            for r in rows
            if r["latency_seconds"] is not None
        ]

        summaries[version] = {
            "questions": len(rows),
            "answers_generated": sum(
                bool(r["generated_answer"]) and not r["error"]
                for r in rows
            ),
            "pipeline_errors": sum(bool(r["error"]) for r in rows),
            "questions_with_context": sum(
                r["context_found"] for r in rows
            ),
            "average_retrieved_documents": round(
                sum(r["retrieved_count"] for r in rows) / len(rows), 2
            ) if rows else None,
            "groundedness_scored": len(scored),
            "groundedness_passes": passes,
            "groundedness_failures": failures,
            "unscored": len(rows) - len(scored),
            "groundedness_pass_rate_percent": round(
                100 * passes / len(scored), 2
            ) if scored else None,
            "average_latency_seconds": round(
                sum(latencies) / len(latencies), 3
            ) if latencies else None,
        }

    return summaries


# ==========================================
# 8. Save JSON and CSV reports
# ==========================================

def save_reports(results):
    payload = {
        "evaluated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "judge_model": JUDGE_MODEL,
        "question_count": len(QUESTIONS),
        "versions": list(dict.fromkeys(
            r["version"] for r in results
        )),
        "summary": summarize_results(results),
        "results": results,
        "metric_notes": {
            "latency": (
                "Includes retrieval, answer generation, and groundedness "
                "judging because the timer covers the complete evaluation call."
            ),
            "groundedness": (
                "LLM-judge estimate, not a guarantee of factual accuracy."
            ),
            "retrieval_quality": (
                "Context presence and document count are proxies, not "
                "a complete retrieval-quality metric."
            ),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if results:
        with OUTPUT_CSV.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print("\nComparison completed.")
    print(f"JSON report: {OUTPUT_JSON}")
    print(f"CSV report:  {OUTPUT_CSV}")

    print("\nSUMMARY")
    print(json.dumps(payload["summary"], indent=2))

    return payload


# ==========================================
# 9. Run comparison
# ==========================================

def main():
    print("Loading V2, V3, and V4 pipelines...")
    pipelines = load_pipeline_versions()
    judge = get_judge()

    results = []
    total = len(QUESTIONS) * len(pipelines)
    completed = 0

    for item in QUESTIONS:
        print(f"\nQuestion: {item['question']}")

        for version_name, evaluate_function in pipelines.items():
            completed += 1
            print(f"[{completed}/{total}] Evaluating {version_name}...")

            result = evaluate_one(
                version_name,
                evaluate_function,
                item,
                judge,
            )
            results.append(result)

            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                score = result["groundedness_score"]
                label = (
                    "PASS" if score is True
                    else "FAIL" if score is False
                    else "UNSCORED"
                )
                print(
                    f"  {label} | "
                    f"{result['latency_seconds']} seconds"
                )

    save_reports(results)


if __name__ == "__main__":
    main()
