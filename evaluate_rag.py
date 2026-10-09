import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Reuse the existing RAG pipeline from app.py.
# This assumes importing app.py does not launch a separate Streamlit run.
from app import hybrid_retrieve, rerank_results, generate_answer

load_dotenv(override=True)

JUDGE_MODEL = "gemini-3.5-flash-lite"
FALLBACK_ANSWER = "The information was not found in the policy documents."
OUTPUT_FILE = Path("rag_evaluation_results.json")

EVALUATION_QUESTIONS = [
    {"question": "Can I change my delivery address after dispatch?", "expected": "After dispatch, an address change may not always be possible. Contact support immediately with the tracking number.", "category": "address_change"},
    {"question": "Can I change my delivery address before dispatch?", "expected": "An address correction may be possible before dispatch, subject to verification.", "category": "address_change"},
    {"question": "Can I cancel my shipment?", "expected": "Cancellation depends on shipment stage, status, and applicable service terms.", "category": "cancellation"},
    {"question": "What should I do if my parcel is lost?", "expected": "Provide the tracking number and contact customer support to initiate an investigation.", "category": "lost_parcel"},
    {"question": "What should I do if my parcel is damaged?", "expected": "Photograph the packaging and damaged item, report the damage promptly, and check the claim time limit.", "category": "damaged_parcel"},
    {"question": "How many delivery attempts are made?", "expected": "A delivery agent may make up to three delivery attempts.", "category": "delivery"},
    {"question": "What happens after a failed delivery attempt?", "expected": "After the final unsuccessful attempt, the shipment may be returned to the sender.", "category": "delivery"},
    {"question": "Can I track my shipment?", "expected": "Use the shipment tracking number to view the latest recorded delivery status.", "category": "tracking"},
    {"question": "What items are prohibited from shipping?", "expected": "Illegal, hazardous, or restricted items must not be sent through the service.", "category": "prohibited_items"},
    {"question": "How do I report a missing parcel?", "expected": "Provide the tracking number and contact customer support to initiate an investigation.", "category": "lost_parcel"},
    {"question": "Can I cancel an order after it has been dispatched?", "expected": "Once a shipment is in transit, cancellation may not be available; contact support for options.", "category": "cancellation"},
    {"question": "Who should I contact about an address correction?", "expected": "Before dispatch, correction may be possible subject to verification; after dispatch, contact support with the tracking number.", "category": "address_change"},
    {"question": "What if my tracking status says delivered but I received nothing?", "expected": "If the provided policy context does not explain this situation, do not invent a procedure; use the required fallback.", "category": "delivery"},
    {"question": "What information should I provide when contacting support?", "expected": "For a suspected lost parcel or post-dispatch address change, provide the tracking number. Claims may require supporting evidence.", "category": "support"},
    {"question": "Can every parcel be redirected after dispatch?", "expected": "No. An address change after dispatch may not always be possible; contact support immediately with the tracking number.", "category": "address_change"},
    {"question": "What is the courier company's CEO's birthday?", "expected": FALLBACK_ANSWER, "category": "out_of_scope"},
    {"question": "What is the capital of France?", "expected": FALLBACK_ANSWER, "category": "out_of_scope"},
    {"question": "What should I do if delivery is delayed?", "expected": "Check tracking status first and contact customer support if the status does not change within the applicable service window.", "category": "delivery"},
    {"question": "Can I ship restricted or hazardous items?", "expected": "Illegal, hazardous, or restricted items must not be sent through the service.", "category": "prohibited_items"},
    {"question": "What should I do if my parcel is both delayed and damaged?", "expected": "Check tracking status and contact support if it does not change within the applicable service window; photograph damage, report it promptly, and check the claim time limit.", "category": "multi_policy"},
]


@st.cache_resource
def get_judge():
    """Create one reusable Gemini judge client."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Check GOOGLE_API_KEY in .env.")
    return ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


def response_text(response):
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


def parse_json_response(raw):
    """Parse JSON even when the model wraps it in a Markdown code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge response did not contain a JSON object.")
    return json.loads(cleaned[start:end + 1])


def judge_groundedness(question, context, answer):
    """Return True/False, or None if judging fails."""
    if not context.strip():
        passed = answer.strip() == FALLBACK_ANSWER
        return {
            "grounded": passed,
            "reason": (
                "No reranked context was available; required fallback was used."
                if passed else
                "No reranked context was available, but the required fallback was not used."
            ),
            "judge_error": None,
        }

    prompt = f"""
You are a strict evaluator for a Courier and Logistics Policy RAG Assistant.
Use ONLY the supplied retrieved policy context. Do not use outside knowledge.

USER QUESTION:
{question}

RETRIEVED POLICY CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Rules:
- grounded=true only if the answer addresses the question and factual/policy claims
  are supported by the supplied context.
- grounded=false if it invents policy, contradicts the context, or makes unsupported claims.
- If context is insufficient, the required fallback is exactly:
  "{FALLBACK_ANSWER}"
- If context is insufficient and the answer uses that exact fallback, score true.
- Do not require exact wording when the answer preserves the policy meaning and conditions.
Return ONLY valid JSON with keys "grounded" (Boolean) and "reason" (brief string).
"""
    try:
        raw = response_text(get_judge().invoke(prompt))
        data = parse_json_response(raw)
        grounded = data.get("grounded")
        if not isinstance(grounded, bool):
            raise ValueError("Judge did not return a Boolean grounded value.")
        return {
            "grounded": grounded,
            "reason": str(data.get("reason", "")),
            "judge_error": None,
        }
    except Exception as exc:
        return {
            "grounded": None,
            "reason": "Groundedness scoring failed.",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_question(item):
    """Run hybrid retrieval, reranking, generation, and groundedness scoring."""
    question = item["question"]
    result = {
        "question": question,
        "expected_answer": item["expected"],
        "category": item["category"],
        "retrieved_count": 0,
        "reranked_count": 0,
        "context_found": False,
        "retrieved_context": "",
        "generated_answer": "",
        "groundedness_score": None,
        "groundedness_reason": "",
        "judge_error": None,
        "error": None,
    }

    try:
        retrieved = hybrid_retrieve(question) or []
        result["retrieved_count"] = len(retrieved)
        reranked = rerank_results(question, retrieved) or []
        result["reranked_count"] = len(reranked)

        context_parts = []
        for document in reranked:
            if isinstance(document, dict):
                text = document.get("text", "")
                if isinstance(text, str) and text.strip():
                    context_parts.append(text.strip())

        context = "\n\n".join(context_parts)
        result["retrieved_context"] = context
        result["context_found"] = bool(context.strip())

        if context:
            answer = generate_answer(question, reranked)
            result["generated_answer"] = (
                answer.strip() if isinstance(answer, str) else str(answer).strip()
            )
        else:
            result["generated_answer"] = FALLBACK_ANSWER

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    judged = judge_groundedness(
        question, result["retrieved_context"], result["generated_answer"]
    )
    result["groundedness_score"] = judged["grounded"]
    result["groundedness_reason"] = judged["reason"]
    result["judge_error"] = judged["judge_error"]
    return result


def save_results(results):
    scored = [r for r in results if isinstance(r["groundedness_score"], bool)]
    passes = sum(r["groundedness_score"] is True for r in scored)
    failures = sum(r["groundedness_score"] is False for r in scored)
    unscored = sum(r["groundedness_score"] is None for r in results)
    pipeline_errors = sum(bool(r["error"]) for r in results)
    payload = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "total_questions": len(results),
        "questions_with_context": sum(r["context_found"] for r in results),
        "answers_generated": sum(bool(r["generated_answer"]) and not r["error"] for r in results),
        "pipeline_errors": pipeline_errors,
        "groundedness_scored": len(scored),
        "groundedness_passes": passes,
        "groundedness_failures": failures,
        "judge_errors_or_unscored": unscored,
        "groundedness_pass_rate_percent": round(100 * passes / len(scored), 2) if scored else None,
        "results": results,
    }
    OUTPUT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main():
    st.set_page_config(
        page_title="Courier & Logistics RAG Evaluation",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Courier & Logistics RAG Evaluation")
    st.caption(
        "20 questions · Hybrid retrieval · FlashRank reranking · Gemini generation "
        "· LLM-as-a-judge groundedness"
    )
    st.info(
        "Groundedness is an LLM-judge estimate, not a guarantee. Review failed cases "
        "and judge errors before reporting final metrics."
    )

    if st.button("Run Full Evaluation", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, item in enumerate(EVALUATION_QUESTIONS):
            status.write(f"Evaluating {index + 1}/{len(EVALUATION_QUESTIONS)}: {item['question']}")
            results.append(evaluate_question(item))
            progress.progress((index + 1) / len(EVALUATION_QUESTIONS))

        payload = save_results(results)
        status.success("Evaluation finished. Results saved.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", payload["total_questions"])
        c2.metric("Context Retrieved", payload["questions_with_context"])
        c3.metric("Grounded Passes", payload["groundedness_passes"])
        c4.metric("Pipeline Errors", payload["pipeline_errors"])

        c5, c6, c7 = st.columns(3)
        c5.metric("Groundedness Scored", payload["groundedness_scored"])
        c6.metric("Groundedness Failures", payload["groundedness_failures"])
        c7.metric("Judge Errors / Unscored", payload["judge_errors_or_unscored"])

        rate = payload["groundedness_pass_rate_percent"]
        if rate is not None:
            st.metric("Groundedness pass rate (scored answers)", f"{rate:.2f}%")
        else:
            st.warning("No groundedness scores were available.")

        table = []
        for row in results:
            score = row["groundedness_score"]
            label = "PASS" if score is True else "FAIL" if score is False else "UNSCORED"
            table.append({
                "Question": row["question"],
                "Category": row["category"],
                "Retrieved": row["retrieved_count"],
                "Reranked": row["reranked_count"],
                "Context Found": row["context_found"],
                "Groundedness": label,
                "Reason": row["groundedness_reason"],
                "Pipeline Error": row["error"] or "",
                "Judge Error": row["judge_error"] or "",
            })

        st.subheader("Evaluation results")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.success(f"JSON saved to: {OUTPUT_FILE.resolve()}")
        st.download_button(
            "Download Full Evaluation JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name="rag_evaluation_results.json",
            mime="application/json",
        )

        for row in results:
            with st.expander(f"{row['category']} — {row['question']}"):
                st.markdown("**Reference answer**")
                st.write(row["expected_answer"])
                st.markdown("**Generated answer**")
                st.write(row["generated_answer"] or "No answer generated.")
                st.markdown("**Groundedness explanation**")
                st.write(row["groundedness_reason"] or "Not scored.")
                if row["error"]:
                    st.error(f"Pipeline error: {row['error']}")
                if row["judge_error"]:
                    st.error(f"Judge error: {row['judge_error']}")
                with st.expander("Retrieved policy context"):
                    st.write(row["retrieved_context"] or "No context retrieved.")


if __name__ == "__main__":
    main()
