from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict, is_dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        llm, embeddings = _ragas_models()
        result = evaluate(dataset,
                          metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=llm, embeddings=embeddings,
                          run_config=RunConfig(timeout=120, max_retries=3, max_workers=8))
        df = result.to_pandas()

        per_question = [
            EvalResult(question=questions[i], answer=answers[i], contexts=list(contexts[i]),
                       ground_truth=ground_truths[i],
                       **{m: _safe_float(row.get(m)) for m in METRICS})
            for i, (_, row) in enumerate(df.iterrows())
        ]
        # nanmean: câu nào RAGAS không chấm được (NaN) thì không kéo điểm trung bình về 0
        aggregate = {m: (round(float(df[m].mean(skipna=True)), 4) if m in df and df[m].notna().any()
                         else 0.0)
                     for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _ragas_models():
    """Judge LLM + embeddings cho RAGAS.

    - OpenAI (không set OPENAI_BASE_URL): trả (None, None) → RAGAS dùng default (gpt-4o-mini + ada-002).
    - Provider OpenAI-compatible khác (VD DeepSeek): judge = LLM_MODEL qua base_url; embeddings =
      bge-m3 local (DeepSeek không có embeddings API). DeepSeek chỉ hỗ trợ n=1 → bỏ ChatOpenAI khỏi
      danh sách "multiple completion" để RAGAS gửi n request riêng (answer_relevancy strictness=3).
    """
    if not OPENAI_BASE_URL:
        return None, None

    import ragas.llms.base as ragas_llm_base
    from langchain_openai import ChatOpenAI
    from langchain_core.embeddings import Embeddings

    ragas_llm_base.MULTIPLE_COMPLETION_SUPPORTED = [
        c for c in ragas_llm_base.MULTIPLE_COMPLETION_SUPPORTED if c is not ChatOpenAI]

    class LocalBGEEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            from src.m2_search import _get_shared_encoder
            return _get_shared_encoder().encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    llm = ChatOpenAI(model=LLM_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, temperature=0)
    return llm, LocalBGEEmbeddings()


def _safe_float(value) -> float:
    """NaN/None → 0.0 (RAGAS trả NaN khi không parse được output của judge LLM)."""
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


# Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → hướng sửa
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating — answer chứa claim không có trong context",
                     "Tighten prompt (chỉ dùng context), temperature=0, yêu cầu trích dẫn nguồn"),
    "context_recall": ("Missing relevant chunks — retriever bỏ sót thông tin cần cho ground truth",
                       "Improve chunking (parent-child), tăng top-k, thêm BM25/HyQA cho query multi-hop"),
    "context_precision": ("Too many irrelevant chunks — chunk liên quan bị xếp sau chunk nhiễu",
                          "Add/tune reranking, metadata filter (version, category), giảm top-k"),
    "answer_relevancy": ("Answer doesn't match question — trả lời lan man hoặc né câu hỏi",
                         "Improve prompt template: trả lời trực tiếp câu hỏi trước, ngắn gọn"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []
    for r in eval_results:
        scores = {m: _safe_float(getattr(r, m)) for m in METRICS}
        avg = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "scores": {m: round(s, 4) for m, s in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json",
                extra: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn; bổ sung per_question + extra)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "failures": failures,
        "per_question": [asdict(r) if is_dataclass(r) else r for r in per_question],
        **(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
