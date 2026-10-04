from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, LLM_MODEL

# Latency từng bước build (giây) — ghi vào report (bonus: latency breakdown)
BUILD_LATENCY: dict[str, float] = {}

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu chính sách nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context được cung cấp.
Quy tắc:
- Trả lời trực tiếp vào câu hỏi ngay ở câu đầu tiên, ngắn gọn (1-4 câu), bằng tiếng Việt.
- Giữ nguyên chính xác con số, đơn vị, chức danh người phê duyệt như trong context.
- Nếu context có nhiều phiên bản của cùng một chính sách, trả lời theo phiên bản hiện hành (mới nhất) và nói rõ phiên bản cũ đã bị thay thế cùng giá trị cũ.
- Câu hỏi cần tính toán: nêu phép tính dựa trên số liệu trong context.
- Câu hỏi có/không: bắt đầu bằng "Có" hoặc "Không", sau đó giải thích theo context.
- KHÔNG thêm thông tin không có trong context. Nếu context không có thông tin → trả lời "Không tìm thấy thông tin."."""


def _parent_key(source: str, parent_id: str) -> str:
    # parent_id chỉ unique trong 1 document → ghép với source
    return f"{source}::{parent_id}"


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1) — hierarchical: index child, trả về parent
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_store: dict[str, str] = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        source = doc["metadata"]["source"]
        for p in parents:
            parent_store[_parent_key(source, p.metadata["parent_id"])] = p.text
        for child in children:
            all_chunks.append({"text": child.text,
                               "metadata": {**child.metadata, "parent_id": child.parent_id,
                                            "parent_key": _parent_key(source, child.parent_id)}})
    BUILD_LATENCY["chunking_s"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_store)} parents "
          f"from {len(docs)} documents ({BUILD_LATENCY['chunking_s']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5) — combined mode: context prepend + HyQA questions
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        indexed = []
        for e, raw in zip(enriched, all_chunks):
            text = e.enriched_text
            if e.hypothesis_questions:
                text += "\n\nCâu hỏi liên quan: " + " | ".join(e.hypothesis_questions)
            indexed.append({"text": text, "metadata": {**e.auto_metadata, **raw["metadata"]}})
        all_chunks = indexed
        BUILD_LATENCY["enrichment_s"] = round(time.time() - t0, 2)
        print(f"  ✓ Enriched {len(enriched)} chunks ({BUILD_LATENCY['enrichment_s']:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_store = parent_store
    BUILD_LATENCY["indexing_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({BUILD_LATENCY['indexing_s']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    BUILD_LATENCY["reranker_load_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({BUILD_LATENCY['reranker_load_s']:.1f}s)", flush=True)

    return search, reranker


def retrieve(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
             top_k: int = RERANK_TOP_K, timings: dict | None = None) -> list[str]:
    """Hybrid search child chunks (top-20) → rerank → map sang top-k parent khác nhau."""
    t0 = time.perf_counter()
    results = search.search(query)
    t1 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    t2 = time.perf_counter()
    if timings is not None:
        timings["search_ms"] = (t1 - t0) * 1000
        timings["rerank_ms"] = (t2 - t1) * 1000

    ranked = reranked if reranked else results
    parent_store = getattr(search, "parent_store", {})
    contexts, seen = [], set()
    for r in ranked:
        key = r.metadata.get("parent_key")
        text = parent_store.get(key, r.text)
        if (key or text) in seen:
            continue
        seen.add(key or text)
        contexts.append(text)
        if len(contexts) == top_k:
            break
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = timings if timings is not None else {}
    contexts = retrieve(query, search, reranker, timings=timings)

    from config import OPENAI_API_KEY
    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["llm_ms"] = (time.perf_counter() - t0) * 1000
    return answer, contexts


def _latency_summary(per_query: list[dict]) -> dict:
    summary = {}
    for step in ["search_ms", "rerank_ms", "llm_ms"]:
        values = sorted(t.get(step, 0.0) for t in per_query)
        if values:
            summary[step] = {"avg": round(sum(values) / len(values), 1),
                             "p50": round(values[len(values) // 2], 1),
                             "max": round(values[-1], 1)}
    totals = sorted(sum(t.values()) for t in per_query)
    if totals:
        summary["total_ms"] = {"avg": round(sum(totals) / len(totals), 1),
                               "p50": round(totals[len(totals) // 2], 1),
                               "max": round(totals[-1], 1)}
    return summary


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []

    for i, item in enumerate(test_set):
        timings: dict = {}
        answer, contexts = run_query(item["question"], search, reranker, timings=timings)
        query_timings.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}... "
              f"({sum(timings.values()):.0f}ms)", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    ragas_s = round(time.time() - t0, 2)
    print(f"  ✓ RAGAS done ({ragas_s:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    latency = {"build_s": {**BUILD_LATENCY, "ragas_eval_s": ragas_s},
               "per_query_ms": _latency_summary(query_timings)}
    print("\nLATENCY BREAKDOWN (per query, ms)")
    print(f"  {'Step':<12} {'avg':>8} {'p50':>8} {'max':>8}")
    for step, s in latency["per_query_ms"].items():
        print(f"  {step:<12} {s['avg']:>8.1f} {s['p50']:>8.1f} {s['max']:>8.1f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, extra={"latency": latency})
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
