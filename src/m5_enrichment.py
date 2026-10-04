from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json, hashlib, threading
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL, CACHE_DIR


# ─── Shared helpers ──────────────────────────────────────

_CLIENT = None


def _chat(system: str, user: str, max_tokens: int = 200, json_mode: bool = False) -> str:
    """Gọi OpenAI chat 1 lần. Trả về "" nếu không có API key hoặc lỗi → caller dùng fallback."""
    global _CLIENT
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
        if _CLIENT is None:
            _CLIENT = OpenAI()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = _CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return ""


# Cache enrichment ra đĩa: chạy lại pipeline không tốn thêm API call
_CACHE_PATH = os.path.join(CACHE_DIR, "enrichment_cache.json")
_CACHE_LOCK = threading.Lock()
_CACHE: dict | None = None


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _CACHE = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _CACHE = {}
    return _CACHE


def _save_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


_COMBINED_PROMPT = """Bạn làm giàu (enrich) 1 đoạn văn trích từ tài liệu nội bộ công ty để phục vụ tìm kiếm.
Trả về JSON đúng schema:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên con số",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn thuộc tài liệu nào (tên chính sách, phiên bản nếu có) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Câu hỏi là những câu nhân viên có thể hỏi mà đoạn văn trả lời được. Viết bằng tiếng Việt."""


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    summary = _chat(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Giữ nguyên các con số.",
        text, max_tokens=150)
    if summary:
        return summary

    # Extractive fallback (không cần API): 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
        text, max_tokens=200)
    if raw:
        questions = [q.strip().lstrip("0123456789.-) ").strip() for q in raw.split("\n")]
        questions = [q for q in questions if q]
        if questions:
            return questions[:n_questions]

    # Extractive fallback: biến câu khẳng định thành câu hỏi
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu tiếng Việt.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
    if context:
        return f"{context}\n\n{text}"

    # Simple fallback
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    raw = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150, json_mode=True)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Metadata JSON parse failed: {e}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    key = hashlib.sha1(f"{LLM_MODEL}|{source}|{text}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    raw = _chat(_COMBINED_PROMPT, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                max_tokens=500, json_mode=True)
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Enrichment JSON parse failed: {e}")
        return {}
    with _CACHE_LOCK:
        cache[key] = result
        _save_cache(cache)
    return result


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    # Combined mode: chạy song song các API call (I/O-bound), kết quả vẫn giữ đúng thứ tự
    combined_results = []
    if use_combined:
        with ThreadPoolExecutor(max_workers=8) as pool:
            combined_results = list(pool.map(
                lambda c: _enrich_single_call(c["text"], c.get("metadata", {}).get("source", "")),
                chunks))

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = combined_results[i]
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
