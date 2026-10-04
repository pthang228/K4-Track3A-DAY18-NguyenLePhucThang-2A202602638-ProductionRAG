# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Lê Phúc Thắng  
**MSSV:** 2A202602638  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trên toàn corpus (26 tài liệu, ~21K ký tự), threshold 0.85 + `all-MiniLM-L6-v2` tạo **208 chunks** (avg 99 ký tự) so với basic **51 chunks** (avg 410). MiniLM là model tiếng Anh nên similarity giữa các câu tiếng Việt thấp → threshold 0.85 cắt quá vụn (gần như 1 câu/chunk). Với tiếng Việt cần model đa ngôn ngữ (bge-m3) hoặc hạ threshold ~0.5–0.6. |
| Hierarchical (parent-child) chunking | M1 | `chunk_hierarchical()` + `retrieve()` trong `pipeline.py` | 104 child (≤256 ký tự) / 26 parent. Retrieve trên child (chính xác), trả **parent** cho LLM (đủ ngữ cảnh) — dedupe theo `source::parent_id` để top-3 là 3 tài liệu khác nhau. Context recall 0.85 → **0.925**. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks theo header `#`/`##`/`###`, metadata `section` giữ tên mục. Phù hợp corpus markdown có cấu trúc đều (mỗi chính sách 3–4 mục). |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") → phải `replace("_", " ")` thì query "nghỉ phép" mới khớp. RRF (k=60) cộng `1/(k+rank+1)` từ 2 danh sách nên không cần chuẩn hoá thang điểm BM25 (0–20) với cosine (0–1). BM25 bắt được từ khóa chính xác (số tiền, "MFA", "PVI") mà dense bỏ sót. Search chỉ ~342 ms/query. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 tách rõ: "nghỉ 12 ngày/năm" 0.991 vs "mật khẩu 90 ngày" 0.0007. Nhưng trên CPU: **~11.2 s/query** cho 20 candidates (82% latency). Context precision 0.775 → **0.85**. Điểm yếu: 2 phiên bản chính sách gần trùng nội dung → reranker hay xếp bản cũ lên trên. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.859 · AR 0.858 · CP 0.850 · CR 0.925 (baseline 0.800 · 0.776 · 0.775 · 0.850). Faithfulness có nhiễu judge: 2/5 câu bottom là câu trả lời **đúng** nhưng bị chấm 0.25–0.33 (NLI tiếng Việt + suy luận). Answer relevancy = 0 khi answer có "Không tìm thấy" (noncommittal). |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` (combined, 1 call/chunk) | 1 prompt JSON trả về `context` (prepend), `questions` (HyQA — nối vào text để index), `summary`, `metadata`. 104 chunks / 23.9 s nhờ 8 luồng song song + cache ra đĩa (chạy lại = 0 call). HyQA giúp nối khoảng cách từ vựng giữa câu hỏi người dùng ("bị phạt bao nhiêu") và văn bản chính sách ("tính phí 2%/tháng"). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Tạo lại venv khi đang activate**
- **Lỗi:** `Error: [Errno 13] Permission denied: '...\K4-Track3A-Day18-Production-RAG\.venv\Scripts\python.exe'`
- **Nguyên nhân & debug:** Prompt đã có `(.venv)` → venv đang chạy; `python -m venv .venv` cố ghi đè chính `python.exe` đang bị Windows khóa. Venv đã tồn tại sẵn nên chỉ cần bỏ qua bước tạo; muốn tạo lại phải `deactivate` trước.

**2. RAGAS trả về toàn 0 và treo ~10 phút**
- **Lỗi:** `openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-....'}}`
- **Nguyên nhân & debug:** `evaluate_ragas()` bọc try/except và RAGAS mặc định `raise_exceptions=False` → lỗi bị nuốt, mọi score = NaN → 0, còn retry làm process treo. Debug bằng cách chạy `evaluate(..., raise_exceptions=True)` trên 1 mẫu → thấy 401 → `.env` vẫn là placeholder `sk-...`. Bài học: khi metric = 0 đồng loạt, nghi ngờ hạ tầng (key, network) trước khi nghi ngờ pipeline.

**3. Dùng DeepSeek thay OpenAI**
- **Lỗi:** `openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid n value (currently only n = 1 is supported)'}}`
- **Nguyên nhân & debug:** `answer_relevancy` sinh 3 câu hỏi ngược (`strictness=3`) bằng 1 request `n=3`; RAGAS chỉ gửi `n` khi LLM nằm trong `MULTIPLE_COMPLETION_SUPPORTED` (có `ChatOpenAI`). Đọc source `ragas/llms/base.py` → thấy nhánh fallback gửi `n` prompt riêng lẻ → loại `ChatOpenAI` khỏi danh sách khi dùng provider khác. DeepSeek cũng không có embeddings API → dùng bge-m3 local (chung instance với M2) qua 1 class `Embeddings` của LangChain. Thêm `OPENAI_BASE_URL`, `LLM_MODEL` vào `config.py` thay vì hard-code `gpt-4o-mini`.

**4. Tải model reranker 2.2 GB bị treo**
- **Hiện tượng:** file `.incomplete` dừng ở 789 MB rồi 134 MB qua nhiều lần thử (cả có/không Xet), tốc độ CDN dao động 86 KB/s – 4 MB/s.
- **Cách giải quyết:** tải bằng `curl -C - --speed-limit 100000 --speed-time 20` trong vòng retry (tự resume khi tụt tốc), đối chiếu `sha256` với `X-Linked-ETag` của Hugging Face, rồi trỏ `RERANKER_MODEL` tới thư mục local.

**5. In tiếng Việt trên console Windows**
- **Lỗi:** `UnicodeEncodeError: 'charmap' codec can't encode character 'ộ'`
- **Cách giải quyết:** console mặc định cp1252 → chạy với `PYTHONIOENCODING=utf-8` (các module trong lab đã `sys.stdout.reconfigure(encoding="utf-8")`).

**Kiến thức còn thiếu & cách bổ sung:**
- Cách RAGAS tính từng metric (statement extraction + NLI cho faithfulness, average precision cho context precision) → đọc source `ragas/metrics/*.py` thay vì chỉ đọc docs; nhờ đó hiểu vì sao câu trả lời đúng vẫn bị chấm thấp và vì sao "Không tìm thấy" làm answer_relevancy = 0.
- Đặc thù tiếng Việt cho retrieval (word segmentation, model embedding đa ngôn ngữ) → so sánh thực nghiệm MiniLM vs bge-m3 trên chính corpus.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách & quy trình nội bộ (HR / IT / Tài chính) bằng tiếng Việt

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG — chunk theo paragraph (~500 ký tự) → embedding bge-m3 → dense search top-3 trên Qdrant → LLM trả lời với prompt "chỉ dựa trên context". Đo trên bộ 20 câu của lab: Faithfulness 0.80 · Answer Relevancy 0.78 · Context Precision 0.78 · Context Recall 0.85.
- **Vấn đề / Bottlenecks:**
  - Tài liệu có nhiều phiên bản (v2023/v2024, mật khẩu v1/v2) → dễ trả lời theo bản cũ hoặc thiếu so sánh bản cũ/mới.
  - Câu hỏi chứa số tiền, mã, từ viết tắt (PVI, MFA, P3-P4) → dense search hay bỏ sót.
  - Câu hỏi multi-hop (phép năm + bảng lương) và câu tính toán (phí 2%/tháng cho 5 ngày quá hạn) trả lời thiếu/sai.
  - 2 PDF scan (BCTC, Nghị định 13/2023) không có text layer → hiện bị bỏ qua hoàn toàn.
  - Chưa có quy trình đo chất lượng mỗi khi đổi prompt/chunking.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical parent-child (child ~256 ký tự để retrieve, parent = section/tài liệu để đưa LLM) — lab cho thấy tăng context recall 0.85 → 0.925. Với tài liệu markdown/có header: dùng structure-aware làm ranh giới parent.
2. **Search retrieval:** Hybrid BM25 (có word segmentation tiếng Việt) + dense bge-m3, gộp bằng RRF k=60 — BM25 bắt mã sản phẩm, số tiền, từ viết tắt mà dense bỏ sót; RRF không cần chuẩn hoá điểm. Thêm query decomposition cho câu hỏi multi-hop (failure #1 của lab).
3. **Reranking:** Có — bge-reranker-v2-m3 (đa ngôn ngữ, tốt cho tiếng Việt). Vì CPU mất ~11 s/query: chạy trên GPU hoặc bản ONNX/quantized, giảm candidates 20 → 10; nếu cần < 1 s thì thử reranker nhỏ hơn và đo lại precision.
4. **Evaluation:** RAGAS 4 metrics trên bộ test 30–50 câu chia theo loại (lookup, version, negation, multi-hop, numeric) + review thủ công bottom-5 mỗi lần chạy (judge có nhiễu ±0.05 và false-negative với tiếng Việt). Chạy eval trong CI mỗi khi đổi prompt/chunking.
5. **Enrichment:** Combined single-call (contextual prepend + HyQA + metadata) có cache theo hash nội dung → chi phí 1 lần/chunk. Bổ sung metadata `version`/`effective_date`/`superseded` để ưu tiên tài liệu hiện hành (failure #5 và pattern precision 0.5 của lab).

6. **Ingestion PDF scan:** OCR (VD: PaddleOCR / Tesseract `vie`) cho 2 PDF scan rồi đưa qua structure-aware chunking, để không mất nguồn tài liệu tài chính và pháp lý.

#### 3. Timeline triển khai
- **Tuần 1:** Mở rộng test set lên 30–50 câu (đủ 6 loại câu hỏi), chạy RAGAS 3 lần lấy trung bình làm baseline; dựng Qdrant + hybrid search (BM25 + dense + RRF).
- **Tuần 2:** Hierarchical chunking + combined enrichment có cache + OCR 2 PDF scan; so sánh RAGAS với baseline (mục tiêu: cả 4 metrics ≥ 0.85).
- **Tuần 3:** Thêm reranker (đo latency trên phần cứng thật), version metadata + boost bản hiện hành, query decomposition cho multi-hop.
- **Tuần 4:** Failure analysis bottom-N, tinh chỉnh prompt (numeric từng bước, negation), tích hợp eval vào CI, báo cáo latency breakdown.
