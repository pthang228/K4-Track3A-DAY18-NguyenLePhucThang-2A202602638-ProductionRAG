# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Lê Phúc Thắng  
**MSSV:** 2A202602638  
**Khóa:** K4 - Track 3A  

---

## Cấu hình chạy

| Thành phần | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic` (paragraph, 500 chars) — 51 chunks | `chunk_hierarchical` (parent 2048 / child 256) — 104 children, 26 parents |
| Enrichment | — | `_enrich_single_call` (1 call/chunk): contextual prepend + HyQA questions + auto metadata |
| Search | Dense only (bge-m3 + Qdrant), top-3 | Hybrid: BM25 (underthesea) + Dense (bge-m3) → RRF (k=60), top-20 child |
| Rerank | — | Cross-encoder `BAAI/bge-reranker-v2-m3` trên 20 child → map sang top-3 **parent** khác nhau |
| Generation | deepseek-chat, prompt tối giản | deepseek-chat, temperature=0, prompt có quy tắc version / yes-no / tính toán |
| Judge RAGAS | deepseek-chat + embeddings bge-m3 local | (giống baseline) |

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8000 | **0.8594** | +0.0594 |
| Answer Relevancy | 0.7759 | **0.8578** | +0.0819 |
| Context Precision | 0.7750 | **0.8500** | +0.0750 |
| Context Recall | 0.8500 | **0.9250** | +0.0750 |

Cả 4 metrics ≥ 0.75, Faithfulness ≥ 0.85. (Số liệu từ `python main.py` — `reports/ragas_report.json`, `reports/naive_baseline_report.json`.)

> **Lưu ý về độ nhiễu của judge:** chạy `naive_baseline.py` riêng lẻ trước đó cho Faithfulness 0.7567 / Answer Relevancy 0.7666 / Context Precision 0.7000 với cùng pipeline — dao động ±0.05 giữa 2 lần chạy do judge LLM. Khi so sánh cần chạy nhiều lần hoặc so trên cùng 1 lần chạy (như bảng trên).

## Latency Breakdown (CPU only, không GPU)

| Bước | avg | p50 | max |
|---|---|---|---|
| Hybrid search (BM25 + dense encode query + Qdrant) | 342 ms | 296 ms | 864 ms |
| Rerank 20 candidates (bge-reranker-v2-m3) | **11 156 ms** | 11 433 ms | 13 547 ms |
| LLM generation (deepseek-chat) | 2 078 ms | 2 077 ms | 3 282 ms |
| **Tổng / query** | **13 576 ms** | 13 532 ms | 16 005 ms |

| Bước build (1 lần) | Thời gian |
|---|---|
| Chunking | 0.07 s |
| Enrichment (104 chunks, 8 luồng song song; lần chạy này đọc từ cache) | 0.04 s (lần đầu: 23.9 s) |
| Indexing (BM25 + bge-m3 encode 104 chunks + Qdrant upsert) | 44.7 s |
| Load reranker | 7.8 s |
| RAGAS eval (20 câu × 4 metrics) | 44.2 s |

**Bottleneck:** reranker chiếm ~82% latency/query vì chạy cross-encoder 568M params trên CPU với 20 cặp (query, child đã enrich + HyQA questions ≈ 400–600 ký tự).

## Bottom-5 Failures

Xếp theo điểm trung bình 4 metrics (từ `failures` trong `reports/ragas_report.json`).

### #1 — Multi-hop: thiếu tài liệu lương (avg 0.35)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024). Lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** 18 ngày phép (đúng) + "Về mức lương … **Không tìm thấy thông tin.**"
- **Scores:** faithfulness 0.89 · answer_relevancy **0.00** · context_precision **0.00** · context_recall 0.50
- **Worst metric:** answer_relevancy (RAGAS đánh dấu câu trả lời "noncommittal" → 0)
- **Error Tree:** Output sai (thiếu nửa sau) → Context đúng? **KHÔNG** — 3 context đều là tài liệu nghỉ phép (v2024, v2023, nghỉ không lương), không có `bang_luong_2024.md` → Query OK? **KHÔNG** — 1 query chứa 2 ý (phép + lương); phần "nghỉ phép / thâm niên" áp đảo cả BM25 lẫn dense nên các child của bảng lương bị đẩy khỏi top.
- **Root cause:** Retrieval cho câu hỏi multi-hop dùng 1 query duy nhất; top-3 parent bị 1 chủ đề chiếm hết.
- **Suggested fix:** Query decomposition (LLM tách thành "số ngày phép thâm niên 9 năm" + "khoảng lương Senior") → retrieve từng sub-query rồi gộp; hoặc diversity (MMR theo `category`) khi chọn top-k parent.

### #2 — Tính toán sai đơn vị thời gian (avg 0.67)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng × 15tr = 300.000 VNĐ/**tháng** → pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** "15.000.000 × 2% = **300.000 VNĐ**" (hiểu là tổng phạt).
- **Scores:** faithfulness **0.33** · answer_relevancy 0.84 · context_precision 1.00 · context_recall 0.50
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? **CÓ** — `tam_ung.md` xếp hạng 1 (precision 1.0), có đủ "15 ngày" và "2%/tháng" → Query OK? **CÓ** → lỗi ở **generation**: LLM bỏ qua đơn vị "/tháng", không pro-rata theo số ngày quá hạn.
- **Root cause:** Suy luận số học nhiều bước (quá hạn = 20 − 15; phí theo tháng → theo ngày) trong 1 lần sinh; claim "300.000 VNĐ" không suy ra trực tiếp từ context nên judge chấm unfaithful. Context recall 0.5 vì ground truth chứa phép tính pro-rata không có nguyên văn trong tài liệu.
- **Suggested fix:** Prompt yêu cầu liệt kê từng bước (số ngày quá hạn → mức phí theo đơn vị gốc → quy đổi); hoặc tool/calculator cho câu hỏi numeric; thêm few-shot cho dạng "phí theo tháng nhưng quá hạn theo ngày".

### #3 — Judge false-negative: câu trả lời chép từ context (avg 0.71)
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ≥ 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả **25.000.000 VNĐ** (100% chi phí đào tạo đã được tài trợ), vì nghỉ việc sau 8 tháng chưa đủ cam kết làm việc ít nhất 1 năm…" — **đúng hoàn toàn**.
- **Scores:** faithfulness **0.25** · answer_relevancy 0.95 · context_precision 1.00 · context_recall 1.00
- **Worst metric:** faithfulness
- **Error Tree:** Output sai? **KHÔNG** (đúng với ground truth) → Context đúng? **CÓ** (precision & recall = 1.0) → lỗi nằm ở **evaluation**, không phải pipeline.
- **Root cause:** Faithfulness của RAGAS 0.1 tách answer thành statements rồi chạy NLI bằng judge LLM; với tiếng Việt + phép suy luận ("8 tháng < 1 năm ⇒ 100%"), judge (deepseek-chat) đánh dấu các statement suy luận là "không có trong context" dù logic suy ra trực tiếp từ điều khoản.
- **Suggested fix:** Không sửa pipeline. Cải thiện đánh giá: chạy RAGAS nhiều lần lấy trung bình, dùng judge mạnh hơn, hoặc review thủ công bottom-N trước khi kết luận; với answer nên trích nguyên văn điều khoản trước rồi mới kết luận để judge dễ đối chiếu.

### #4 — Judge false-negative trên câu negation (avg 0.83)
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Chưa được hưởng PVI, chỉ được tham gia BHXH bắt buộc.
- **Got:** "Không. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI, chỉ được tham gia bảo hiểm xã hội bắt buộc." — gần như **nguyên văn** câu trong `thu_viec.md`.
- **Scores:** faithfulness **0.33** · answer_relevancy 0.98 · context_precision 1.00 · context_recall 1.00
- **Worst metric:** faithfulness
- **Error Tree:** Output sai? **KHÔNG** → Context đúng? **CÓ** (`thu_viec.md` rank 1, `bao_hiem_suc_khoe.md` rank 2) → lỗi ở **evaluation** (judge NLI xử lý kém câu phủ định tiếng Việt "chưa được…").
- **Root cause:** Giống #3 — nhiễu của judge với negation. Context thứ 2 (`bao_hiem_suc_khoe.md` — "gói bảo hiểm … cho tất cả nhân viên chính thức") chỉ nói về nhân viên chính thức, không nhắc thử việc, có thể khiến judge không ghép được với câu trong `thu_viec.md`.
- **Suggested fix:** Như #3; thêm test case negation vào bộ đánh giá thủ công; cân nhắc metric bổ sung (answer correctness so với ground truth).

### #5 — Thiếu phiên bản cũ cho câu hỏi version (avg 0.80)
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có — v2.0 bắt buộc MFA cho email, VPN, hệ thống nội bộ. Chính sách cũ v1.0 **không** yêu cầu MFA.
- **Got:** "Có. Tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Scores:** faithfulness 1.00 · answer_relevancy 0.70 · context_precision 1.00 · context_recall **0.50**
- **Worst metric:** context_recall
- **Error Tree:** Output đúng nhưng thiếu ý → Context đúng? **MỘT PHẦN** — có `mat_khau_v2.md` (rank 1) nhưng thiếu `mat_khau_v1.md`; slot 2–3 bị `mua_sam.md`, `vpn_truy_cap.md` chiếm (do từ khóa "VPN") → Query OK? Có, nhưng query không nhắc tới "phiên bản cũ" nên v1 (không có chữ MFA) bị chấm thấp ở cả BM25, dense và reranker.
- **Root cause:** Tài liệu phiên bản cũ không chứa từ khóa của câu hỏi → không retrieve được bằng similarity; ground truth lại yêu cầu so sánh 2 phiên bản.
- **Suggested fix:** Version-aware retrieval: metadata `policy_family` + `version` + `superseded_by`; khi retrieve được 1 version thì kéo kèm các version cùng family (document linking).

### Pattern phụ: Context Precision 0.5 ở các câu version
Câu 4, 5, 7 (phép năm, thâm niên, đổi mật khẩu) có precision 0.5 vì reranker xếp **v2023/v1.0 (cũ) lên rank 1**, v2024/v2.0 (hiện hành) rank 2 — hai bản gần như trùng nội dung nên cross-encoder không phân biệt được. Answer vẫn đúng nhờ prompt "ưu tiên phiên bản hiện hành". Fix: boost theo `effective_date`/`version` sau rerank, hoặc filter `status != superseded` rồi mới thêm bản cũ làm context phụ.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Một nửa: 18 ngày phép đúng, phần lương trả "Không tìm thấy thông tin" (answer_relevancy = 0 vì noncommittal).
2. **Context đúng?** → Không: top-3 parent = nghỉ phép v2024, nghỉ phép v2023, nghỉ phép không lương. `bang_luong_2024.md` (có dòng `Senior (P3-P4) | 20.000.000 - 35.000.000`) không lọt top-3 → context_precision 0, context_recall 0.5.
3. **Query rewrite OK?** → Không có bước rewrite: 1 câu hỏi 2 ý gửi thẳng vào hybrid search; tín hiệu "nghỉ phép / thâm niên / ngày" lấn át "lương / Senior".
4. **Fix ở bước:** **Query processing (trước retrieval)** — query decomposition thành 2 sub-query, retrieve riêng, gộp bằng RRF, rerank theo từng sub-query, đảm bảo mỗi sub-query có ít nhất 1 parent trong context. Phòng thủ thêm ở bước chọn context: MMR/diversity theo `category` để top-3 không bị 1 chủ đề chiếm hết. Generation đã làm đúng (không bịa lương khi thiếu context) — giữ nguyên.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (#1) + diversity khi chọn parent.
- Version-aware metadata (`policy_family`, `version`, `superseded`) → boost bản hiện hành lên rank 1 (precision 0.5 → 1.0 cho 3 câu) và kéo kèm bản cũ (#5).
- Prompt từng bước cho câu numeric (#2).
- Giảm latency rerank: rerank trên text child gốc (không kèm HyQA questions), giảm candidates 20 → 10, hoặc dùng bản ONNX/quantized; kỳ vọng ~11 s → ~3 s/query trên CPU.
- Chạy RAGAS 3 lần lấy trung bình để tách nhiễu judge (#3, #4) khỏi lỗi thật.
