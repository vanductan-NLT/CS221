# Kế hoạch đồ án CS221: PopQA và Adaptive Retrieval

Cập nhật: 25/10/2026 · Bản chính (cùng chỉnh, bình luận): https://claude.ai/artifact/NDqXEvuWYUXPZnS73jK1Fq · Bản vẽ trực quan: https://claude.ai/artifact/19SG5F1iWGcjD5Ra1zFLLJ

## Tóm tắt

Nhóm 4 người chạy lại paper "When Not to Trust Language Models" (ACL 2023, Long Paper, https://aclanthology.org/2023.acl-long.546/) trên PopQA, rồi mở rộng bằng mô hình đời mới. Mục tiêu: xong mọi phần trước Chủ Nhật 15/11, còn 1 tuần dự phòng trước buổi trình bày (khoảng cuối tháng 11).

- Câu hỏi của paper: khi nào nên tin trí nhớ của mô hình ngôn ngữ, khi nào nên tra cứu thêm Wikipedia?
- Ý chính: chỉ tra cứu khi câu hỏi về thực thể ít nổi tiếng (Adaptive Retrieval) vừa đúng hơn vừa rẻ hơn.
- Đã có: cô duyệt paper; code tác giả https://github.com/AlexTMallen/adaptive-retrieval; notebook `adaptive_retrieval_repro.ipynb` + dataset BM25 trong thư mục này (chưa chạy trên GPU thật).
- GPU: Kaggle + Colab miễn phí, mỗi người một tài khoản.
- Trình bày 15 phút: 60% trình bày + chạy lại, 30% mở rộng, 10% phân tích.
- Còn mở: ngày trình bày chính xác; cô trả lời về phạm vi tái lập.

## Phạm vi tái lập

| Ưu tiên | Mô hình | Cách chạy | Cho thấy gì |
|---|---|---|---|
| P0 (bắt buộc) | GPT-Neo 1.3B, OPT 1.3B | vanilla, BM25, rồi tính adaptive | Đủ để kiểm tra 4 kết luận |
| P1 | GPT-Neo 2.7B, OPT 2.7B; GenRead cho 2 model 1.3B | vanilla, BM25, adaptive; GenRead | Hiệu ứng khi mô hình to lên |
| P2 (nếu còn giờ GPU) | GPT-J 6B, OPT 6.7B (8-bit) | vanilla, BM25, adaptive | Gần với Hình 7 của paper |

Không chạy lại được (nói rõ khi trình bày):

| Phần | Vì sao | Làm gì thay |
|---|---|---|
| GPT-3 davinci-002/003 | OpenAI đã ngừng từ đầu 2024 | Trích số của paper |
| Contriever | Tác giả mất file kết quả (issue #5) | Dùng BM25; làm lại Bảng 1 bằng BM25 |
| OPT 13B, GPT-NeoX 20B | Cần GPU 32 GB | Ngoài phạm vi |
| EntityQuestions | Ưu tiên PopQA trước | Để dành nếu dư thời gian |

- Số câu: đủ 14.267 nếu tốc độ cho phép; nếu không, mọi lần chạy dùng chung một tập con cố định ≥3.000 câu. Chốt sau smoke test.
- Đối chứng: chạy `run_model.py` gốc với GPT-Neo 1.3B trên 300 câu, so với notebook trên cùng 300 câu.
- Số của paper nằm trong biểu đồ (Hình 7, 9, 13): đọc bằng WebPlotDigitizer, ghi là số xấp xỉ.

## Ba phần chấm điểm

4 kết luận cần kiểm tra:
1. Thực thể càng nổi tiếng thì mô hình càng hay trả lời đúng.
2. Mô hình to hơn chủ yếu giỏi hơn ở câu nổi tiếng.
3. Retrieval giúp câu ít nổi tiếng, có thể hại câu nổi tiếng.
4. Adaptive tốt bằng hoặc hơn cả hai cách, và tra cứu ít lần hơn.

| Phần | Làm gì | Xong khi có |
|---|---|---|
| 60% | Bài toán, vì sao, phương pháp, kết quả chạy lại | Bảng paper vs nhóm; hình theo độ nổi tiếng; 4 kết luận đúng/sai |
| 30% | A (chính): Qwen2.5 base 0.5B → 7B × vanilla, BM25, adaptive. Thêm B hoặc C (chọn ngày 8/11) | Bảng so sánh mới; 4 kết luận còn đúng không |
| 10% | 4 phân tích trên kết quả đã lưu | 3–4 phát hiện, mỗi cái 1 con số + 1 ví dụ |

- B: xếp hạng lại 5 đoạn BM25 bằng mô hình embedding.
- C: adaptive theo độ tự tin của model thay cho độ nổi tiếng (cần sửa notebook để lưu xác suất).
- Phân tích: (1) chấm chặt hơn, bỏ cách chấm "chuỗi con"; (2) theo 16 loại quan hệ; (3) retrieval gây hại, ví dụ thật; (4) đoạn văn có đáp án mà model vẫn sai.
- Lưu ý: model mới có thể đã học thuộc PopQA.

## Phân công

| Người | Phụ trách | Trình bày |
|---|---|---|
| Tân | Pipeline: smoke test, đối chứng code gốc, adaptive, gộp bảng và hình | Mở đầu, phương pháp, kết luận (3,5') |
| TV2 | GPT-Neo 1.3B, 2.7B (GPT-J 6B nếu P2), GenRead | Bài toán, vì sao, PopQA (3') |
| TV3 | OPT 1.3B, 2.7B (6.7B nếu P2); đọc số từ hình paper | Kết quả tái lập (2,5') |
| TV4 | Qwen2.5 + hướng B hoặc C | Mở rộng (4,5') |
| Cả nhóm | Mỗi người 1 phân tích 10% | Phân tích (1,5') |

## Lịch 3 tuần

Tuần 1 (26/10–1/11): chạy được và chốt phạm vi
- [ ] Tân nhắn cô hỏi phạm vi (26/10)
- [ ] Cả nhóm tạo + xác minh tài khoản Kaggle
- [ ] Tân: upload dataset BM25, smoke test (Qwen2.5-0.5B, 300 câu), ghi tốc độ
- [ ] Tân: chạy `run_model.py` gốc 300 câu, so với notebook
- [ ] Cả nhóm: đọc paper mục 1–2, Hình 1, 4, 7, 9; học 4 bài nền (LM + few-shot, BM25, RAG, chấm điểm)
- [ ] TV3: đọc số từ Hình 7, 9, 13
- [ ] CN 1/11: họp chốt số câu, model, ai chạy gì; bắt đầu P0

Tuần 2 (2/11–8/11): chạy lại, bắt đầu mở rộng
- [ ] TV2, TV3: P0 rồi P1
- [ ] Tân: adaptive, bảng paper vs nhóm, hình theo độ nổi tiếng
- [ ] TV4: Qwen2.5 0.5B, 1.5B, 3B × vanilla, BM25
- [ ] Th5 5/11: bảng P0 đủ
- [ ] CN 8/11: bảng tái lập đủ; chọn B hay C

Tuần 3 (9/11–15/11): mở rộng, phân tích, slide
- [ ] TV4 xong hướng A + hướng phụ
- [ ] Mỗi người xong 1 phân tích
- [ ] TV2, TV3: P2 nếu còn giờ GPU
- [ ] T4 11/11: nộp slide phần mình · T7 14/11: ghép, bấm giờ · CN 15/11: tập lần 1

Dự phòng (16/11 → ngày trình bày): sửa slide, chạy bù, tập thêm 1–2 lần.

## Slide 15 phút

| Slide | Nội dung | Phút | Người nói |
|---|---|---|---|
| 1 | Tên đề tài, giới thiệu nhóm | 0,25 | Tân |
| 2–4 | Bài toán, vì sao, PopQA | 3 | TV2 |
| 5–6 | Phương pháp; thiết lập tái lập, lỗi code gốc | 3 | Tân |
| 7–8 | Kết quả tái lập, 4 kết luận | 2,5 | TV3 |
| 9–11 | Mở rộng | 4,5 | TV4 |
| 12 | Phân tích | 1,5 | 1 người nói |
| 13 | Kết luận, hạn chế | 0,25 | Tân |

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| GPU miễn phí chạy quá chậm | Tập con cố định ≥3.000 câu; bỏ P2; GenRead chỉ 1–2 model |
| Notebook lỗi trên GPU thật | Phát hiện ở smoke test tuần 1; thử fp32 hoặc 4-bit theo HUONG_DAN.md |
| Phiên bị ngắt | Notebook tự chạy tiếp; Kaggle "Save & Run All" |
| Hết VRAM với 6B/6.7B | 8-bit hoặc bỏ (P2) |
| Số lệch paper > ~5 điểm | Kiểm tra bằng đối chứng code gốc; ghi nguyên nhân vào slide 6 |
| Cô không đồng ý phạm vi | Điều chỉnh ngay tuần 1; có thể thêm EntityQuestions |
| Thành viên trễ việc | Mốc Th5 + CN; người khác chạy thay |

## Tin nhắn gửi cô

> Em chào cô, nhóm em chọn paper "When Not to Trust Language Models" (ACL 2023, Long Paper). Paper có nhiều thực nghiệm (10 mô hình × 4 cách × 2 bộ dữ liệu), nên nhóm em đề xuất tái lập trên PopQA với GPT-Neo 1.3B/2.7B và OPT 1.3B/2.7B (thêm 6B/6.7B nếu đủ GPU) × 3 cách (không truy xuất, BM25, GenRead), cùng phương pháp Adaptive Retrieval. Nhóm em không tái lập được phần GPT-3 (OpenAI đã ngừng model), Contriever (tác giả đã mất file kết quả truy xuất) và các mô hình 13B/20B (vượt quá GPU miễn phí). Cô cho em hỏi phạm vi như vậy có được tính là chạy lại thực nghiệm không ạ? Em cảm ơn cô.
