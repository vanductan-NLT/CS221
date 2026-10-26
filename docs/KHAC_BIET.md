# Khác biệt so với paper

Mọi chỗ nhóm làm khác paper hoặc khác code gốc đều ghi ở đây. Slide 6 lấy nội dung từ file này.

| # | Khác gì | Paper / code gốc | Nhóm | Vì sao | Ảnh hưởng dự kiến |
|---|---|---|---|---|---|
| 1 | Mô hình | 10 mô hình, gồm GPT-3 và 13B/20B | GPT-Neo 1.3B/2.7B, OPT 1.3B/2.7B (6B/6.7B nếu đủ GPU) | GPT-3 đã ngừng; 13B/20B vượt GPU miễn phí | Không so được hàng GPT-3, 13B, 20B |
| 2 | Contriever | Có | Không | Tác giả mất file kết quả truy xuất | Bảng 1 làm lại bằng BM25 |
| 3 | Độ chính xác số | fp32 (int8 cho 13B/20B) | fp16 trên GPU | Vừa bộ nhớ GPU miễn phí | Có thể lệch nhẹ |
| 4 | Sinh văn bản | Từng câu một | Theo batch, sắp theo độ dài | Nhanh hơn nhiều | Lý thuyết là không đổi kết quả; kiểm ở issue #11 |
| 5 | Ví dụ mẫu | Chọn ngẫu nhiên mỗi lần chạy | Cố định theo `id` câu hỏi | Chạy tiếp được, các cách dùng chung ví dụ | Lệch nhỏ do ví dụ khác |
| 6 | Ghép kết quả cho adaptive | Sắp xếp theo `question` | Theo `id` | 881 câu hỏi trùng văn bản | Sửa lỗi ghép nhầm |
| 7 | Ví dụ mẫu của BM25 | Code gốc với file BM25 hiện có: không kèm đoạn văn | Kèm đoạn văn | Chờ quyết định ở issue #12 | Chưa biết, đo ở issue #11 |
| 8 | Nguồn bốc ví dụ mẫu (kết quả đối chứng 26/10/2026) | Bốc 1 ví dụ cho mỗi loại quan hệ khác, từ cả 14.267 câu, bằng bộ sinh số ngẫu nhiên chung seed 633 | Bốc từ `POOL` bằng `random.Random("{SEED}-{id}")`; `POOL` cũng là cả 14.267 câu (đã kiểm: cùng file với `upstream/data/popQA.tsv`) | Cố định theo `id` để chạy tiếp được (dòng 5) | Trên 300 câu GPT-Neo 1.3B, vanilla của nhóm đúng nhiều hơn code gốc 9 câu (14,33% so với 11,33%; không câu nào code gốc đúng mà nhóm sai), chưa rõ nguyên nhân. Xem `docs/DOI_CHUNG_2026-10-26.md` |

Thêm dòng mới khi có thay đổi. Ghi rõ issue liên quan.
