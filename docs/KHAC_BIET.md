# Khác biệt so với paper

Mọi chỗ nhóm làm khác paper hoặc khác code gốc đều ghi ở đây. Slide 6 lấy nội dung từ file này.

| # | Khác gì | Paper / code gốc | Nhóm | Vì sao | Ảnh hưởng dự kiến |
|---|---|---|---|---|---|
| 1 | Mô hình | 10 mô hình, gồm GPT-3 và 13B/20B | GPT-Neo 1.3B/2.7B, OPT 1.3B/2.7B (6B/6.7B nếu đủ GPU) | GPT-3 đã ngừng; 13B/20B vượt GPU miễn phí | Không so được hàng GPT-3, 13B, 20B |
| 2 | Contriever | Có | Không | Tác giả mất file kết quả truy xuất | Bảng 1 làm lại bằng BM25 |
| 3 | Độ chính xác số | fp32 (int8 cho 13B/20B) | fp16 trên GPU | Vừa bộ nhớ GPU miễn phí | Có thể lệch nhẹ |
| 4 | Sinh văn bản | Từng câu một | Theo batch, sắp theo độ dài | Nhanh hơn nhiều | Lý thuyết là không đổi kết quả; kiểm ở issue #8 |
| 5 | Ví dụ mẫu | Chọn ngẫu nhiên mỗi lần chạy | Cố định theo `id` câu hỏi | Chạy tiếp được, các cách dùng chung ví dụ | Lệch nhỏ do ví dụ khác |
| 6 | Ghép kết quả cho adaptive | Sắp xếp theo `question` | Theo `id` | 881 câu hỏi trùng văn bản | Sửa lỗi ghép nhầm |
| 7 | Ví dụ mẫu của BM25 | Code gốc với file BM25 hiện có: không kèm đoạn văn | Kèm đoạn văn | Chờ quyết định ở issue #9 | Chưa biết, đo ở issue #8 |

Thêm dòng mới khi có thay đổi. Ghi rõ issue liên quan.
