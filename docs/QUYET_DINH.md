# Nhật ký quyết định

## Đã chốt

| Ngày | Quyết định | Lý do |
|---|---|---|
| 25/10/2026 | Chọn paper "When Not to Trust Language Models" (ACL 2023) | Cô đã duyệt; có code công khai; chạy được mô hình gốc trên GPU miễn phí |
| 25/10/2026 | Hướng mở rộng chính: mô hình đời mới (Qwen2.5 bản base) | Đúng ví dụ của cô; chi phí vừa phải |
| 26/10/2026 | Code gốc chạy bằng `--sample 300` trên file `popQA.tsv` đầy đủ, không lọc file xuống 300 dòng. Notebook `upstream_kaggle.ipynb` tự kiểm id kết quả khớp `data/subset_300_ids.txt` rồi mới coi là đúng | Lọc file khiến 15 ví dụ mẫu chỉ bốc trong 300 dòng (loại `color` chỉ có 2 dòng), khác paper và khác notebook chính. Với seed 633, code gốc chọn đúng 300 id đó (đã thử trên CPU) |

## Đang chờ

| Issue | Câu hỏi | Chốt dựa vào |
|---|---|---|
| #5 | Cô có chấp nhận phạm vi tái lập (mô hình nhỏ, không GPT-3, không Contriever) không? | Câu trả lời của cô |
| #12 | Ví dụ mẫu của BM25 có kèm đoạn văn không? | Kết quả đối chứng ở issue #11 |
| #13 | Chạy đủ 14.267 câu hay một tập con cố định (tối thiểu 3.000)? | Tốc độ đo ở smoke test (issue #8) |
| #32 | Hướng phụ B (xếp hạng lại đoạn BM25) hay C (adaptive theo độ tự tin)? | Kết quả tái lập và thời gian còn lại |
