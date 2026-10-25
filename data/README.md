# Dữ liệu

File lớn không đưa lên git. Thư mục `data/raw/` đã nằm trong `.gitignore`, dùng để chứa bản tải về trên máy.

| Dữ liệu | Ở đâu | Ghi chú |
|---|---|---|
| PopQA (14.267 câu) | `upstream/data/popQA.tsv`, hoặc Hugging Face `akariasai/PopQA` | Có `id`, `prop_id`, `s_pop`, `question`, `possible_answers` |
| Kết quả BM25 cho PopQA | Dataset Kaggle `vanductan/popqa-bm25-retrieval` (tạo ở issue #7) | `wikidata_bm25_14282_retrieved_docs.jsonl`, 42 MB, mỗi câu 5 đoạn (`id`, `text`, `hasanswer`). Nguồn: file `PopQA_BM25.jsonl.zip` tác giả đăng ở https://github.com/AlexTMallen/adaptive-retrieval/issues/5 |
| Kết quả BM25 cho EntityQuestions | Cũng ở issue #5 của repo tác giả | Chỉ dùng nếu dư thời gian |
| Contriever | Không còn | Tác giả xác nhận đã mất file. Không tái lập phần này |

Tên dataset Kaggle có thể khác nếu tài khoản Kaggle khác `vanductan`. Tạo xong thì sửa lại dòng trên cho đúng.

Đoạn văn lấy từ Wikipedia (giấy phép CC BY-SA).
