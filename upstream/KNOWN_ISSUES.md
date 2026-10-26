# Lỗi đã biết của code gốc

Kiểm tra ngày 25/10/2026 trên commit `8172ab0`, với file BM25 tác giả đăng ở issue #5 của repo tác giả (`wikidata_bm25_14282_retrieved_docs.jsonl`, 14.282 dòng).

## Ảnh hưởng tới kết quả

1. **Ví dụ mẫu của BM25 không có đoạn văn (đã xác minh).**
   - Câu hỏi chính tra đoạn văn bằng nguyên văn câu hỏi: `retrieval_dict[row.question]`.
   - 15 ví dụ mẫu tra bằng `row.question.replace("?", "").lower()`.
   - Khóa trong file BM25 là nguyên văn câu hỏi, nên câu hỏi chính khớp 14.267/14.267, còn ví dụ mẫu khớp 0/14.267.
   - Kết quả: mọi ví dụ mẫu của BM25 rơi về dạng không có đoạn văn, code chỉ in `missing retrieval` rồi chạy tiếp.
   - Không rõ file gốc tác giả dùng khi viết paper có khóa kiểu nào (thư mục Google Drive đã mất).
   - Notebook của nhóm thì có kèm đoạn văn cho ví dụ mẫu. Quyết định chọn cách nào: issue #12.
2. **Câu hỏi trùng văn bản dùng chung một đoạn văn.**
   - PopQA có 881 câu hỏi trùng văn bản, phủ 2.080 dòng (ví dụ hai người khác nhau cùng tên).
   - `retrieval_dict` lấy văn bản câu hỏi làm khóa, nên các dòng trùng nhận cùng một đoạn văn.
   - Notebook adaptive gốc ghép kết quả bằng cách sắp xếp theo `question`, nên với câu trùng, dòng vanilla và dòng BM25 có thể bị ghép nhầm.
   - Notebook của nhóm ghép theo `id`.
3. **Prompt dài hơn 2048 token bị cắt âm thầm.** `call_model` giữ lại 2048 − 15 token cuối. Với BM25 (mỗi ví dụ kèm đoạn văn), các ví dụ đầu có thể bị cắt mất, kể cả cắt giữa chừng.
4. **Cách chấm điểm dễ dãi.** Đúng nếu đáp án là chuỗi con của dự đoán. Ví dụ, alias `pol` của `politician` khiến dự đoán `police officer` cũng bị tính là đúng.
5. **`has_answer` của GenRead bị thổi lên.** Điều kiện `args.eval_method == "genread" and pa in response or ...` sai thứ tự ưu tiên toán tử. Ngoài ra `response` là toàn bộ văn bản (gồm cả prompt có các ví dụ mẫu chứa đáp án), nên gần như luôn đúng.

## Làm code không chạy hoặc chạy sai

6. README ghi `--ret_file`, nhưng `argparse` nhận `--ret_path`. Dùng `--ret_path`.
7. `--continue_from` đọc file với `sep="\t"`, trong khi kết quả được ghi bằng dấu phẩy, nên chạy tiếp từ chỗ dừng bị hỏng.
8. Các mô hình GPT-3 (`text-davinci-002/003`) đã bị OpenAI ngừng từ đầu 2024. `openai.Completion` cũng không còn trong thư viện `openai` từ bản 1.0. Phần GPT-3 không chạy lại được.
9. `requirements.txt` không ghi phiên bản. Code còn import `jsonlines`, `seaborn`, `matplotlib`, `pandas` mà không khai báo.
10. Mô hình được nạp ở fp32 rồi mới chuyển lên GPU. Với 2.7B trở lên dễ hết RAM trên Kaggle/Colab miễn phí.
