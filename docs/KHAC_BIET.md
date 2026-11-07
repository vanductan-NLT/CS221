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
| 9 | Đặt lại seed trước khi chọn câu, khi chạy code gốc trên Kaggle bản 11/2026 (Python 3.13, torch 2.11) | `np.random.seed(633)` ở đầu chương trình, sau đó mới import thư viện, nạp mô hình rồi `knowledge.sample(...)` | Notebook [cs221-popqa-gpt-neo-bm25-shot-passages](https://www.kaggle.com/code/vanductan/cs221-popqa-gpt-neo-bm25-shot-passages) chèn `random.seed(633)` và `np.random.seed(633)` ngay trước dòng chọn câu (vào bản chép của `run_model.py`, không sửa `upstream/`) | Môi trường Kaggle mới rút đúng 5 số ngẫu nhiên của numpy giữa lúc đặt seed và lúc chọn câu, nên 300 câu chọn ra lệch khỏi `data/subset_300_ids.txt`; lượt chạy đầu bị chặn ở bước kiểm ID (issue #12) | Không đổi kết quả so với code gốc chạy trên môi trường sạch: sau khi đặt lại seed, code chọn đúng 300 ID và đúng thứ tự như lượt đối chứng 26/10/2026 (`results/RUNS.md`) |
| 10 | Cửa sổ ngữ cảnh của Qwen2.5 (phần mở rộng, issue #30) | Code gốc chỉ cắt prompt ở 2048 token cho OPT và GPT-NeoX-20B; mô hình khác dùng `tokenizer.model_max_length` (GPT-Neo: 2048; Qwen2.5: 131.072, tức gần như không cắt) | Notebook Qwen thêm Qwen vào điều kiện cắt 2048 của code gốc (sửa bản chép của `run_model.py`, không sửa `upstream/`) | Tân chọn ngày 07/11/2026: giữ cùng cửa sổ với GPT-Neo/OPT để khác biệt giữa các mô hình chỉ đến từ mô hình | Prompt BM25 dài hơn 2048 token mất vài ví dụ mẫu đầu, như GPT-Neo/OPT. Qwen có thể đạt cao hơn một chút nếu được đọc đủ prompt |

Thêm dòng mới khi có thay đổi. Ghi rõ issue liên quan.

## Lỗi của code tác giả

Mục này mô tả code tác giả đang đăng (`upstream/`, commit `8172ab0`). Nhóm kiểm ngày 25/10/2026 với file BM25 tác giả đăng lại ở issue #5 của repo tác giả. Thư mục Google Drive gốc, nơi tác giả để file truy xuất khi viết paper, đã mất, nên **không biết các lỗi liên quan tới file BM25 có xảy ra với bộ file tác giả dùng khi viết paper không**. Vì vậy nhóm chỉ nói "code đang đăng có lỗi này", không nói "paper sai". Chi tiết từng lỗi: [`upstream/KNOWN_ISSUES.md`](../upstream/KNOWN_ISSUES.md) (số "mục" ở dưới là số mục trong file đó).

### Loại làm chạy hỏng (phải lách trong `notebooks/upstream_kaggle.ipynb`)

| Lỗi | Chuyện gì xảy ra | Nhóm xử lý |
|---|---|---|
| Thiếu thư viện (mục 9) | `requirements.txt` không ghi phiên bản và không có `jsonlines`, `seaborn`, `matplotlib`, `pandas`, dù `run_model.py` import chúng. Thiếu `jsonlines` thì code dừng ngay với `ModuleNotFoundError` | Notebook cài `jsonlines` và `openai`; `seaborn`, `matplotlib`, `pandas` Kaggle có sẵn |
| Sai tên tham số (mục 6) | README ghi `--ret_file`, nhưng `argparse` chỉ nhận `--ret_path` | Dùng `--ret_path` |
| `--continue_from` hỏng (mục 7) | Code đọc file kết quả với `sep="\t"` nhưng file được ghi bằng dấu phẩy, nên chạy tiếp từ chỗ dừng bị hỏng | Notebook không dùng `--continue_from`, chạy lại từ đầu mỗi lần (300 câu, mỗi cách chưa tới 5 phút) |
| GPT-3 không chạy lại được (mục 8) | `text-davinci-*` đã bị OpenAI ngừng; `openai.Completion` không còn từ bản `openai` 1.0 | Ngoài phạm vi, xem dòng 1 của bảng ở trên |
| Nạp mô hình fp32 rồi mới chuyển lên GPU (mục 10) | Từ 2.7B trở lên dễ hết RAM trên Kaggle/Colab miễn phí | Notebook của nhóm nạp thẳng fp16 hoặc 4-bit (dòng 3) |

### Loại lỗi âm thầm (vẫn chạy, nhưng làm số lệch)

| Lỗi | Chuyện gì xảy ra | Nhóm xử lý |
|---|---|---|
| Ví dụ mẫu của BM25 không có đoạn văn (mục 1) | Câu hỏi chính tra đoạn văn bằng nguyên văn câu hỏi, còn 15 ví dụ mẫu tra bằng bản đã bỏ `?` và viết thường. Khóa trong file BM25 là nguyên văn, nên câu chính khớp 14.267/14.267 còn ví dụ mẫu khớp 0/14.267. Code chỉ in `missing retrieval` rồi chạy tiếp (lần chạy đối chứng 26/10/2026: đúng 4.500 dòng, bằng 300 câu × 15 ví dụ) | Notebook của nhóm kèm đoạn văn cho ví dụ mẫu (dòng 7); chờ quyết định ở issue #12 |
| Câu hỏi trùng văn bản dùng chung một đoạn văn (mục 2) | PopQA có 881 câu hỏi trùng văn bản, phủ 2.080 dòng. Khóa là văn bản nên các dòng trùng nhận cùng một đoạn văn; notebook adaptive gốc ghép kết quả theo `question` nên có thể ghép nhầm dòng vanilla với dòng BM25 | Notebook của nhóm ghép theo `id` (dòng 6) |
| Prompt quá 2048 token bị cắt mà không báo (mục 3) | Code giữ 2048 − 15 token cuối; với BM25 các ví dụ đầu có thể bị cắt mất, kể cả cắt giữa chừng | Notebook của nhóm cắt từ bên trái như vậy; đo trên 300 câu BM25 ngày 26/10/2026: code gốc 0/300 prompt bị cắt, notebook nhóm 159/300 vì kèm đoạn văn cho ví dụ mẫu (xem `docs/DOI_CHUNG_2026-10-26.md`) |
| Chấm điểm dễ dãi (mục 4) | Đúng nếu đáp án là chuỗi con của dự đoán; ví dụ alias `pol` của `politician` làm dự đoán `police officer` bị tính đúng | Giữ nguyên để so được với paper; chấm chặt hơn là một cột riêng, xem issue #35 |
| `has_answer` của GenRead bị thổi lên (mục 5) | Điều kiện sai thứ tự ưu tiên toán tử, và `response` gồm cả prompt có ví dụ mẫu chứa đáp án nên gần như luôn đúng | Notebook của nhóm tính trên đoạn GenRead sinh ra, không tính trên cả prompt |

Cả mục này chỉ nói về code đang đăng, không phải kết luận về số liệu trong paper.
