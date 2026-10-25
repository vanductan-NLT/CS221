# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này. Đọc hết trước khi làm bất kỳ issue nào.

## Bối cảnh

- Đồ án cuối kỳ CS221 (Nhập môn NLP, UIT), nhóm 4 người. Người làm chính với Claude Code: Tân.
- Paper: "When Not to Trust Language Models" (Mallen và cộng sự, ACL 2023). Bộ dữ liệu PopQA: 14.267 câu hỏi về thực thể, 16 loại quan hệ, mỗi câu có độ nổi tiếng `s_pop` (lượt xem trang Wikipedia mỗi tháng).
- Các cách chạy: `vanilla` (mô hình tự nhớ), `BM25` (dán 1 đoạn Wikipedia vào trước câu hỏi), `genread` (mô hình tự viết đoạn nền), `adaptive` (chỉ tra cứu khi `s_pop` dưới ngưỡng, ngưỡng riêng cho từng loại quan hệ; tính từ kết quả vanilla và BM25, không chạy model thêm).
- 4 kết luận cần kiểm tra:
  1. Thực thể càng nổi tiếng thì mô hình càng hay trả lời đúng.
  2. Mô hình to hơn chủ yếu giỏi hơn ở câu nổi tiếng.
  3. Retrieval giúp câu ít nổi tiếng, có thể hại câu nổi tiếng.
  4. Adaptive tốt bằng hoặc hơn cả hai cách, và tra cứu ít lần hơn.
- Điểm: 60% trình bày + chạy lại, 30% mở rộng, 10% phân tích. Trình bày 15 phút.

## Người dùng đang học

Tân mới bắt đầu học NLP, biết Python cơ bản. Khi làm việc:

- Giải thích bằng tiếng Việt: trực giác và ví dụ trước, công thức sau.
- Sau mỗi thay đổi, tóm tắt ngắn: đã làm gì, vì sao, cách tự kiểm tra lại.
- Khi issue có chỗ chưa rõ hoặc phải chọn giữa hai cách, hỏi trước khi làm.
- Không làm thay phần cần hiểu để trả lời Q&A. Gặp khái niệm mới (BM25, few-shot, logprob…), giải thích ngắn gọn.

## Cách nhận và làm một issue

1. Đọc issue bằng `gh issue view <số>`. Kiểm tra mục "Phụ thuộc": issue phụ thuộc chưa đóng thì báo lại, đừng làm tiếp.
2. Tạo nhánh `issue-<số>-<tên-ngắn>` từ `main`.
3. Làm đúng phạm vi issue. Việc phát sinh thì đề xuất issue mới, không tự mở rộng.
4. Commit message tiếng Việt, có `#<số>`. Mở PR với `Closes #<số>` trong mô tả. Tân duyệt rồi merge.
5. Issue có nhãn `tay` là việc Tân tự làm (chạy GPU trên Kaggle, nhắn cô…). Claude Code chỉ hỗ trợ: viết code, đọc log lỗi, giải thích.

## Luật giữ tính tái lập (không tự ý đổi)

Các thiết lập dưới đây phải giống code gốc. Muốn đổi thì hỏi Tân, rồi ghi vào `docs/KHAC_BIET.md`.

- Template câu hỏi `Q: {question} A:`; 15 ví dụ mẫu, mỗi ví dụ thuộc một loại quan hệ khác với câu đang hỏi.
- Giải mã greedy, `max_new_tokens = 15`, lấy dòng đầu của output.
- Ngữ cảnh tối đa 2048 token, cắt từ bên trái như code gốc.
- Chấm điểm: đúng nếu một đáp án trong `possible_answers` xuất hiện trong dự đoán (chuỗi con; thử nguyên bản, chữ thường, viết hoa chữ đầu). Cách chấm chặt hơn là một cột **mới**, không thay cột `is_correct`.
- Seed 633. Adaptive: chia 75/25, lặp 100 lần, ngưỡng theo từng loại quan hệ.
- Ghép kết quả theo `id`, không theo văn bản câu hỏi (PopQA có 881 câu hỏi trùng văn bản, phủ 2.080 dòng).

## GPU và nơi chạy

- Không chạy mô hình ngôn ngữ thật trong Claude Code. Viết notebook hoặc script để Tân chạy trên Kaggle (T4 ×2 hoặc P100) hoặc Colab (T4).
- Được chạy trên CPU: phân tích, gộp bảng, vẽ hình, test nhỏ với dữ liệu giả hoặc mô hình tí hon.
- Notebook chính: `notebooks/adaptive_retrieval_repro.ipynb` (bản văn bản: `.py` cùng tên). Hai file phải giữ khớp nhau khi sửa.

## Dữ liệu và kết quả

- PopQA: `upstream/data/popQA.tsv`.
- File BM25 (`wikidata_bm25_14282_retrieved_docs.jsonl`, 42 MB): không đưa lên git; lấy từ dataset Kaggle, xem `data/README.md`.
- Kết quả đầy đủ notebook xuất ra (có cột `question`, `passage`): để ở `results/raw/`, không commit.
- Kết quả rút gọn: `results/slim/`, được commit. Quy ước cột và tên file: `results/README.md`.
- Số của paper chỉ đọc được từ biểu đồ: `docs/paper_numbers.csv`, luôn ghi là xấp xỉ.

## Không được

- Bịa số liệu, kết quả hay trích dẫn. Chưa chạy thì ghi "chưa có".
- Sửa file trong `upstream/`. Thư mục đó giữ nguyên code tác giả để đối chứng; muốn sửa thì chép ra `scripts/` hoặc `notebooks/`.
- Commit file lớn (trên 10 MB), khóa API, `kaggle.json`.

## Tài liệu trong repo

- Kế hoạch: `docs/KE_HOACH.md`
- Khác biệt so với paper: `docs/KHAC_BIET.md`
- Quyết định đã chốt và đang chờ: `docs/QUYET_DINH.md`
- Lỗi đã biết của code gốc: `upstream/KNOWN_ISSUES.md`
