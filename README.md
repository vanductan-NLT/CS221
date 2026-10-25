# CS221 · Tái lập "When Not to Trust Language Models"

Đồ án cuối kỳ môn Nhập môn Xử lý ngôn ngữ tự nhiên (CS221, UIT). Nhóm chạy lại paper
[When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://aclanthology.org/2023.acl-long.546/)
(Mallen và cộng sự, ACL 2023, Long Paper) trên bộ dữ liệu PopQA, rồi đổi từng thứ một để xem kết luận của paper còn đúng không.

| Phần điểm | Làm gì |
|---|---|
| 60% | Trình bày bài toán, phương pháp, kết quả chạy lại so với paper |
| 30% | Mở rộng: mô hình đời mới (Qwen2.5), thêm một hướng phụ |
| 10% | Phân tích: vì sao số chênh, mô hình sai ở đâu |

Trình bày 15 phút. Mục tiêu: xong mọi phần trước 15/11/2026.

## Cách làm việc

- Mỗi việc là một **issue**, xếp theo **milestone** M0 → M6. Chưa gán người, chưa gán ngày.
- Nhãn `claude-code`: Claude Code viết được. Nhãn `tay`: phải tự làm (chạy GPU trên Kaggle, nhắn cô, đọc số từ hình).
- Giao việc cho Claude Code: mở repo, rồi nói `Đọc CLAUDE.md rồi làm issue #<số>`.
- Mọi khác biệt so với paper ghi vào [`docs/KHAC_BIET.md`](docs/KHAC_BIET.md). Mọi quyết định ghi vào [`docs/QUYET_DINH.md`](docs/QUYET_DINH.md).

## Thứ tự gợi ý để bắt đầu

1. Issue #1: mở repo bằng Claude Code, kiểm tra khung.
2. Issue #2, #3, #4 (làm tay, song song): nhắn cô, tạo tài khoản Kaggle, upload dữ liệu BM25.
3. Issue #6, #11, #20 (Claude Code, làm được ngay): notebook chạy code gốc, lưu độ tự tin, script rút gọn kết quả.
4. Issue #5 và #7 (chạy trên Kaggle): smoke test, chạy đối chứng.
5. Issue #8 → #9 → #10: so sánh, quyết định, chốt tập câu hỏi. Sau đó mới vào M2.
6. Các bài học nền ở M1 làm xen kẽ, không phải chờ.

## Thư mục

| Thư mục | Chứa gì |
|---|---|
| `notebooks/` | Notebook chạy lại thực nghiệm trên Kaggle/Colab và hướng dẫn chạy |
| `upstream/` | Code gốc của tác giả, giữ nguyên ở commit `8172ab0`, chỉ đọc. Lỗi đã biết: [`upstream/KNOWN_ISSUES.md`](upstream/KNOWN_ISSUES.md) |
| `scripts/` | Script chạy trên CPU: rút gọn kết quả, gộp bảng, so sánh |
| `data/` | Hướng dẫn lấy dữ liệu (file lớn không đưa lên git) |
| `results/` | Kết quả rút gọn để đưa lên git; bản đầy đủ để ngoài git |
| `docs/` | Kế hoạch, ghi chú paper, nhật ký khác biệt, nhật ký quyết định |

## Chạy ở đâu

Mô hình chạy trên GPU miễn phí: Kaggle (T4 ×2 hoặc P100) hoặc Colab (T4). Xem [`notebooks/HUONG_DAN.md`](notebooks/HUONG_DAN.md).
Phân tích, gộp bảng, vẽ hình chạy được trên máy thường.

## Nguồn

- Paper: https://aclanthology.org/2023.acl-long.546/
- Code gốc: https://github.com/AlexTMallen/adaptive-retrieval (MIT)
- File BM25 tác giả đăng lại: https://github.com/AlexTMallen/adaptive-retrieval/issues/5
- PopQA trên Hugging Face: https://huggingface.co/datasets/akariasai/PopQA
