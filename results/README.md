# Kết quả

| Thư mục | Chứa gì | Lên git? |
|---|---|---|
| `results/raw/` | File CSV notebook xuất ra, có cột `question` và `passage` (300 ký tự đầu của đoạn văn), vài MB mỗi lần chạy đủ 14.267 câu | Không |
| `results/slim/` | Bản rút gọn, đủ để gộp bảng và phân tích | Có |
| `results/tables/` | Bảng tổng hợp sinh từ `results/slim/` | Có |
| `results/figs/` | Hình vẽ | Có |

Lý do tách: bản đầy đủ của nhiều mô hình × nhiều cách cộng lại nặng, mà phân tích chỉ cần vài cột. Khi cần đọc đoạn văn để tìm ví dụ (phân tích 10%), mở bản trong `results/raw/`.

## Tên file

Giữ nguyên tên notebook đặt: `{mô hình}__{dtype}__{cách}__shots{số shot}.csv`, ví dụ `EleutherAI_gpt-neo-1.3B__fp16__BM25__shots15.csv`. Đừng đổi tên tay, vì script gộp bảng đọc thông tin từ tên file.

## Cột

Bản đầy đủ (notebook ghi): `id, prop, prop_id, question, s_pop, pred, is_correct, has_answer, retrieval_id, passage, model, method, n_shots`.

Bản rút gọn: bỏ `question` và `passage`, giữ phần còn lại. Do `scripts/slim_results.py` sinh ra (issue #23).

- `has_answer`: đoạn văn top-1 có chứa đáp án hay không (chỉ có ở BM25 và GenRead).

## Ghi lại mỗi lần chạy

Sau mỗi lần chạy trên Kaggle, thêm một dòng vào `results/RUNS.md`: ngày, mô hình, cách, số câu, GPU, thời gian chạy, link phiên bản notebook Kaggle.
