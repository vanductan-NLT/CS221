# Kết quả

| Thư mục | Chứa gì | Lên git? |
|---|---|---|
| `results/raw/` | File CSV notebook xuất ra, có cột `question` và `passage` (300 ký tự đầu của đoạn văn), vài MB mỗi lần chạy đủ 14.267 câu | Không |
| `results/slim/` | Bản rút gọn, đủ để gộp bảng và phân tích | Có |
| `results/tables/` | Bảng tổng hợp sinh từ `results/slim/`: `summary.csv` (độ chính xác từng cách, adaptive, tỉ lệ phải tra cứu) do `python scripts/aggregate.py` ghi (issue #24) | Có |
| `results/figs/` | Hình vẽ | Có |

Lý do tách: bản đầy đủ của nhiều mô hình × nhiều cách cộng lại nặng, mà phân tích chỉ cần vài cột. Khi cần đọc đoạn văn để tìm ví dụ (phân tích 10%), mở bản trong `results/raw/`.

## Tên file

Giữ nguyên tên notebook đặt: `{mô hình}__{dtype}__{cách}__shots{số shot}.csv`, ví dụ `EleutherAI_gpt-neo-1.3B__fp16__BM25__shots15.csv`. Đừng đổi tên tay, vì script gộp bảng đọc thông tin từ tên file.

File do code gốc `run_model.py` ghi giữ tên của code gốc: `model={mô hình}-input={tên lượt chạy}-method={cách}-shots={số shot}-n={số câu}.csv`, ví dụ `model=EleutherAI_gpt-neo-1.3B-input=popqa-shot-passages-full-method=BM25-shots=15-n=14267.csv` (code gốc chạy fp32).

## Cột

Bản đầy đủ (notebook ghi): `id, prop, prop_id, question, s_pop, pred, is_correct, has_answer, retrieval_id, passage, model, method, n_shots`.

Bản đầy đủ của code gốc có thêm `prompt`, `generation`, alias, link Wikidata… nên nặng hơn nhiều (khoảng 240 MB cho 14.267 câu), và không có cột `model, method, n_shots` (đã nằm trong tên file).

Bản rút gọn: giữ `id, prop, prop_id, s_pop, pred, is_correct, has_answer, retrieval_id, conf, model, method, n_shots` (cột nào file có), bỏ phần còn lại; cần thêm thông tin câu hỏi thì ghép với `upstream/data/popQA.tsv` theo `id`. Do `scripts/slim_results.py` sinh ra (issue #23), cùng tên với file đầy đủ:

```
python scripts/slim_results.py            # mọi file trong results/raw/
```

- `has_answer`: đoạn văn top-1 có chứa đáp án hay không (chỉ có ở BM25 và GenRead).
- `conf` (issue #14): độ tự tin của câu trả lời = trung bình log-xác suất của các token tạo nên `pred` (dòng đầu của phần mô hình sinh ra, sau khi bỏ `\n\n` ở đầu như code gốc). Luôn ≤ 0; càng gần 0 càng tự tin. Để trống khi `pred` rỗng. Chỉ có ở các lượt chạy từ 07/11/2026 (`notebooks/gpt_neo_vanilla_opt_1_3b.ipynb`, `notebooks/qwen2_5_0_5b_1_5b.ipynb`); bật `output_scores` không đổi prompt hay câu trả lời.

## Ghi lại mỗi lần chạy

Sau mỗi lần chạy trên Kaggle, thêm một dòng vào `results/RUNS.md`: ngày, mô hình, cách, số câu, GPU, thời gian chạy, link phiên bản notebook Kaggle.
