# Nhật ký các lần chạy

Mỗi lần chạy trên Kaggle hoặc Colab thêm một dòng.

| Ngày | Mô hình | Cách | Số câu | GPU | Thời gian | Notebook (link phiên bản) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 25/10/2026 | Qwen/Qwen2.5-0.5B | vanilla, BM25, genread | 300 | Tesla T4 ×2 | 8 phút 12 giây (gồm tải mô hình) | chưa có | Smoke test (issue #8), fp16. Tốc độ: vanilla 14,2 câu/giây, BM25 2,4 câu/giây, genread 1,4 câu/giây. Độ chính xác: vanilla 13,3%, BM25 26,3%, genread 14,3%. |
| 26/10/2026 | EleutherAI/gpt-neo-1.3B (code gốc `run_model.py`, fp32) | vanilla, BM25 | 300 | Tesla T4 ×2 (chạy trên 1 GPU) | 8,6 phút (vanilla 4,5 + BM25 4,2; mỗi lần gồm tải mô hình) | [popqa-control-upstream-gptneo1p3b](https://www.kaggle.com/code/vanductan/popqa-control-upstream-gptneo1p3b) | Đối chứng, bên A, 15 ví dụ mẫu, repo ở commit `e17e0b7`. Độ chính xác: vanilla 11,33%, BM25 24,33%. So sánh: `docs/DOI_CHUNG_2026-10-26.md`. |
| 26/10/2026 | EleutherAI/gpt-neo-1.3B (notebook nhóm, fp16) | vanilla, BM25 | 300 | Tesla T4 ×2 (chạy trên 1 GPU) | 4,8 phút chạy (vanilla 9,29 câu/giây, BM25 1,18 câu/giây) + 0,9 phút tải mô hình | [popqa-control-ours-gptneo1p3b](https://www.kaggle.com/code/vanductan/popqa-control-ours-gptneo1p3b) | Đối chứng, bên B, 15 ví dụ mẫu, `SUBSET_IDS` = `data/subset_300_ids.txt`. Độ chính xác: vanilla 14,33%, BM25 26,67%. |
