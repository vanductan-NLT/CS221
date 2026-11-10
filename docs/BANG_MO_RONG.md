# Bảng mở rộng

## Hướng C: adaptive theo độ tự tin (issue #34)

Quyết định chọn hướng C: `docs/QUYET_DINH.md` (10/11/2026).

### Câu hỏi

Paper quyết định "có tra cứu không" bằng độ nổi tiếng `s_pop` của thực thể. Ngoài đời không ai biết trước `s_pop` của một câu hỏi, nhưng mô hình biết nó chắc đến đâu về câu trả lời của chính nó. Nếu dùng độ tự tin thay cho độ nổi tiếng thì adaptive còn tốt không?

### Cách làm

- Độ tự tin là cột `conf` của lượt vanilla (issue #14): trung bình log-xác suất của các token trong câu trả lời. Gần 0 là rất chắc; càng âm càng không chắc.
- Quy trình giống hệt adaptive gốc (`scripts/aggregate.py`, hàm `adaptive_eval`): với mỗi loại quan hệ chia câu thành 10 bin theo điểm, chọn ngưỡng tốt nhất trên 75% câu (dưới ngưỡng dùng BM25, từ ngưỡng trở lên dùng vanilla), đo trên 25% còn lại, lặp 100 lần, seed 633. Chỉ đổi "điểm" từ log(`s_pop`) sang `conf`.
- Hai cách điểm dùng đúng cùng 100 lần chia, nên so cặp được.
- Số câu: đủ 14.267. Chạy lại: `python scripts/adaptive_by_confidence.py`; kiểm: `python scripts/test_adaptive_by_confidence.py`. Kết quả đầy đủ (kể cả lượt 300 câu): `results/tables/adaptive_by_confidence.csv`.

### Kết quả 1: độ chính xác và số lần tra cứu

| Mô hình | vanilla | BM25 | adaptive theo độ nổi tiếng | adaptive theo độ tự tin | chênh (điểm %) | tra cứu: nổi tiếng → tự tin | conf thắng (trên 100 lần chia) | AUROC độ tự tin | AUROC độ nổi tiếng |
|---|---|---|---|---|---|---|---|---|---|
| GPT-Neo 1.3B | 11,35% | 21,61% | 21,61% | 21,54% ± 0,54 | -0,07 | 94,9% → 95,7% | 25 | 0,78 | 0,53 |
| GPT-Neo 2.7B | 12,37% | 22,75% | 23,27% | 23,25% ± 0,57 | -0,02 | 90,8% → 91,6% | 42 | 0,77 | 0,56 |
| OPT 1.3B | 12,29% | 21,96% | 22,04% | 22,00% ± 0,60 | -0,04 | 90,7% → 93,1% | 37 | 0,79 | 0,56 |
| OPT 2.7B | 12,56% | 23,23% | 23,79% | 23,78% ± 0,61 | -0,01 | 88,5% → 90,8% | 48 | 0,82 | 0,62 |
| OPT 6.7B (fp16) | 15,66% | 24,62% | 25,79% | 25,65% ± 0,66 | -0,14 | 83,7% → 87,0% | 25 | 0,83 | 0,63 |
| GPT-J 6B (fp16) | 16,09% | 24,43% | 25,56% | 25,38% ± 0,59 | -0,18 | 83,3% → 89,5% | 18 | 0,81 | 0,63 |
| Qwen2.5 0.5B | 11,99% | 22,55% | 22,72% | 22,65% ± 0,56 | -0,07 | 93,4% → 92,0% | 28 | 0,86 | 0,53 |
| Qwen2.5 1.5B | 15,95% | 24,88% | 25,69% | 25,67% ± 0,62 | -0,02 | 82,2% → 87,5% | 44 | 0,83 | 0,58 |

Qwen2.5 3B chưa có: lượt BM25 còn 1.289 câu (issue #30). Có đủ thì chạy lại `scripts/adaptive_by_confidence.py` và thêm dòng.

Đọc bảng:
- **Độ chính xác không đổi.** Adaptive theo độ tự tin thấp hơn theo độ nổi tiếng từ 0,01 đến 0,18 điểm, ở cả 8 mô hình; độ lệch giữa các lần chia là 0,54 đến 0,66 điểm. Cách theo độ tự tin thắng ít hơn một nửa trong 100 lần chia ở cả 8 mô hình (18 đến 48 lần).
- **Tra cứu nhiều hơn, không ít hơn.** Ở 7 trên 8 mô hình, dùng độ tự tin phải tra cứu nhiều câu hơn, thêm từ 0,8 đến 6,2 điểm phần trăm số câu (GPT-J nhiều nhất); chỉ Qwen2.5 0.5B tra cứu ít hơn (93,4% xuống 92,0%).
- **Độ tự tin phân biệt câu đúng/sai tốt hơn nhiều.** AUROC (xác suất chọn ngẫu nhiên một câu vanilla đúng và một câu vanilla sai thì điểm của câu đúng cao hơn; 0,5 là đoán mò, 1 là hoàn hảo): độ tự tin 0,77 đến 0,86, độ nổi tiếng 0,53 đến 0,63. Hai điểm chỉ liên quan yếu (hệ số Spearman 0,08 đến 0,27).

### Kết quả 2: vì sao điểm phân biệt tốt hơn mà adaptive không tốt hơn

Adaptive không cần biết "mô hình có đúng không" mà cần biết **tra cứu có giúp hay hại**. Bảng dưới xếp mọi câu theo điểm tăng dần, chia 5 nhóm bằng số câu; mỗi ô là BM25 hơn vanilla bao nhiêu điểm phần trăm (dương là tra cứu giúp). Gộp mọi loại quan hệ cho dễ đọc; adaptive thật chọn ngưỡng riêng cho từng loại quan hệ. Chạy lại ra `results/tables/adaptive_by_confidence_bins.csv`.

| Mô hình | Điểm | Nhóm 1 (thấp nhất) | Nhóm 2 | Nhóm 3 | Nhóm 4 | Nhóm 5 (cao nhất) |
|---|---|---|---|---|---|---|
| GPT-Neo 1.3B | độ nổi tiếng | +16,4 | +12,7 | +9,7 | +8,5 | +4,0 |
| GPT-Neo 1.3B | độ tự tin | +10,9 | +12,0 | +9,5 | +10,5 | +8,3 |
| GPT-Neo 2.7B | độ nổi tiếng | +18,7 | +14,0 | +9,8 | +7,9 | +1,5 |
| GPT-Neo 2.7B | độ tự tin | +12,3 | +10,4 | +9,6 | +10,7 | +8,9 |
| OPT 1.3B | độ nổi tiếng | +16,0 | +12,5 | +9,6 | +8,2 | +2,0 |
| OPT 1.3B | độ tự tin | +9,8 | +10,4 | +8,3 | +10,5 | +9,3 |
| OPT 2.7B | độ nổi tiếng | +20,5 | +14,4 | +10,2 | +8,2 | +0,1 |
| OPT 2.7B | độ tự tin | +9,9 | +13,3 | +10,9 | +12,8 | +6,4 |
| OPT 6.7B (fp16) | độ nổi tiếng | +18,9 | +13,7 | +8,9 | +6,6 | -3,3 |
| OPT 6.7B (fp16) | độ tự tin | +11,2 | +11,7 | +10,9 | +11,0 | +0,0 |
| GPT-J 6B (fp16) | độ nổi tiếng | +17,9 | +14,0 | +8,8 | +6,4 | -5,4 |
| GPT-J 6B (fp16) | độ tự tin | +9,6 | +9,4 | +8,7 | +13,0 | +1,1 |
| Qwen2.5 0.5B | độ nổi tiếng | +17,6 | +13,5 | +9,4 | +8,6 | +3,9 |
| Qwen2.5 0.5B | độ tự tin | +5,9 | +11,4 | +9,5 | +14,9 | +11,2 |
| Qwen2.5 1.5B | độ nổi tiếng | +17,5 | +13,0 | +9,4 | +6,7 | -1,9 |
| Qwen2.5 1.5B | độ tự tin | +8,3 | +10,2 | +11,1 | +12,3 | +2,8 |

- Theo **độ nổi tiếng**, lợi ích của tra cứu giảm đều từ nhóm ít nổi tiếng (+16 đến +21 điểm) xuống nhóm nổi tiếng nhất, ở cả 8 mô hình. Ở ba mô hình (OPT 6.7B, GPT-J 6B, Qwen2.5 1.5B) nhóm nổi tiếng nhất bị **hại** (-3,3, -5,4, -1,9). Đây là chỗ adaptive gốc có thể kiếm thêm điểm: bỏ tra cứu ở nhóm đó.
- Theo **độ tự tin**, lợi ích gần như phẳng ở bốn nhóm đầu (từ +6 đến +15). Nhóm tự tin nhất chỉ giảm rõ ở OPT 6.7B, GPT-J 6B và Qwen2.5 1.5B (từ 0,0 đến +2,8); ở năm mô hình còn lại vẫn từ +6,4 đến +11,2. Vậy ở ba mô hình đó, vanilla và BM25 đúng gần bằng nhau ở nhóm tự tin nhất, nên bỏ tra cứu ở đó chỉ tiết kiệm lượt tra cứu, không thêm điểm; ở năm mô hình kia BM25 vẫn hơn hẳn ngay cả ở nhóm tự tin nhất, nên gần như không có chỗ nào đáng bỏ tra cứu. Ở cả 8 mô hình, độ tự tin không tìm ra nhóm mà tra cứu gây hại (nhỏ nhất là 0,0).

### Khi nào thay được, khi nào không

- **Thay được** nếu chỉ cần một điểm mà không phải biết trước độ nổi tiếng: độ tự tin cho độ chính xác gần như bằng (kém 0,01 đến 0,18 điểm, nằm trong độ lệch giữa các lần chia).
- **Không tốt hơn**: không giảm số lần tra cứu (7 trên 8 mô hình tra cứu nhiều hơn) và không bắt được nhóm câu nổi tiếng mà tra cứu làm hại (OPT 6.7B, GPT-J 6B, Qwen2.5 1.5B).
- Đây là kết quả không thắng, ghi đúng như đo. Giải thích ở trên là suy ra từ bảng nhóm điểm, chưa kiểm riêng bằng thí nghiệm khác (ví dụ ghép hai điểm).

### Giới hạn

- Cùng một dữ liệu cho cả 100 lần chia, nên "conf thắng bao nhiêu lần" cho thấy độ ổn định, không phải kiểm định thống kê.
- Độ tự tin tính trên câu trả lời vanilla (giải mã greedy, 15 token, lấy dòng đầu). Chưa thử cách tính khác.
- Chưa có Qwen2.5 3B (BM25 chưa xong, issue #30). Bảng 300 câu nằm trong file CSV nhưng nhiễu, không dùng để kết luận.
