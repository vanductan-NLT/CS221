# Phân tích 3 model: DeepSeek API · FactCG · HalluGuard trên LLM-AggreFact

**Phạm vi:** tập test `lytang/LLM-AggreFact` (29,320 mẫu, 11 dataset). **Metric:** Balanced Accuracy (BAcc) = (TPR + TNR) / 2, lấy trung bình macro trên 11 dataset. Mọi model dùng ngưỡng cố định, **không tune ngưỡng**.  
**Nguồn số liệu:** output các notebook và file kết quả trong repo (xem Nguồn). Số liệu từ bài báo chỉ dùng để tham chiếu và luôn được ghi rõ nguồn.

---

## Tóm tắt

| Model | BAcc macro | Kết luận |
| --- | --- | --- |
| **DeepSeek API** (`deepseek-chat`) | **77.78** ±1.0 | Cao nhất trong các lần nhóm chạy, nhưng là LLM thương mại ra sau MiniCheck khoảng 2 năm và phải trả phí theo token |
| **FactCG-DeBERTa-v3-Large** (0.4B) | 74.79 ±1.1 | Hơn MiniCheck-DeBERTa **+1.64** trên cùng kiến trúc, nên phần lợi chủ yếu đến từ dữ liệu huấn luyện |
| **HalluGuard-Qwen3-4B** | 67.15 ±1.0 | Thấp vì chạy ở chế độ *không thinking*. Con số này khớp ablation của chính tác giả (67.6) |
| *Tham chiếu:* Bespoke-MiniCheck-7B (nhóm chạy lại) | 76.05 ±1.1 |  |

---

## 1. Thiết lập

|  | DeepSeek | FactCG | HalluGuard | Bespoke-7B *(tham chiếu)* |
| --- | --- | --- | --- | --- |
| Loại | LLM đóng, gọi qua API | Encoder phân loại (DeBERTa-v3-large) | SLM sinh văn bản (Qwen3-4B + LoRA/ORPO) | LLM chuyên biệt cho fact-checking |
| Tham số | không công bố | ~0.4B | 4B | 7B |
| Cách chấm | Cả văn bản trong 1 request, prompt zero-shot trả lời Yes/No, temperature 0 | `MiniCheck.score`, chunk 500, P(supported) > 0.5 | `MiniCheck.score`, chunk 500, trả lời đúng 1 từ YES/NO, **tắt thinking** | `MiniCheck.score`, chunk 500, P("Yes") > 0.5 |
| Phần cứng | Server DeepSeek, 10 luồng | 2× T4 | 2× T4 (vLLM) | 2× T4 (vLLM) |
| Thời gian chạy full | 43.6 phút | khoảng > 4h (không đo được chính xác vì bị ngắt rồi resum giữa chừng) | 6.3 giờ (0.78 s/mẫu) | 4.4 giờ |
| Chi phí | Trả theo token (§3.3) | 0 (Kaggle) | 0 (Kaggle) | 0 (Kaggle) |
| Mẫu lỗi | Không có request nào bị bỏ. Phản hồi không rõ Yes/No bị gán nhãn 0; repo không lưu nên không đếm được | 0 | 1 mẫu ExpertQA (prompt dài 4,312 token, vượt giới hạn 4,096), đã loại khỏi metric | 0 |

---

## 2. Kết quả theo dataset (BAcc %)

In **đậm** = cao nhất hàng. `*` = nhóm chạy lại trong `test_nb_ver1`.

| Dataset | Mẫu âm / dương | DeepSeek | Bespoke-7B* | FactCG | MiniCheck-DeBERTa* | HalluGuard |
| --- | --- | --- | --- | --- | --- | --- |
| AggreFact-CNN | 57 / 501 | 67.90 | **69.99** | 69.25 | 64.42 | 57.35 |
| AggreFact-XSum | 273 / 285 | **75.48** | 74.94 | 73.21 | 71.05 | 63.40 |
| ClaimVerify | 299 / 789 | 75.01 | 73.94 | **77.36** | 75.54 | 60.78 |
| ExpertQA | 731 / 2,971 | **60.37** | 57.96 | 58.94 | 58.95 | 55.30 |
| FactCheck-GPT | 1,190 / 376 | **78.87** | 77.99 | 72.06 | 73.04 | 73.84 |
| Lfqa | 790 / 1,121 | **87.37** | 86.14 | 86.51 | 83.93 | 72.93 |
| RAGTruth | 1,269 / 15,102 | **87.02** | 82.33 | 80.11 | 78.79 | 68.45 |
| Reveal | 1,310 / 400 | 88.32 | 87.82 | **88.39** | 87.43 | 83.14 |
| TofuEval-MediaS | 172 / 554 | 71.76 | **73.70** | 68.80 | 69.34 | 63.11 |
| TofuEval-MeetB | 150 / 622 | **82.50** | 74.84 | 74.87 | 72.70 | 71.16 |
| Wice | 247 / 111 | **80.94** | 76.92 | 73.18 | 69.44 | 69.17 |
| **Macro** | 29,320 | **77.78** | 76.05 | 74.79 | 73.15 | 67.15 |
| TPR / TNR (macro) |  | 84.8 / 70.7 | 79.4 / 72.8 | 73.9 / 75.7 | n/a | **92.3 / 42.0** |

> Các tập ít mẫu âm có khoảng tin cậy rộng: CNN (57 mẫu âm) khoảng ±5–7 điểm, Wice, MediaS và MeetB khoảng ±4–5 điểm cho *mỗi* model. Chênh lệch vài điểm trên các tập này chưa đủ để kết luận.

---

## 3. DeepSeek API

### 3.1 Làm tốt / kém ở đâu

Để so với model mở mạnh nhất nhóm đã chạy, chọn Bespoke-7B (cả hai đều có confusion matrix, nên tính được khoảng tin cậy của hiệu số):

| Dataset | DeepSeek | Bespoke-7B* | Δ | CI95 của Δ | Kết luận |
| --- | --- | --- | --- | --- | --- |
| RAGTruth | 87.02 | 82.33 | **+4.69** | ±1.49 | Hơn rõ |
| TofuEval-MeetB | 82.50 | 74.84 | **+7.66** | ±5.52 | Hơn rõ |
| Wice | 80.94 | 76.92 | +4.01 | ±6.58 | Hơn nhưng trong nhiễu |
| ExpertQA | 60.37 | 57.96 | +2.41 | ±2.80 | Trong nhiễu |
| Lfqa, FactCheck-GPT, ClaimVerify, XSum, Reveal |  |  | +0.5 → +1.2 | ±2.2 → ±5.1 | Ngang nhau |
| TofuEval-MediaS | 71.76 | 73.70 | −1.94 | ±5.47 | Không hơn |
| AggreFact-CNN | 67.90 | 69.99 | −2.09 | ±9.15 | Không hơn |
| **Macro** | **77.78** | 76.05 | **+1.72** | ±1.48 | Hơn, vừa vượt nhiễu |

- **Mạnh:** DeepSeek đứng đầu 7/11 dataset trong cả 7 model nhóm đã chạy (tính cả Flan-T5 và RoBERTa): XSum, ExpertQA, FactCheck-GPT, Lfqa, RAGTruth, MeetB, Wice. Hai tập hơn rõ ràng, vượt nhiễu, là **RAGTruth** (tập lớn nhất, 16k mẫu) và **MeetB** (biên bản họp dài).
- **Yếu tương đối:** ở CNN, MediaS và ClaimVerify, DeepSeek chỉ xếp 4/7. Nó thua khoảng 2 điểm (trong nhiễu) trước cả những model mở nhỏ hơn rất nhiều: Flan-T5 0.8B ở CNN và MediaS, RoBERTa và FactCG 0.4B ở ClaimVerify. 
- **Kiểu lỗi ở các tập yếu:** TNR thấp. CNN 38.6%, MediaS 47.7%, ClaimVerify 60.5%. Nghĩa là DeepSeek hay chấp nhận claim không được văn bản hỗ trợ (False Positive), nhất là với tóm tắt tin tức và hội thoại. Tính chung, TPR 84.8% > TNR 70.7%, tức DeepSeek hơi dễ dãi (nghiêng về "Yes").
- **Điểm tuyệt đối thấp nhất là ExpertQA (60.37)**, nhưng ở tập này DeepSeek vẫn cao nhất. ExpertQA khó với mọi model: không model nào vượt 60.4.

### 3.2 Bối cảnh bắt buộc khi so sánh

1. **Khác thế hệ.** MiniCheck công bố 04/2024 (EMNLP 2024). Lần chạy này (22/09/2026) dùng tên `deepseek-chat`. Theo [changelog chính thức](https://api-docs.deepseek.com/updates/), từ 24/04/2026 tên này trỏ tới **DeepSeek-V4-Flash (non-thinking)** và bị đánh dấu ngưng từ 24/07/2026. Vậy model thực tế ra sau MiniCheck khoảng 2 năm, **không phải DeepSeek-V3** như notebook ghi. Notebook không log trường `model` của response nên chưa xác nhận được 100%.
2. **Khác cách chấm.** DeepSeek đọc nguyên văn bản trong 1 request. Các model mở thì bị cắt văn bản thành chunk 400–500 token rồi lấy max theo chunk. Vì vậy đây là so sánh *hệ thống với hệ thống*, không phải *phương pháp với phương pháp*.
3. **Không tái lập hoàn toàn được.** API đóng và alias đổi model theo thời gian. Ngoài ra LLM-AggreFact công khai từ 2024, nên không loại trừ được khả năng model mới đã thấy dữ liệu này lúc huấn luyện (contamination).
4. **So với số tham chiếu:** GPT-4 (75.64, số tham chiếu trong notebook) thấp hơn 2.1 điểm. Bespoke-7B theo bài báo (77.41) thấp hơn 0.4 điểm, nằm trong nhiễu. Top [leaderboard LLM-AggreFact](https://llm-aggrefact.github.io/) là 77.4. Như vậy DeepSeek **ngang nhóm dẫn đầu**, không bỏ xa.

### 3.3 Chi phí

- Mỗi lần chạy full tốn **≈ 20–23 triệu token input**. Cách ước lượng: văn bản trung bình 447 từ, claim 20 từ, system prompt khoảng 50 từ, nhân 29,320 mẫu, với 1.3–1.5 token/từ. Output khoảng 1 token/mẫu.
- Theo [bảng giá niêm yết](https://api-docs.deepseek.com/quick_start/pricing) ngày 28/09/2026 ($0.15–0.30 / 1M token input cache-miss, tùy khung giờ), một lần chạy full tốn **≈ 2–4 USD**.

- Số tiền nhỏ, nhưng có 3 điểm cần nêu rõ: mỗi lần chạy lại hay làm ablation đều tốn tiền; cần API key và mạng; dữ liệu bị gửi ra bên thứ ba. Ngược lại, FactCG, HalluGuard và MiniCheck đều **mở trọng số và chạy miễn phí** trên GPU Kaggle.

---

## 4. FactCG vs MiniCheck-DeBERTa (cùng kiến trúc, khác checkpoint)

|  | MiniCheck-DeBERTa-v3-Large | FactCG-DeBERTa-v3-Large |  |  |
| --- | --- | --- | --- | --- |
| Backbone | DeBERTa-v3-large (~0.4B) | Giống |  |  |
| Checkpoint | `lytang/MiniCheck-DeBERTa-v3-Large` | `yaxili96/FactCG-DeBERTa-v3-Large` |  |  |
| Dữ liệu fine-tune | ANLI subset + C2D, D2C (dữ liệu tổng hợp của MiniCheck) | Toàn bộ dữ liệu bên trái, **cộng thêm** CG2C-MHQA (8,213 mẫu) và CG2C-Doc (6,433 mẫu): dữ liệu multi-hop sinh từ đồ thị ngữ cảnh |  |  |
| Định dạng input | `doc </s> claim` | Template hỏi đáp của FactCG ("...can we conclude that...? Yes/No") |  |  |
| Chunk | 400 (mặc định MiniCheck) | 500 (bài báo FactCG dùng 550) |  |  |
| Ngưỡng / phần cứng | 0.5 / T4 | 0.5 / T4 |  |  |
| Dataset | MiniCheck-DeBERTa* | FactCG | Δ | Nhiễu ±(95%) |
| --- | :---: | :---: | :---: | :---: |
| AggreFact-CNN | 64.42 | 69.25 | +4.83 | 9.4 |
| Wice | 69.44 | 73.18 | +3.74 | 7.0 |
| Lfqa | 83.93 | 86.51 | **+2.58** | 2.2 |
| TofuEval-MeetB | 72.70 | 74.87 | +2.17 | 5.8 |
| AggreFact-XSum | 71.05 | 73.21 | +2.16 | 5.0 |
| ClaimVerify | 75.54 | 77.36 | +1.82 | 4.1 |
| RAGTruth | 78.79 | 80.11 | +1.32 | 1.7 |
| Reveal | 87.43 | 88.39 | +0.96 | 2.7 |
| ExpertQA | 58.95 | 58.94 | −0.01 | 2.6 |
| TofuEval-MediaS | 69.34 | 68.80 | −0.54 | 5.7 |
| FactCheck-GPT | 73.04 | 72.06 | −0.98 | 3.8 |
| **Macro** | **73.15** | **74.79** | **+1.64** | ≈1.5 |

- **FactCG hơn +1.64 BAcc** với cùng kích thước, cùng phần cứng và cùng chi phí. Chênh lệch này vừa vượt nhiễu. FactCG tăng ở 8/11 dataset. Xét riêng từng tập, chỉ **Lfqa** vượt nhiễu.
- **Khớp bài báo FactCG** (Bảng 10, không tune ngưỡng): MiniCheck-DBT 73.1 → FactCG-DBT 75.6 (+2.5). Số MiniCheck-DeBERTa nhóm chạy gần như trùng bảng này, tức tái lập thành công. FactCG của nhóm thấp hơn bài báo 0.8 điểm. Chunk 500 so với 550 là một khác biệt chưa được kiểm chứng.
- **Wice tăng +3.7**, phù hợp với mục tiêu multi-hop của FactCG. Nhưng Wice chỉ có 358 mẫu (nhiễu ±7), nên chỉ nêu là *xu hướng*, chưa phải kết luận.
- **Hồ sơ lỗi:** FactCG cân bằng nhất nhóm (TPR 73.9 / TNR 75.7). Điểm yếu là bỏ sót claim đúng: TPR ở ExpertQA chỉ 45.5%, FactCheck-GPT 53.5%, Wice 54.1%.
- Cách tune ngưỡng theo từng dataset trên tập dev (bài báo gọi là "dynamic threshold", đạt 77.2) **không áp dụng** ở đây vì cả nhóm thống nhất không tune.

---

## 5. HalluGuard-Qwen3-4B

### Vì sao chỉ đạt 67.15

|  | Run của nhóm | Tác giả ([arXiv 2510.00880 v1](https://arxiv.org/abs/2510.00880)) |
| --- | --- | --- |
| Chế độ suy luận | **Tắt thinking**, trả lời đúng 1 từ YES/NO | Bật thinking, trả nhãn kèm justification |
| Xử lý văn bản | Chunk 500 qua `MiniCheck.score`, lấy max theo chunk | — |
| BAcc toàn benchmark | **67.15** | 75.7 (thinking) / **67.6 (non-thinking)** |

**Kết quả của nhóm khớp ablation non-thinking của chính tác giả** (67.15 so với 67.6). Vậy khoảng cách tới 75.7 chủ yếu do cách chạy (protocol), không phải lỗi code.

**Hành vi: thiên lệch "YES" (dễ dãi)**

- Macro TPR 92.3%, TNR chỉ 42.0%. Model dự đoán "supported" cho 84.6% số mẫu, trong khi tỷ lệ thật là 77.9%.
- Thiên lệch tăng dần theo độ dài văn bản:

| Độ dài văn bản (từ) | < 200 | 200–400 | 400–700 | > 700 |
| --- | --- | --- | --- | --- |
| TNR HalluGuard | 56.4% | 46.7% | 39.6% | 36.4% |
| TNR FactCG (đối chứng) | 87.2% | 80.6% | 77.3% | 69.7% |

- Cách đọc bảng này:
- Ngay với văn bản ngắn (chỉ 1 chunk), TNR đã thấp, nên thiên lệch nằm sẵn trong chế độ non-thinking.
- Văn bản dài còn tệ hơn vì HalluGuard trả điểm cứng 0/1, và MiniCheck lấy max theo chunk. Chỉ cần 1 chunk trả "YES" là cả claim thành "YES".
- Độ dài có tương quan với dataset, nên bảng này chỉ mang tính gợi ý.
- Vì điểm là 0/1 cứng, run này **không có xác suất**, nên không phân tích ngưỡng hay AUROC được.

---

## Nguồn

**Số liệu của nhóm**

- DeepSeek: `test_nb_ver1/deepseek-api-fact-checking-full-benchmark-on-ka.ipynb`. Confusion matrix lấy từ output của cell 18.
- Bespoke-7B: `test_nb_ver1/bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb`
- MiniCheck-DeBERTa: `test_nb_ver1/minicheck-llm-aggrefact-benchmark-evaluation-on-k.ipynb`
- FactCG: `test_nb_factcg/results/full_benchmark_run_v2/` (branch `model/factcg-minicheck-kaggle`). TPR/TNR tính lại từ `predictions_incremental.csv`.
- HalluGuard: `test_nb_halluguard/results/run_v2/` (branch `model/halluguard-minicheck-kaggle`). TPR/TNR và phân tích theo độ dài tính lại từ `predictions_final.csv.gz`.

**Tham chiếu ngoài**

- Tang et al., *MiniCheck*, EMNLP 2024 — [arXiv 2404.10774](https://arxiv.org/abs/2404.10774)
- Lei et al., *FactCG*, NAACL 2025 — [arXiv 2501.17144](https://arxiv.org/abs/2501.17144) (Bảng 3: dynamic threshold; Bảng 10: fixed threshold)
- Bergeron et al., *HalluGuard* — [arXiv 2510.00880 v1](https://arxiv.org/abs/2510.00880) · [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.835.pdf)
- [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/) · [DeepSeek API changelog](https://api-docs.deepseek.com/updates/) · [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
