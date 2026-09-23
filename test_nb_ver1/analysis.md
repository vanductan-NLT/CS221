

Sau khi kiểm tra toàn bộ mã nguồn (`minicheck/inference.py`, `minicheck/minicheck.py`), cấu hình notebook thực nghiệm [`bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb`](file:///c:/Users/racin/Downloads/nlp/result/bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb) và đối chiếu với bài báo gốc EMNLP 2024 (*MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*), kết quả chạy thực tế của bạn đạt **76.05%** so với công bố **77.41%** (chênh lệch **-1.36%**). 


---

### 1. Có phải do chưa Tune ngưỡng (Threshold) như bài báo gốc?

👉 **ĐÚNG, ĐÂY LÀ MỘT YẾUU TỐ CHÍNH VỀ PHƯƠNG PHÁP LUẬN.**

1. **Giao thức đánh giá của bài báo gốc (EMNLP 2024)**:
   * Benchmark `LLM-AggreFact` gồm 11 dataset có **mức độ lệch nhãn (Class Imbalance) cực kỳ nặng**:
     * `RAGTruth`: **92.2%** nhãn 1 (15,102 mẫu Supported vs 1,269 mẫu Unsupported).
     * `AggreFact-CNN`: **89.8%** nhãn 1 (501 mẫu Supported vs 57 mẫu Unsupported).
     * `FactCheck-GPT`: **76.0%** nhãn 0 (1,190 mẫu Unsupported vs 376 mẫu Supported).
     * `Wice`: **69.0%** nhãn 0 (247 mẫu Unsupported vs 111 mẫu Supported).
   * Trong bài báo gốc, để đánh giá công bằng các fact-checker trên chỉ số **Balanced Accuracy** ($BAcc = \frac{\text{Sensitivity} + \text{Specificity}}{2}$), tác giả thực hiện **quét ngưỡng $\theta \in [0.50, 0.95]$ trên tập validation (`dev` split)** cho từng dataset nhằm tìm ngưỡng tối ưu hóa $BAcc$, sau đó mới cố định ngưỡng đó để chấm điểm trên tập `test`.
2. **Cách chạy thực tế trong code hiện tại**:
   * Khi bạn gọi `scorer.score()` (hoặc dùng thư viện `MiniCheck` / Ollama / Guardrails AI), hàm dự đoán đang **áp cứng một ngưỡng cố định duy nhất là 0.5** cho toàn bộ 11 dataset:
     ```python
     pred_label.append(1 if final_score > 0.5 else 0)
     ```
   * Khi cố định ngưỡng 0.5 trên dữ liệu lệch nhãn nặng:
     * Trên **Wice** (tập thiên về nhãn 0): Ngưỡng 0.5 khiến mô hình quá thận trọng, **Recall của nhãn 1 rơi xuống chỉ còn 63.96%** (bỏ sót tới 36.04% số mẫu đúng), kéo $BAcc$ của Wice tụt thảm hại từ **83.00% xuống 76.92% (-6.08%)**.
     * Trên **RAGTruth** (tập 92% nhãn 1): Ngưỡng 0.5 tạo ra tới **2,837 ca False Negative (bỏ sót)**, kéo điểm từ **84.00% xuống 82.33% (-1.67%)**.

---

### 2. Nhưng ngưỡng có phải là nguyên nhân DUY NHẤT? (Thủ phạm lớn hơn)

Nếu chỉ do ngưỡng, mức tụt sẽ diễn ra đồng đều. Tuy nhiên, khi nhìn vào bảng kết quả chi tiết từng dataset trong notebook của bạn:

| Dataset | Bài báo (Paper) | Thực tế (Kaggle 2x T4) | Chênh lệch | Đặc trưng dữ liệu |
| :--- | :---: | :---: | :---: | :--- |
| **AggreFact-CNN** | 65.50% | **69.99%** | **+4.49%** 📈 | Văn bản ngắn, tin tức |
| **FactCheck-GPT** | 77.70% | **77.99%** | **+0.29%** 📈 | Đoạn văn ngắn |
| **Reveal** | 88.00% | **87.82%** | **-0.18%** | Tài liệu vừa |
| **Lfqa** | 86.70% | **86.14%** | **-0.56%** | Q&A dài |
| **ExpertQA** | 59.20% | **57.96%** | **-1.24%** | Đa lĩnh vực chuyên gia |
| **ClaimVerify** | 75.30% | **73.94%** | **-1.36%** | Fact-checking |
| **RAGTruth** | 84.00% | **82.33%** | **-1.67%** | Tài liệu RAG dài |
| **TofuEval-MediaS** | 76.00% | **73.70%** | **-2.30%** 📉 | Bài phỏng vấn rất dài |
| **AggreFact-XSum** | 77.80% | **74.94%** | **-2.86%** 📉 | Tóm tắt trừu tượng |
| **TofuEval-MeetB** | 78.30% | **74.84%** | **-3.46%** 📉 | Biên bản cuộc họp rất dài |
| **Wice** | 83.00% | **76.92%** | **-6.08%** 📉 | Wikipedia đa đoạn |
| **TRUNG BÌNH (AVG)**| **77.41%** | **76.05%** | **-1.36%** | **29,320 mẫu** |

Nhìn vào bảng trên, các dataset tài liệu ngắn thậm chí còn **cao hơn hoặc ngang bằng bài báo**, nhưng các dataset **tài liệu dài và ngữ cảnh phức tạp** (Wice, MeetB, MediaS, XSum, RAGTruth) lại **tụt rất sâu**. 

Nguyên nhân đến từ **3 yếu tố kỹ thuật sau**:

#### A. Thủ phạm ngầm: Cấu hình `CHUNK_SIZE = 500` trong Notebook
* Trong thiết kế gốc của `Bespoke-MiniCheck-7B`, ngữ cảnh mô hình hỗ trợ tới **32,768 tokens**, giá trị `default_chunk_size = 32768 - 300 = 32,468 tokens`. Tức là bài báo gốc **không bao giờ chia nhỏ văn bản** mà đưa toàn bộ tài liệu vào một lần.
* Tuy nhiên, trong notebook Kaggle (Cell 12), để tránh tràn RAM/VRAM trên card T4 16GB, bạn đã cấu hình:
  ```python
  CHUNK_SIZE = 500
  p_labels, r_probs, _, _ = scorer_7b.score(
      docs=b_docs, claims=b_claims, chunk_size=CHUNK_SIZE
  )
  ```
* **Hậu quả khi kết hợp với thuật toán SentenceFusion (`min-max`)**:
  * Mã nguồn `inference.py` tổng hợp điểm qua công thức:
    $$\text{Sentence Score} = \max_{\text{chunks}} P(\text{chunk}, \text{sentence})$$
    $$\text{Final Claim Score} = \min_{\text{sentences}} (\text{Sentence Score})$$
  * Khi băm văn bản thành các đoạn 500 token:
    1. Nếu một claim đòi hỏi thông tin suy luận nằm rải rác ở 2 đoạn (multi-hop reasoning), hoặc câu dẫn chứng bị cắt đôi ngay ranh giới 500 token $\rightarrow$ **Không có chunk nào chứa trọn vẹn bằng chứng**.
    2. Cả 2 chunk đều cho xác suất thấp $\rightarrow \max$ thấp $\rightarrow \min$ sập về sát 0 $\rightarrow$ **Mô hình phán đoán nhầm thành Unsupported (False Negative)**.
* **Minh chứng từ dữ liệu lỗi trong notebook của bạn**:
  * Trong tổng số **6,322 ca đoán sai**, có tới **4,925 ca là False Negative (chiếm 77.9%)**, gấp gần **4 lần** số ca False Positive (1,397 ca)!
  * Cả 3 Case Study mẫu trong Cell 18 đều là ca **False Negative cực đoan** (xác suất mô hình đưa ra chỉ $0.0005$ dù ground truth là $1$).

#### B. Sai số dấu phẩy động phần cứng (Hardware Precision: FP16 vs BF16)
* Nhóm tác giả chạy thực nghiệm trên **NVIDIA A6000 (48GB VRAM)** có Compute Capability 8.6 $\rightarrow$ Code tự động chạy định dạng **`torch.bfloat16`**.
* Bạn chạy trên **Kaggle 2x NVIDIA T4** có Compute Capability 7.5 $\rightarrow$ Code tự động fallback về **`torch.float16`** (`inference.py` dòng 315).
* Kiến trúc LLaMA-3 (nền tảng của Bespoke-7B) nổi tiếng trong cộng đồng mã nguồn mở là rất dễ gặp hiện tượng **activation outliers** gây tràn số hoặc trôi phân phối logit khi ép chạy trên **FP16** thay vì **BF16**.

#### C. Cơ chế trích xuất xác suất `logprobs=5` của vLLM
* Trong `inference.py`, vLLM được cấu hình `logprobs=5`. Hàm `get_support_prob` chỉ quét 5 token có xác suất cao nhất xem có chữ `"Yes"` hay không.
* Nếu ở các ca ranh giới phân vân, token `"Yes"` bị đẩy xuống vị trí thứ 6, hàm sẽ trả về xác suất bằng $0$ (thay vì $0.2 - 0.4$), triệt tiêu khả năng cân chỉnh xác suất mượt mà.

---

### 3. Tổng kết Insight & Hướng biện luận cho Đồ Án / Báo Cáo

Nếu bạn đang làm báo cáo hoặc bảo vệ đồ án, sự chênh lệch **76.05% vs 77.41% (-1.36%)** không hề là "thất bại", mà là **một điểm sáng nghiên cứu (Ablation / Resource Constraint Finding)** rất giá trị:

1. **Khẳng định tính tái lập (Reproducibility)**:
   * Trên môi trường tài nguyên giới hạn (2 card T4 16GB miễn phí của Kaggle so với GPU A6000/A100 cấp trung tâm dữ liệu), bạn đã tái lập được **~98.2% hiệu năng gốc** của mô hình SOTA 7B trên toàn bộ 29,320 mẫu của 11 bộ dữ liệu.
2. **Đóng góp phân tích nguyên nhân (Root-Cause Contribution)**:
   * Bạn đã chỉ ra được sự đánh đổi (**Trade-off**) giữa việc tiết kiệm bộ nhớ (băm nhỏ `chunk_size = 500`) và khả năng hiểu ngữ cảnh văn bản dài:
     * Các tác vụ tin tức ngắn (CNN, FactCheck-GPT) không bị ảnh hưởng, thậm chí tăng điểm.
     * Các tác vụ đa đoạn / hội thoại dài (Wice, MeetB, MediaS) chịu tổn thương nặng nề do cơ chế cắt ngữ cảnh sinh ra tới 77.9% lỗi False Negative.
3. **Hướng khắc phục nếu có thêm tài nguyên (Next Steps)**:
   * **Nới rộng Chunk Size**: Vì bạn đã đặt `max_model_len = 4096`, hoàn toàn có thể tăng `chunk_size` từ 500 lên **2,500 – 3,000 tokens** (vẫn vừa vặn trong VRAM T4 mà bao phủ trọn vẹn 98% độ dài tài liệu của benchmark, không bị xé vụn văn bản).
   * **Tune ngưỡng trên tập Dev**: Áp dụng search threshold $\theta \in [0.3, 0.7]$ trên tập validation của từng dataset trước khi suy luận nhãn, $BAcc$ trung bình sẽ tiệm cận hoặc vượt mức 77.41% của bài báo.