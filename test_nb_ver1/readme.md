# 🧪 MiniCheck Benchmark - Thực Nghiệm Tái Lập (Version 1)

Thư mục này chứa toàn bộ các Jupyter Notebook và tài liệu phân tích nhanh trong giai đoạn thực nghiệm đầu tiên (**Version 1**) nhằm tái lập (reproduce) và đối chiếu kết quả của bài báo EMNLP 2024:
> **"MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents"**  
> *Liyan Tang, Philippe Laban, Greg Durrett (EMNLP 2024)*  
> Benchmark: `LLM-AggreFact` (11 datasets, ~29,320 mẫu kiểm định fact-checking).

---

## 📌 1. Tại Sao Lại Là "Version 1" (`test_nb_ver1`)?

Thư mục được đánh dấu là **`ver 1`** với các lý do cụ thể sau:
* **Trình bày còn dài và chi tiết**: Các notebook trong phiên bản này được tạo ra trong giai đoạn thử nghiệm ban đầu nên còn lưu giữ toàn bộ log in kiểm thử, các hàm phụ trợ debug, và các bước phân tích trung gian.
* **Chạy song song nên so sánh phân tán**: Do thời gian gấp rút, việc chạy thực nghiệm được tiến hành song song giữa file mô hình `Bespoke` và file các mô hình `MiniCheck` nhỏ, dẫn đến việc các notebook chủ yếu đối chiếu số liệu riêng lẻ với kết quả trong bài báo gốc chứ chưa được tổng hợp thành một bảng so sánh chung thống nhất.
* **Số liệu trực quan hóa chưa tối ưu**: Các biểu đồ và số liệu phân tích lỗi còn hơi rối và nhiều thông tin thô.
* **Mục đích lưu trữ**: Phiên bản này đóng vai trò là **bản ghi thực nghiệm gốc (Raw Baseline)** để bảo toàn toàn bộ mã nguồn và checkpoint ban đầu trước khi tinh gọn, làm sạch code và chuẩn hóa giao diện báo cáo ở các phiên bản tiếp theo.

---

## 📂 2. Cấu Trúc Thư Mục & Các Notebook Thực Nghiệm

```
test_nb_ver1/
├── bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb   # Thực nghiệm Bespoke-MiniCheck-7B (vLLM)
├── minicheck-llm-aggrefact-benchmark-evaluation-on-k.ipynb  # Thực nghiệm các mô hình nhỏ (Encoder/Seq2Seq)
├── deepseek-api-fact-checking-full-benchmark-on-ka.ipynb    # Thực nghiệm nhánh LLM-as-a-Judge (DeepSeek API)
├── analysis.md                                              # Phân tích NHANH nguyên nhân sai lệch kết quả
└── readme.md                                                # Tài liệu hướng dẫn & tổng quan này
```

---

## 🔬 3. Chi Tiết Các Notebook Thực Nghiệm

### 3.1. `bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb`
* **Mô hình**: `Bespoke-MiniCheck-7B` (xây dựng dựa trên nền tảng LLaMA-3-8B được fine-tune chuyên biệt cho tác vụ fact-checking).
* **Kiến trúc & Engine**: Tích hợp vLLM engine để tối ưu hóa throughput, tận dụng song song 2x GPU NVIDIA T4.
* **Mục tiêu**: Đánh giá toàn diện 11 dataset trên benchmark `LLM-AggreFact`, so sánh độ chính xác cân bằng (Balanced Accuracy) với công bố của nhóm tác giả bài báo gốc.

### 3.2. `minicheck-llm-aggrefact-benchmark-evaluation-on-k.ipynb`
* **Mô hình**: Nhánh các mô hình ngôn ngữ nhỏ (Small Language Models / Encoder-based / Seq2Seq) thuộc framework MiniCheck:
  * `flan-t5-large`
  * `deberta-v3-large`
  * `roberta-large`
* **Đặc điểm**: Kích thước nhỏ gọn, tiêu thụ ít tài nguyên, tốc độ suy luận nhanh trên GPU đơn/kép.
* **Mục tiêu**: Đánh giá độ hiệu quả và sự đánh đổi (trade-off) giữa tài nguyên tính toán và độ chính xác kiểm định so với các mô hình 7B.

### 3.3. `deepseek-api-fact-checking-full-benchmark-on-ka.ipynb`
* **Nhánh tiếp cận**: **LLM-as-a-Judge** (Prompt-based Fact-Checking).
* **Lý do chọn DeepSeek thay vì GPT-4**:
  * Bài báo gốc sử dụng `GPT-4` (`gpt-4-0613` / `gpt-4-turbo`) làm đại diện cho nhóm LLM lớn đóng vai trò giám khảo.
  * Tuy nhiên, hiện tại các phiên bản API cũ của GPT-4 khó truy cập lại đúng bản gốc, đồng thời chi phí API để chạy toàn bộ 29,320 mẫu là cực kỳ đắt đỏ.
  * Nhóm quyết định sử dụng **DeepSeek API (`deepseek-chat` / DeepSeek-V3)**: Hiệu năng lý luận mạnh mẽ, chi phí hợp lý, đại diện xuất sắc cho nhánh LLM-as-a-judge trong kỷ nguyên mã nguồn mở mới.
* **Kỹ thuật triển khai**:
  * Sử dụng prompt chuẩn định dạng zero-shot fact-checking của bài báo MiniCheck.
  * Xử lý đa luồng (`MAX_WORKERS=10`), cơ chế Exponential Backoff retry để chống rate limit.
  * Thu thập xác suất thông qua `logprobs` của API.

---

## ⚡ 4. Lưu Ý Quan Trọng Khi Chạy Trên Kaggle

Toàn bộ các notebook trong thư mục này được thiết kế riêng để chạy trên môi trường **Kaggle Notebooks (2x NVIDIA T4 16GB)**. Do đó, cú pháp và cấu trúc code có một số điểm khác biệt so với môi trường server chuẩn của bài báo gốc:

1. **Quản lý bộ nhớ chống tràn VRAM (OOM - Out Of Memory)**:
   * Kích hoạt cờ bộ nhớ CUDA:
     ```python
     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
     ```
   * Chia nhỏ quá trình xử lý theo batch và sub-batch (`batch_size=8`, `sub_batch_size=100`).
   * Chủ động thu hồi rác và giải phóng cache GPU sau mỗi sub-batch:
     ```python
     gc.collect()
     torch.cuda.empty_cache()
     ```
2. **Cấu hình Chunking rút gọn**:
   * Để vLLM và các mô hình không bị OOM khi xử lý các văn bản dài (như Wice, TofuEval), tham số `CHUNK_SIZE` trong code được hạ xuống **`500`** (thay vì giữ nguyên toàn bộ context 32k tokens như bài báo).
3. **Cơ chế lưu trữ Checkpoint (`/kaggle/working`)**:
   * Do giới hạn session của Kaggle (tối đa 9 - 12 tiếng hoặc có thể bị gián đoạn kết nối), code tự động lưu kết quả trung gian sau từng dataset vào `/kaggle/working/` để có thể resume mà không phải chạy lại từ đầu.

---

## 📊 5. Phân Tích Kết Quả & Tài Liệu `analysis.md`

File [`analysis.md`](file:///d:/CS221/CS221/test_nb_ver1/analysis.md) trong thư mục này là bản **phân tích NHANH** đối với 2 file thực nghiệm (Bespoke và nhóm mô hình nhỏ) nhằm giải thích nguyên nhân tại sao kết quả thực tế đạt **76.05%** (thấp hơn ~1.36% so với con số **77.41%** của bài báo):

* **Nguyên nhân chính 1 (Ngưỡng phân loại)**: Mã nguồn thực nghiệm đang áp cứng ngưỡng cố định `threshold = 0.5` cho toàn bộ 11 dataset, trong khi bài báo gốc thực hiện tune ngưỡng tối ưu $\theta \in [0.5, 0.95]$ trên tập validation (dev split) của từng dataset để xử lý hiện tượng mất cân bằng nhãn nặng.
* **Nguyên nhân chính 2 (Cắt nhỏ ngữ cảnh - Chunk Size 500)**: Việc giảm `CHUNK_SIZE = 500` để tránh OOM trên Kaggle đã làm đứt gãy các câu dẫn chứng nằm ở ranh giới đoạn hoặc các suy luận bắc cầu (multi-hop). Kết hợp với thuật toán `min-max` của SentenceFusion, tỷ lệ **False Negative (đoán nhầm thành Unsupported)** bị đẩy lên tới **77.9%** trên các dataset văn bản dài.
* **Nguyên nhân phụ khác**: Sai lệch phần cứng do T4 tự động fallback về **FP16** thay vì **BF16** (trên GPU A6000 của tác giả), và giới hạn `logprobs=5` của vLLM.

> 📝 **Lưu ý**: Bản phân tích trong `analysis.md` hiện tại là phân tích sơ bộ nhanh để nắm bắt nguyên nhân kỹ thuật. **Kết quả phân tích chuyên sâu, toàn diện và đầy đủ biểu đồ sẽ được Thịnh cập nhật sau.**

---

## 🚀 6. Kế Hoạch Cho Các Phiên Bản Tiếp Theo (Ver 2+)

* [ ] Refactor và gộp chung pipeline so sánh các mô hình vào một notebook tinh gọn.
* [ ] Cập nhật báo cáo phân tích lỗi chuyên sâu chi tiết.
* [ ] Chạy thêm model mới, dựa trên notebook cũ.
