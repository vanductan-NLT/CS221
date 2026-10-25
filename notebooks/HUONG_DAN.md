# Tái lập thực nghiệm "When Not to Trust Language Models" (Adaptive Retrieval) trên Kaggle / Colab free

File chính: **`adaptive_retrieval_repro.ipynb`** (upload lên Kaggle hoặc Colab, chạy từ trên xuống).
`adaptive_retrieval_repro.py` là cùng nội dung ở dạng văn bản thuần (dễ đọc/diff).

## 1. Pipeline làm gì

```
PopQA gốc (GitHub tác giả)  ─┐
Retrieval gốc BM25/Contriever │ (file của tác giả, đọc từ dataset Kaggle bạn upload; không cần dựng index Wikipedia)
                              ▼
   LM (local GPU hoặc API) chạy 4 cách: vanilla | BM25 | contriever | genread
                              ▼
   results/*.csv (lưu theo chunk, tự resume)
                              ▼
   Adaptive Retrieval: ngưỡng popularity theo loại quan hệ, 75/25 split × 100 lần
                              ▼
   Bảng tổng hợp + biểu đồ accuracy-theo-độ-phổ-biến + phân tích "retrieval gây hại"
```

Phần adaptive là **port nguyên văn thuật toán từ notebook của tác giả** và đã được kiểm tra cho ra **cùng kết quả số học** với code gốc trên dữ liệu tổng hợp (khớp đến 10 chữ số). Prompt, few-shot, cách cắt đoạn Contriever, cách chấm điểm giữ đúng như `run_model.py` gốc.

## 2. Chạy từng bước

### Bước 0 — Đưa notebook lên nền tảng
- **Kaggle:** *Create → New Notebook → File → Import Notebook* (chọn file `.ipynb`). Ở *Settings*: **Accelerator = GPU (T4 ×2 hoặc P100)**, **Internet = On** (cần để cài thư viện, tải model HF, tải Drive; Kaggle có thể yêu cầu xác minh số điện thoại để bật Internet).
- **Colab:** *File → Upload notebook*. *Runtime → Change runtime type → T4 GPU*. Giữ `MOUNT_DRIVE = True` để checkpoint lưu vào Google Drive (`MyDrive/adaptive_repro`), tránh mất kết quả khi session hết hạn.

### Bước 0b — Đưa retrieval của tác giả vào Kaggle (thư mục Drive trong README đã mất, trả 404)
Tác giả xác nhận trong [issue #5](https://github.com/AlexTMallen/adaptive-retrieval/issues/5) rằng Drive không còn. Hiện chỉ còn **BM25**: file `PopQA_BM25.jsonl.zip` do tác giả đăng ở issue đó (14282 record, phủ 100% PopQA, mỗi record có `id` trùng `id` PopQA và 5 đoạn). **Contriever không còn bản gốc**, nên mặc định notebook bỏ qua `contriever`.
1. Thư mục sẵn sàng upload: `kaggle_dataset/popqa-bm25-retrieval/` (đã bỏ `__MACOSX`, có `dataset-metadata.json`). Upload bằng `kaggle datasets create -p kaggle_dataset/popqa-bm25-retrieval` hoặc *Create → New Dataset* trên web (đổi `id` trong metadata nếu cần).
2. Trong notebook: *Add Input* → chọn dataset vừa tạo. Notebook tự dò trong `/kaggle/input` các file `.json`/`.jsonl` có `bm25` / `contriever` trong tên file hoặc tên thư mục (bỏ qua file ẩn và `__MACOSX`). Nếu dò sai, đặt `RETRIEVAL_DIR` hoặc `RET_FILES` thủ công.
3. Retrieval được ghép theo `id` PopQA khi file có `id` khớp (tách đúng ~880 nhóm câu hỏi trùng văn bản), nếu không thì ghép theo văn bản câu hỏi. Mục 3 in `khớp theo id N/14267` để kiểm tra.
4. Thiếu contriever: chạy với `METHODS = ["vanilla", "BM25", "genread"]` (hoặc thêm `wiki_intro`, xem mục 3 notebook, khác paper); báo cáo phải nêu rõ không có contriever.

### Bước 1 — Smoke test (bắt buộc làm trước)
Để nguyên `SMOKE_TEST = True` và chạy **tất cả các ô**. Ô này dùng `Qwen/Qwen2.5-0.5B` với 300 câu. Mục đích: kiểm tra tải được dữ liệu, tải được retrieval, GPU chạy được, và **đo tốc độ thật (câu/giây)** của máy bạn — notebook in ETA sau mỗi chunk.

Cần thấy ở các ô đầu:
- `popQA` đã tải (14267 dòng).
- Ô môi trường in `CUDA: True <tên GPU>`, compute capability, VRAM. Nếu thấy `CUDA: False` thì chưa bật GPU (CPU chạy ~0.14 câu/s với model 1.3B).
- Ô nạp model in `dtype yêu cầu`, `dtype tham số thực`, thiết bị và VRAM: đây là chỗ xác nhận model chạy fp16/fp32 thật.
- Mục 3 in ra tên các file retrieval tìm thấy và `BM25: ... phủ ~100% PopQA`. Nếu độ phủ thấp, kiểm tra trường `question` trong file.
- Prompt mẫu ở mục 6 trông đúng dạng `Q: ... A:`.

### Bước 2 — Chạy thật
Đặt `SMOKE_TEST = False`, chọn:
- `MODEL_NAME`, `QUANT` (xem bảng dưới)
- `DTYPE`: `"auto"` (mặc định: GPU → fp16, CPU → fp32), hoặc `"fp16"` / `"fp32"` / `"bf16"`. `bf16` cần GPU compute capability ≥ 8.0 (T4, P100 không có), nếu không đạt notebook in cảnh báo và dùng fp16. dtype thật nằm trong tên file kết quả (vd. `..._gpt-neo-1.3B__fp16__vanilla__shots15.csv`) nên các lần chạy fp32/fp16 không bị trộn khi resume.
- `N_QUESTIONS` (3000 là điểm bắt đầu hợp lý; `0` = toàn bộ ~14k)
- `METHODS` (thứ tự khuyến nghị: vanilla → BM25 → contriever → genread)

Rồi chạy lại từ đầu. Nếu session bị ngắt: **chạy lại y nguyên**, notebook bỏ qua các câu đã xong.

### Bước 3 — Đọc kết quả
- Bảng `combo / parametric / retrieval / adaptive / gain vs best / frac. retrieving`: so sánh độ chính xác của LM thuần, LM + retrieval, và hệ adaptive; `frac. retrieving` là tỉ lệ câu thực sự phải truy xuất (liên quan chi phí).
- Bảng ngưỡng theo từng loại quan hệ (`retrieve_if_s_pop_below`).
- Biểu đồ `figs/acc_vs_pop__<model>.png` (tương tự Hình 8 của paper).
- Bảng "retrieval giúp hay hại" (tương tự Bảng 1 của paper).

### Bước 4 — Mở rộng
Đổi `MODEL_NAME` rồi chạy lại để có thêm model (kết quả mỗi model lưu file riêng). So sánh nhiều model để dựng lại ý chính của paper: scaling không giúp phần long-tail, và model lớn cần retrieval ít hơn.

## 3. Chọn model theo GPU free (T4 ~15GB)

| Mục tiêu | `MODEL_NAME` | `QUANT` |
|---|---|---|
| Smoke test | `Qwen/Qwen2.5-0.5B` | `none` |
| Nhanh | `Qwen/Qwen2.5-1.5B`, `EleutherAI/gpt-neo-1.3B`, `facebook/opt-1.3b` | `none` |
| Gần paper | `EleutherAI/gpt-neo-2.7B`, `facebook/opt-2.7b` | `none` |
| Model lớn | `Qwen/Qwen2.5-7B`, `EleutherAI/gpt-j-6b` | `4bit` |

- Dùng **base model**, không dùng bản instruct.
- Model đời mới đã học nhiều dữ liệu hơn nên đường cong và ngưỡng sẽ khác paper (điều này bình thường, không phải lỗi).
- Nếu hết VRAM, notebook tự giảm batch; bạn cũng có thể giảm `BATCH_SIZE`.

**Ước lượng chi phí tính toán (tự ước, chưa đo trên GPU thật):** prompt 15-shot của `vanilla` khoảng vài trăm token, còn các method có retrieval mỗi ví dụ few-shot kèm một đoạn văn nên prompt dài hơn nhiều (có thể chạm giới hạn 2048 token và bị cắt từ bên trái, giống paper). Vì vậy `BM25`/`contriever` chậm hơn `vanilla` đáng kể, và `genread` chậm nhất (gọi model 2 lần/câu, sinh thêm 150 token). Hãy dựa vào ETA ở smoke test thay vì con số này.

**Giới hạn nền tảng:** Kaggle và Colab free đều giới hạn giờ GPU và thời lượng session (Colab có thể ngắt bất kỳ lúc nào). Hãy kiểm tra hạn mức hiện hành của tài khoản bạn. Trên Kaggle, dùng *Save Version → Save & Run All (Commit)* để chạy nền; kết quả chỉ được giữ khi commit. Có thể đưa output của lần chạy trước làm Input để chạy tiếp (copy các file `results/*.csv` vào lại thư mục làm việc).

## 4. Dùng API thay vì model local

Đặt `BACKEND = "api"` rồi chọn:
- `API_PROVIDER = "openai"` (mọi dịch vụ tương thích OpenAI, đặt thêm `API_BASE_URL`) hoặc `"anthropic"`.
- `API_MODEL = "<tên model>"`. Dùng model **không phải reasoning** (reasoning model dùng hết `max_tokens` cho phần suy luận nên trả về rỗng).
- Khoá API: lưu vào **Kaggle Secrets / Colab Secrets** (hoặc biến môi trường) với tên đúng như `API_KEY_NAME`. **Không dán khoá vào notebook.**

Mặc định API chạy 0-shot (như paper với GPT-3). Notebook có ô ước lượng số token đầu vào để bạn tự tính chi phí trước khi chạy. Kết quả API được cache trên đĩa nên chạy lại không tốn thêm. Model API mạnh sẽ nhớ nhiều kiến thức hơn nên hiệu ứng "popularity" thể hiện khác paper.

## 5. Khác biệt so với code gốc (có chủ đích)

| Điểm | Code gốc | Notebook này | Lý do |
|---|---|---|---|
| Nạp model | fp32 lên RAM rồi chuyển GPU | theo `DTYPE` (GPU: fp16, CPU: fp32) hoặc 4-bit, nạp trực tiếp | vừa RAM + VRAM của free tier |
| Sinh văn bản | từng câu một | theo batch, sắp theo độ dài | nhanh hơn nhiều |
| Few-shot | ngẫu nhiên mỗi lần chạy | cố định theo `id` câu hỏi | resume cho prompt giống hệt, các method so sánh cặp công bằng |
| Ghép kết quả | sắp xếp theo `question` | ghép theo `id` | PopQA có ~1.2k câu hỏi trùng văn bản |
| API | `openai.Completion` + `text-davinci-*` | chat API tương thích OpenAI / Anthropic | các model cũ đã ngừng |
| Bước adaptive | — | cùng thuật toán; thêm bỏ qua loại quan hệ <8 mẫu | tránh lỗi chia 0 khi lấy mẫu con |

## 6. Đã kiểm chứng gì, chưa kiểm chứng gì

Đã chạy thử trong môi trường dựng sẵn (CPU, model GPT-2 nhỏ ngẫu nhiên, dữ liệu retrieval giả đúng định dạng JSONL/JSON, máy chủ API giả):
- Chạy trọn pipeline 4 method; resume khi bị cắt giữa chừng; backend HF, OpenAI-compatible, Anthropic; tự xử lý khi API từ chối `max_tokens`/`temperature`.
- Bước adaptive **khớp số học** với notebook gốc của tác giả.
- Biểu đồ và các bảng phân tích hiển thị đúng.

**Chưa kiểm chứng:**
- Chạy trên **GPU thật** và với **model thật** (không có GPU/HF trong môi trường kiểm thử). Hãy dùng smoke test để xác nhận.
- Tải **thư mục Google Drive của tác giả**: môi trường kiểm thử không truy cập được Drive nên không biết tên/định dạng file thật. Notebook tự dò (JSONL/JSON, trường `question` và `ctxs[0].text/id/hasanswer`) và in cảnh báo nếu lệch.
- Phương án dự phòng `wiki_intro` (lấy đoạn mở đầu Wikipedia): chưa chạy thử được vì Wikipedia bị chặn ở môi trường kiểm thử.

## 7. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Mục 3 báo lỗi Drive (404) / không thấy file retrieval | Làm Bước 0b (dataset Kaggle + Add Input), rồi chạy lại ô; hoặc đặt `RETRIEVAL_DIR` / `RET_FILES` thủ công |
| `CUDA: False` / chạy rất chậm | Kaggle: Settings → Accelerator → GPU T4/P100, chạy lại từ ô đầu |
| `phủ` thấp (<95%) | Khoá `question` trong file retrieval khác văn bản câu hỏi PopQA; kiểm tra bằng ví dụ in ra |
| CUDA out of memory | Giảm `BATCH_SIZE`, giảm `MAX_CTX`, hoặc dùng `QUANT="4bit"` |
| 4-bit báo lỗi bitsandbytes | Khởi động lại runtime sau khi cài thư viện (ô `%pip`), rồi chạy lại từ ô đầu |
| Output toàn rỗng / sai bất thường ở model Qwen fp16 | Thử `DTYPE="fp32"` (nếu vừa VRAM), `QUANT="4bit"` hoặc một model khác (fp16 đôi khi gây tràn số ở một số model) |
| Bước adaptive báo "quá ít mẫu" | Tăng `N_QUESTIONS` (nên ≥3000) |
| API trả lỗi tham số | Notebook tự thử `max_completion_tokens` và bỏ `temperature`; nếu vẫn lỗi, kiểm tra tên model và dùng model không phải reasoning |
