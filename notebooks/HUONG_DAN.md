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

## 8. Đối chiếu với code gốc: cùng 300 câu

Để biết notebook của nhóm có làm đúng như code tác giả không, ta chạy cả hai trên **cùng 300 câu** (`data/subset_300_ids.txt`) rồi so (issue #10 và #11). Cùng một tập câu để chênh lệch còn lại chỉ do khác biệt code, không phải do bốc mẫu. Danh sách 300 id do `scripts/make_subset.py` tạo (seed 633, cùng cách xáo trộn với notebook chính).

### 8.1. Notebook của nhóm chạy đúng 300 câu
Trong ô cấu hình của `adaptive_retrieval_repro.ipynb`, đặt `SUBSET_IDS` là đường dẫn file hoặc URL (đặt thì `N_QUESTIONS` bị bỏ qua):

```python
SUBSET_IDS = "https://raw.githubusercontent.com/vanductan-NLT/CS221/main/data/subset_300_ids.txt"
```

URL trên chỉ có sau khi file đã nằm trên nhánh `main`; trước đó dùng nhánh đang làm thay cho `main`, hoặc upload file lên Kaggle rồi đặt đường dẫn. Id bị trùng hoặc không có trong PopQA thì notebook dừng và báo rõ.

### 8.2. Code gốc: `notebooks/upstream_kaggle.ipynb`
1. Kaggle → *Create → New Notebook → File → Import Notebook* → chọn `upstream_kaggle.ipynb`.
2. *Settings*: **Accelerator = GPU** (T4 ×2 hoặc P100), **Internet = On**. *Add Input* → dataset BM25 (`popqa-bm25-retrieval`).
3. Ô cấu hình: mặc định GPT-Neo 1.3B, `vanilla` + `BM25`, 300 câu, 15 ví dụ mẫu. Nếu PR chưa merge vào `main`, đặt `REPO_REF` là tên nhánh của PR. Repo phải public thì notebook mới clone được.
4. *Run All*. Notebook clone repo, chép `upstream/` ra thư mục làm việc, chạy `run_model.py` (không sửa) hai lần, rồi **kiểm id kết quả có đúng 300 id của `subset_300_ids.txt` không**.
5. Cần thấy dòng `ĐÚNG: cả 2 kết quả khớp đúng 300 id ...` rồi mới tới bảng độ chính xác. Dòng `KẾT QUẢ KHÔNG KHỚP ...` nghĩa là không dùng số đó để đối chứng.
6. *Save Version → Save & Run All (Commit)* để giữ thư mục `upstream_results/`. Ô cuối in sẵn một dòng gợi ý để ghi vào `results/RUNS.md` (điền thêm ngày và link phiên bản notebook).

Danh sách những chỗ phải lách để không sửa `run_model.py` (cài `jsonlines`/`openai`, `--ret_path`, `--sample 300` trên file đầy đủ, chạy trên bản sao của `upstream/`) nằm ở đầu notebook. Chưa có số đo tốc độ của code gốc trên GPU thật; số này sẽ có sau lần chạy đầu tiên.

| Triệu chứng | Cách xử lý |
|---|---|
| `Chưa bật GPU` | *Settings → Accelerator → GPU*, chạy lại từ ô đầu |
| `git clone` lỗi | Bật *Internet*; kiểm tra repo còn public và `REPO_REF` đúng tên nhánh |
| `Không thấy file BM25 .jsonl` | *Add Input* dataset BM25, hoặc đặt `RET_PATH` thủ công |
| `KẾT QUẢ KHÔNG KHỚP data/subset_300_ids.txt` | Xem cột `thieu`/`thua`. Thường do phiên bản thư viện đổi cách dùng bộ sinh số ngẫu nhiên; dán bảng kiểm vào issue #10 để xử lý, đừng dùng số đó |

## 9. GPT-Neo 1.3B/2.7B, BM25 có đoạn văn trong ví dụ mẫu, đủ 14.267 câu

`gpt_neo_bm25_shot_passages_phien1.ipynb` là bản tải từ Kaggle **kèm output và log** của phiên 1 ([phiên bản 4](https://www.kaggle.com/code/vanductan/cs221-popqa-gpt-neo-bm25-shot-passages?scriptVersionId=366304010)). Mở trên GitHub là đọc được log, không cần chạy lại. Phiên 2 ở `gpt_neo_bm25_shot_passages_phien2.ipynb` ([phiên bản 5](https://www.kaggle.com/code/vanductan/cs221-popqa-gpt-neo-bm25-shot-passages?scriptVersionId=366459037)).

- Chạy code gốc `run_model.py` (fp32), chỉ sửa chỗ tra đoạn văn cho 15 ví dụ mẫu, và đặt lại seed 633 trước khi chọn câu (dòng 9 của `docs/KHAC_BIET.md`).
- Hai mô hình chạy song song, mỗi mô hình một GPU T4. Phiên tự dừng sau 11,4 giờ, lưu checkpoint; phiên sau chạy tiếp từ checkpoint.
- Phiên 1: 1.3B xong đủ 14.267 câu; 2.7B dừng ở 8.029 câu.
- Phiên 2: 2.7B chạy tiếp từ checkpoint, xong đủ 14.267 câu; 1.3B đã có CSV nên bỏ qua. Số liệu và thời gian: `results/RUNS.md`.

## 10. Vanilla GPT-Neo 1.3B/2.7B và OPT 1.3B (vanilla + BM25), có cột `conf`

`gpt_neo_vanilla_opt_1_3b.ipynb` (Kaggle: [cs221-popqa-gpt-neo-vanilla-opt-1-3b](https://www.kaggle.com/code/vanductan/cs221-popqa-gpt-neo-vanilla-opt-1-3b)) là notebook ở mục 9, mở rộng thành nhiều lượt chạy:

- Danh sách lượt ở biến `JOBS` (mô hình × cách × GPU). Mỗi GPU một hàng đợi, hai GPU chạy song song. Mặc định: GPU 0 chạy OPT 1.3B BM25; GPU 1 chạy lần lượt vanilla GPT-Neo 2.7B, GPT-Neo 1.3B, OPT 1.3B.
- Cùng code gốc fp32, cùng seed 633 và cùng thứ tự câu với mục 9, nên vanilla và BM25 của một mô hình có cùng câu hỏi và cùng 15 ví dụ mẫu.
- Thêm cột `conf` (issue #14, định nghĩa ở `results/README.md`). Đã thử trên CPU với GPT-Neo và OPT tí hon: câu trả lời giống hệt bản không có `conf`; chạy tiếp từ checkpoint cho cùng kết quả.
- Trước khi chạy đủ, notebook chạy thử 300 câu cho mọi lượt, kiểm đúng 300 ID theo thứ tự và dựng lại từng prompt.
- Muốn chạy mô hình khác (ví dụ OPT 2.7B, Qwen2.5): sửa `JOBS` ở cell "Hàm chạy", giữ nguyên phần còn lại.
- `gpt_neo_vanilla_opt_1_3b_phien1.ipynb`: bản đã chạy trên Kaggle ([phiên bản 1](https://www.kaggle.com/code/vanductan/cs221-popqa-gpt-neo-vanilla-opt-1-3b?scriptVersionId=366470167)), kèm output và log. OPT 1.3B vanilla dừng ở 8.332 câu; chạy tiếp ở notebook Kaggle `cs221-popqa-opt-1-3b-2-7b` (cùng code, `JOBS` mới: OPT 1.3B vanilla và OPT 2.7B). Số liệu và thời gian: `results/RUNS.md`.
- `opt_1_3b_2_7b_phien1.ipynb`: phiên tiếp theo ([phiên bản 1](https://www.kaggle.com/code/vanductan/cs221-popqa-opt-1-3b-2-7b?scriptVersionId=366647869)), kèm output và log: chạy nốt OPT 1.3B vanilla, chạy OPT 2.7B vanilla và 8.086 câu đầu của OPT 2.7B BM25. Chỉ khác notebook trên ở `JOBS` và giới hạn giờ, nên nhận checkpoint của phiên trước.
- `opt_2_7b_bm25_chay_tiep.ipynb`: chạy nốt 6.181 câu còn lại của OPT 2.7B BM25 ([phiên bản 1](https://www.kaggle.com/code/danielndap/cs221-popqa-opt-2-7b-bm25-chay-tiep?scriptVersionId=366779792), tài khoản bạn Tân), kèm output và log. Chỉ khác `gpt_neo_vanilla_opt_1_3b.ipynb` ở `JOBS` và giới hạn giờ, nên nhận checkpoint của phiên trước.

## 11. Qwen2.5 0.5B/1.5B (bản base), vanilla và BM25 (phần mở rộng, issue #30)

`qwen2_5_0_5b_1_5b.ipynb` là bản đã chạy trên Kaggle ([phiên bản 1](https://www.kaggle.com/code/chanejlaforet/cs221-popqa-qwen2-5-0-5b-1-5b?scriptVersionId=366477306)), kèm output và log. Ô đầu notebook là hướng dẫn 4 bước cho người chạy nhờ.

- Cùng notebook ở mục 10, khác ba chỗ: `JOBS` là bốn lượt Qwen; mã nguồn tác giả và PopQA tải từ GitHub (commit `8172ab0`, kiểm hash) thay vì dataset của nhóm; Qwen được cắt prompt ở 2048 token như OPT (dòng 10 của `docs/KHAC_BIET.md`).
- Chỉ cần Add Input dataset `vanductan/popqa-bm25-retrieval` (công khai).
- Qwen2.5-3B: cùng notebook, `JOBS` đổi thành 3B BM25 (GPU 0) và 3B vanilla (GPU 1), chạy ở [cs221-popqa-qwen2-5-3b](https://www.kaggle.com/code/chanejlaforet/cs221-popqa-qwen2-5-3b). Vanilla xong trong phiên 1; BM25 cần 3 phiên (5,2–5,4 giây/câu). `qwen2_5_3b_phien2.ipynb` là phiên 2 ([phiên bản](https://www.kaggle.com/code/chanejlaforet/cs221-popqa-qwen2-5-3b?scriptVersionId=366806767)), kèm output và log: nhận checkpoint phiên 1, chạy BM25 từ câu 7.608 đến 12.978. Phiên sau: Add Input file `popqa-qwen-checkpoint.zip` của phiên trước, đặt `TIME_BUDGET_HOURS` nhỏ hơn quota còn lại khoảng 0,7 giờ, rồi chạy lại toàn bộ notebook.
