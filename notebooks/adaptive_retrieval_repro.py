# %% [markdown]
# # Tái lập thực nghiệm "When Not to Trust Language Models" (Mallen et al., ACL 2023)
# **Chạy được trên Kaggle (GPU T4/P100) hoặc Colab free.** Pipeline đầy đủ:
#
# 1. Tải PopQA gốc của tác giả (14k câu hỏi + độ phổ biến `s_pop`).
# 2. Tải kết quả retrieval (BM25, Contriever) mà tác giả đã chạy sẵn (Google Drive) — không cần dựng index Wikipedia.
# 3. Chạy LM với 4 cách: `vanilla`, `BM25`, `contriever`, `genread` (model local trên GPU hoặc qua API).
# 4. Chạy **Adaptive Retrieval** (chọn ngưỡng popularity theo từng loại quan hệ, 75/25 split × 100 lần — port đúng từ notebook gốc).
# 5. Vẽ biểu đồ độ chính xác theo độ phổ biến, bảng tổng hợp, phân tích "retrieval gây hại".
#
# **Cách dùng:** chạy lần lượt từng ô từ trên xuống. Để `SMOKE_TEST = True` ở lần đầu (model nhỏ, ~300 câu) để kiểm tra mọi thứ chạy được,
# sau đó đặt `SMOKE_TEST = False` và chỉnh `MODEL_NAME`, `N_QUESTIONS`. Mọi kết quả được lưu theo từng chunk và **tự chạy tiếp (resume)** nếu session bị ngắt.

# %% [code] tags=install
%pip install -q -U transformers accelerate bitsandbytes gdown jsonlines scipy seaborn openai anthropic requests

# %% [markdown]
# ## 0. Môi trường và đường dẫn lưu
# - **Kaggle:** bật *Internet* và chọn *GPU* trong Settings. Kết quả lưu ở `/kaggle/working` (chỉ giữ lại khi bạn *Save Version → Save & Run All*).
# - **Colab:** nên mount Google Drive để checkpoint không mất khi session hết hạn (`MOUNT_DRIVE = True`).

# %% [code]
import os, sys, json, re, time, math, random, glob, hashlib, threading, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

MOUNT_DRIVE = True   # chỉ áp dụng cho Colab

IN_KAGGLE = Path("/kaggle/working").exists()
IN_COLAB = "google.colab" in sys.modules
if IN_KAGGLE:
    WORK = Path("/kaggle/working/adaptive_repro")
elif IN_COLAB:
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = Path("/content/drive/MyDrive/adaptive_repro")
    else:
        WORK = Path("/content/adaptive_repro")
else:
    WORK = Path("./adaptive_repro")
if os.environ.get("ADAPTIVE_WORK"):          # dùng cho test cục bộ
    WORK = Path(os.environ["ADAPTIVE_WORK"])
for sub in ["data", "retrieval", "results", "figs", "cache"]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)
print("WORK =", WORK)

import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(không có GPU)")
if torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability(0)
    print(f"  {torch.cuda.device_count()} GPU | compute capability {_cc[0]}.{_cc[1]} | VRAM {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB | bf16 gốc: {_cc[0] >= 8}")
else:
    print("!! Không có GPU: model sẽ chạy trên CPU (~0.14 câu/s với 1.3B). Kaggle: Settings -> Accelerator -> GPU.")

# %% [markdown]
# ## 1. Cấu hình
# **Gợi ý chọn model cho GPU free (T4 ~15GB):**
#
# | Mục tiêu | `MODEL_NAME` | `QUANT` |
# |---|---|---|
# | Smoke test | `Qwen/Qwen2.5-0.5B` | `none` |
# | Model nhỏ (nhanh) | `Qwen/Qwen2.5-1.5B`, `EleutherAI/gpt-neo-1.3B`, `facebook/opt-1.3b` | `none` |
# | Model vừa (giống paper hơn) | `EleutherAI/gpt-neo-2.7B`, `facebook/opt-2.7b` | `none` |
# | Model lớn (cần để thấy rõ lợi ích adaptive) | `Qwen/Qwen2.5-7B`, `EleutherAI/gpt-j-6b` | `4bit` |
#
# Dùng **base model** (không phải bản instruct) vì prompt kiểu `Q: ... A:` được thiết kế cho base LM. Các model đời mới được huấn luyện trên nhiều dữ liệu hơn nên "ngưỡng phổ biến" sẽ dịch chuyển so với paper — đó là kết quả bình thường.
#
# **Chế độ API:** đặt `BACKEND = "api"`, `API_PROVIDER = "openai"` (mọi dịch vụ tương thích OpenAI qua `API_BASE_URL`) hoặc `"anthropic"`, rồi điền `API_MODEL`. Dùng model **không phải reasoning**. Khoá API lấy từ Colab/Kaggle Secrets hoặc biến môi trường — đừng dán thẳng vào notebook.

# %% [code] tags=config
SMOKE_TEST   = True            # True: chạy thử nhanh. False: chạy thật.
BACKEND      = "hf"            # "hf" (model local trên GPU) | "api"
METHODS      = ["vanilla", "BM25", "contriever", "genread"]   # thêm "wiki_intro" nếu dùng fallback Wikipedia (xem mục 3)

# --- backend "hf" ---
MODEL_NAME   = "Qwen/Qwen2.5-1.5B"
QUANT        = "none"          # "none" (không lượng tử hoá, dùng DTYPE) | "4bit" | "8bit"
DTYPE        = "auto"          # "auto" (GPU: fp16, CPU: fp32) | "fp16" | "bf16" (cần GPU Ampere+, T4/P100 không có) | "fp32"
GPU_INDEX    = 0
BATCH_SIZE   = 8               # tự giảm một nửa nếu hết VRAM
MAX_CTX      = 2048            # cắt prompt từ bên trái như paper (OPT/GPT-Neo đều 2048)

# --- backend "api" ---
API_PROVIDER = "openai"        # "openai" (tương thích OpenAI) | "anthropic"
API_MODEL    = ""              # ví dụ Anthropic: "claude-haiku-5-5"; OpenAI-compatible: tên model của dịch vụ bạn dùng
API_BASE_URL = None            # None = OpenAI chính chủ; hoặc URL của dịch vụ tương thích OpenAI
API_KEY_NAME = "OPENAI_API_KEY"   # tên secret/biến môi trường (Anthropic: "ANTHROPIC_API_KEY")
API_WORKERS  = 4

# --- thực nghiệm ---
RETRIEVAL_DIR = None           # thư mục chứa file BM25/contriever (vd. "/kaggle/input/<tên-dataset>"); None = tự dò trong /kaggle/input và WORK/retrieval
N_QUESTIONS  = 3000            # 0 = toàn bộ ~14k. Bước adaptive cần đủ mẫu mỗi loại quan hệ: nên >= 3000, tốt nhất toàn bộ.
SUBSET_IDS   = None            # file id cố định, mỗi dòng một id (đường dẫn hoặc URL), vd. "data/subset_300_ids.txt". Đặt thì bỏ qua N_QUESTIONS. None = lấy N_QUESTIONS câu đầu của hoán vị seed 633.
N_SHOTS      = None            # None = tự chọn (hf: 15-shot như paper cho Neo/OPT; api: 0-shot như paper cho GPT-3)
MAX_NEW_TOKENS = 15
GENREAD_TOKENS = 150
CHUNK        = 256             # lưu checkpoint sau mỗi chunk
SEED         = 633

# --- smoke test ghi đè ---
if SMOKE_TEST:
    N_QUESTIONS = 300 if BACKEND == "hf" else 100
    if BACKEND == "hf":
        MODEL_NAME, QUANT = "Qwen/Qwen2.5-0.5B", "none"

# %% [code]
if N_SHOTS is None:
    N_SHOTS = 15 if BACKEND == "hf" else 0
TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

def resolve_dtype(choice):
    """Đổi lựa chọn DTYPE thành dtype thật sự dùng được trên máy này (in cảnh báo nếu phải đổi)."""
    cuda = torch.cuda.is_available()
    if choice == "auto":
        choice = "fp16" if cuda else "fp32"
    if choice in ("fp16", "bf16") and not cuda:
        print(f"!! DTYPE={choice} cần GPU, máy này không có GPU -> dùng fp32 (CPU rất chậm; bật Accelerator GPU trong Settings)")
        choice = "fp32"
    if choice == "bf16" and torch.cuda.get_device_capability(GPU_INDEX)[0] < 8:
        print("!! GPU này không có bf16 gốc (cần compute capability >= 8.0, T4/P100 không đạt) -> dùng fp16")
        choice = "fp16"
    assert choice in TORCH_DTYPES, f"DTYPE không hợp lệ: {choice!r}"
    return choice

if BACKEND == "hf":
    # nếu MODEL_NAME là đường dẫn thư mục (vd. /kaggle/input/...), chỉ lấy tên thư mục cuối
    _name = Path(MODEL_NAME).name if os.path.isdir(MODEL_NAME) else MODEL_NAME
    DTYPE_USED = resolve_dtype(DTYPE)
    # dtype nằm trong tên kết quả để các lần chạy fp32/fp16/... không bị trộn khi resume
    _name += "__" + (DTYPE_USED if QUANT == "none" else f"{QUANT}-{DTYPE_USED}")
else:
    DTYPE_USED = None
    _name = f"api-{API_PROVIDER}-{API_MODEL}"
MODEL_ALIAS = _name.replace("/", "_")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"backend={BACKEND} model={MODEL_ALIAS} shots={N_SHOTS} N={('subset ' + str(SUBSET_IDS)) if SUBSET_IDS else (N_QUESTIONS or 'all')}")
if BACKEND == "hf":
    print(f"dtype yêu cầu={DTYPE} -> dùng={DTYPE_USED} | quant={QUANT}")

# %% [markdown]
# ## 2. Dữ liệu PopQA gốc
# Lấy trực tiếp từ repo tác giả. `s_pop` = lượt xem Wikipedia của chủ thể (độ phổ biến). Lấy mẫu **lồng nhau** (xáo trộn cố định theo `SEED` rồi lấy N đầu) nên tăng `N_QUESTIONS` sau này sẽ tái sử dụng được kết quả đã chạy. Muốn chạy đúng một danh sách câu hỏi cố định (vd. 300 câu dùng chung với code gốc, tạo bằng `scripts/make_subset.py`), đặt `SUBSET_IDS` ở ô cấu hình.

# %% [code]
import urllib.request
POPQA_URL = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
popqa_path = WORK / "data" / "popQA.tsv"
if not popqa_path.exists():
    urllib.request.urlretrieve(POPQA_URL, popqa_path)
df_all = pd.read_csv(popqa_path, sep="\t")
df_all["id"] = df_all["id"].astype(int)
PROP_IDS = sorted(df_all["prop_id"].unique().tolist())
POOL = {pid: g.to_dict("records") for pid, g in df_all.groupby("prop_id")}   # nguồn lấy few-shot
perm = df_all.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
if SUBSET_IDS:
    # tập câu hỏi cố định (vd. 300 câu dùng chung với code gốc để đối chứng): giữ đúng thứ tự trong file
    if str(SUBSET_IDS).startswith(("http://", "https://")):
        with urllib.request.urlopen(SUBSET_IDS) as resp:
            _subset_text = resp.read().decode("utf-8")
    else:
        _subset_text = Path(SUBSET_IDS).read_text(encoding="utf-8")
    _subset = [int(tok) for tok in _subset_text.split()]
    assert len(_subset) == len(set(_subset)), "SUBSET_IDS có id bị trùng"
    _missing = sorted(set(_subset) - set(df_all["id"]))
    assert not _missing, f"SUBSET_IDS có id không có trong PopQA: {_missing[:5]}"
    eval_df = df_all.set_index("id", drop=False).loc[_subset].reset_index(drop=True)
else:
    eval_df = perm.head(N_QUESTIONS if N_QUESTIONS else len(perm)).copy()
print(df_all.shape, "-> đánh giá trên", len(eval_df), "câu", f"(theo SUBSET_IDS={SUBSET_IDS})" if SUBSET_IDS else "")
print(eval_df["prop"].value_counts().to_string())

# %% [markdown]
# ## 3. Dữ liệu retrieval của tác giả (BM25, Contriever)
# Tác giả cung cấp sẵn kết quả retrieval (top đoạn Wikipedia 12/2018 cho từng câu hỏi) trong thư mục Google Drive ở README. Ô dưới **tự dò** file BM25/Contriever (tên file hoặc tên thư mục chứa "bm25"/"contriever") trong `/kaggle/input` (dataset bạn add vào notebook), hoặc trong `RETRIEVAL_DIR` nếu bạn đặt. Chỉ khi không thấy file nào mới thử `gdown` từ Drive (link Drive của tác giả hiện trả 404 nên nên dùng dataset).
#
# - **Kaggle (khuyến nghị):** tải tay thư mục Drive về máy → *Create Dataset* → trong notebook *Add Input* chọn dataset đó → chạy lại ô này.
# - Colab: để file vào `WORK/retrieval/` (Drive đã mount) hoặc đặt `RETRIEVAL_DIR`.
# - Có thể ép đường dẫn thủ công qua `RET_FILES = {"BM25": "...", "contriever": "..."}`.
# - File được kỳ vọng là JSONL, mỗi dòng có `question` và `ctxs[0]` gồm `text`, `id`, `hasanswer` (đúng format mà `run_model.py` của tác giả đọc). Nếu mỗi dòng còn có `id` trùng `id` PopQA (như file BM25 tác giả đăng ở issue #5) thì ghép theo `id`, nếu không thì ghép theo văn bản câu hỏi.
# - File ẩn và thư mục `__MACOSX` (rác khi nén zip trên Mac) bị bỏ qua.

# %% [code]
DRIVE_URL = "https://drive.google.com/drive/folders/1ggeoHbSPobbGOljOlwl_d16yssSygYqy"
RET_FILES = {"BM25": None, "contriever": None}      # ép đường dẫn thủ công nếu cần

def norm_q(q):
    return re.sub(r"\s+", " ", str(q).replace("?", "")).strip().lower()

def retrieval_roots():
    """Nơi tìm file retrieval: RETRIEVAL_DIR nếu đặt, ngược lại /kaggle/input (dataset đã add) rồi WORK/retrieval."""
    if RETRIEVAL_DIR:
        return [Path(RETRIEVAL_DIR)]
    return [r for r in (Path("/kaggle/input"), WORK / "retrieval") if r.exists()]

def list_retrieval_files():
    """[(thư mục gốc, đường dẫn file)] cho các file .json/.jsonl dưới các thư mục gốc.
    Bỏ qua file ẩn và rác của macOS (`__MACOSX/._x.jsonl` sinh ra khi nén zip trên Mac)."""
    return [(root, p) for root in retrieval_roots() for p in root.rglob("*")
            if p.is_file() and p.suffix in (".jsonl", ".json")
            and not p.name.startswith(".") and "__MACOSX" not in p.parts]

if not list_retrieval_files() and any(m in METHODS for m in ("BM25", "contriever")):
    print("Chưa thấy file retrieval trong:", [str(r) for r in retrieval_roots()] or "(không có thư mục nào)")
    if IN_KAGGLE:
        print("   -> Kaggle: Add Input -> dataset bạn đã upload chứa file BM25/contriever (hoặc đặt RETRIEVAL_DIR). Thử tải từ Drive dự phòng...")
    try:
        import gdown
        gdown.download_folder(url=DRIVE_URL, output=str(WORK / "retrieval"), quiet=False, use_cookies=False)
    except Exception as e:
        print("!! Tải tự động từ Drive lỗi:", repr(e))
        print("   -> Tải tay thư mục:", DRIVE_URL, "rồi upload thành dataset (Kaggle) hoặc đặt vào", WORK / "retrieval")

files = list_retrieval_files()
print("Các file retrieval tìm thấy:")
for root, p in files:
    print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")
for method in ("BM25", "contriever"):
    if RET_FILES[method] is None:
        # khớp theo đường dẫn tương đối với thư mục gốc (tên file hoặc tên thư mục con đều được)
        cands = [p for root, p in files if method.lower() in str(p.relative_to(root)).lower()]
        if len(cands) > 1:
            print(f"!! {method}: có {len(cands)} file khớp tên, lấy file đầu tiên: {cands[0].name}. Đặt RET_FILES['{method}'] nếu sai.")
        RET_FILES[method] = str(cands[0]) if cands else None
print(RET_FILES)

# %% [code]
def iter_records(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            for rec in json.load(f):
                yield rec
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

ID2Q = dict(zip(df_all["id"], df_all["question"]))

def load_retrieval(path):
    """Chỉ giữ top-1 passage cho từng câu hỏi (như paper).
    Khoá theo `id` PopQA (int) khi record có `id` và câu hỏi khớp đúng câu của PopQA với id đó: PopQA có ~880 nhóm câu hỏi
    trùng văn bản (khác chủ thể), khoá theo `id` giữ passage/hasanswer riêng cho từng dòng. Đồng thời giữ khoá theo văn bản
    câu hỏi (str) làm phương án dự phòng cho file không có `id` (các câu trùng văn bản sẽ dùng chung một passage)."""
    out = {}
    for rec in iter_records(path):
        ctxs = rec.get("ctxs") or rec.get("passages") or []
        if not ctxs:
            continue
        c = ctxs[0]
        entry = {"text": c.get("text", ""), "id": c.get("id"), "hasanswer": bool(c.get("hasanswer", False))}
        out[norm_q(rec["question"])] = entry
        try:
            rid = int(rec.get("id"))
        except (TypeError, ValueError):
            continue
        if rid in ID2Q and norm_q(ID2Q[rid]) == norm_q(rec["question"]):
            out[rid] = entry
    return out

def get_ctx(method, row):
    """Passage retrieved của một dòng PopQA: ưu tiên khoá `id`, rồi đến văn bản câu hỏi."""
    d = RET[method]
    return d.get(int(row["id"])) or d.get(norm_q(row["question"]))

RET = {}
for method, path in RET_FILES.items():
    if path:
        RET[method] = load_retrieval(path)
        rows_all = df_all.to_dict("records")
        cov = np.mean([get_ctx(method, r) is not None for r in rows_all])
        n_id = sum(isinstance(k, int) for k in RET[method])
        k0 = next(k for k in RET[method] if isinstance(k, str))
        print(f"{method}: {sum(isinstance(k, str) for k in RET[method])} câu hỏi duy nhất, khớp theo id {n_id}/{len(df_all)}, "
              f"phủ {cov:.1%} PopQA. Ví dụ: {k0!r} -> {RET[method][k0]['text'][:120]!r}")
        if cov < 0.95:
            print(f"   !! độ phủ thấp: kiểm tra trường 'question' trong file {path} có khớp văn bản câu hỏi PopQA không")
        elif n_id < 0.95 * len(df_all):
            print("   (file không có `id` khớp PopQA: ghép theo văn bản câu hỏi, các câu trùng văn bản dùng chung passage)")

# %% [markdown]
# ### (Tuỳ chọn) Fallback khi không tải được retrieval gốc: `wiki_intro`
# Lấy đoạn mở đầu trang Wikipedia của chủ thể (`s_wiki_title`) làm "retrieval". **Khác paper:** đây là liên kết thực thể lý tưởng (không bị nhầm người trùng tên) và dùng Wikipedia hiện tại thay vì bản 2018, nên sẽ ít hiện tượng "retrieval gây hại" hơn. Chỉ nên dùng như phương án dự phòng; đặt `USE_WIKI_FALLBACK = True` và thêm `"wiki_intro"` vào `METHODS`.

# %% [code]
USE_WIKI_FALLBACK = False

def build_wiki_intro(df, max_words=100):
    import requests
    cache_path = WORK / "cache" / "wiki_intro.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    titles = [t for t in df["s_wiki_title"].astype(str).unique() if t not in cache]
    S = requests.Session()
    S.headers["User-Agent"] = "adaptive-retrieval-repro/0.1 (research reproduction)"
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        r = S.get("https://en.wikipedia.org/w/api.php", timeout=30, params={
            "action": "query", "format": "json", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "exlimit": 20, "redirects": 1, "titles": "|".join(batch)})
        q = r.json().get("query", {})
        nmap = {x["from"]: x["to"] for x in q.get("normalized", [])}
        rmap = {x["from"]: x["to"] for x in q.get("redirects", [])}
        pages = {p["title"]: p.get("extract", "") for p in q.get("pages", {}).values()}
        for t in batch:
            ft = nmap.get(t, t); ft = rmap.get(ft, ft)
            cache[t] = " ".join(pages.get(ft, "").split()[:max_words])
        if (i // 20) % 25 == 0:
            json.dump(cache, open(cache_path, "w")); print(f"  wiki_intro {i}/{len(titles)}")
        time.sleep(0.1)
    json.dump(cache, open(cache_path, "w"))
    out = {}
    for row in df.to_dict("records"):
        txt = cache.get(str(row["s_wiki_title"]), "")
        answers = json.loads(row["possible_answers"])
        out[int(row["id"])] = out[norm_q(row["question"])] = {
            "text": txt, "id": row["s_wiki_title"], "hasanswer": any(a.lower() in txt.lower() for a in answers)}
    return out

if USE_WIKI_FALLBACK:
    RET["wiki_intro"] = build_wiki_intro(df_all)

METHODS_RUN = [m for m in METHODS if m in ("vanilla", "genread") or m in RET]
skipped = [m for m in METHODS if m not in METHODS_RUN]
print("Sẽ chạy:", METHODS_RUN, "| bỏ qua (thiếu retrieval):", skipped)

# %% [markdown]
# ## 4. Prompt (giữ đúng định dạng của tác giả)
# - Template câu hỏi: `Q: {question} A:`; greedy decoding; lấy dòng đầu của output.
# - Few-shot: 1 ví dụ cho mỗi loại quan hệ **khác** với câu đang hỏi (15 ví dụ), xáo trộn thứ tự. Với BM25/Contriever, mỗi ví dụ cũng kèm đoạn retrieved của nó. Few-shot được chọn **cố định theo `id` câu hỏi** nên chạy lại/resume cho prompt giống hệt, và các method dùng cùng bộ ví dụ (so sánh cặp công bằng).
# - Contriever: cắt bỏ câu cuối của đoạn (đoạn bị cắt dở) như code gốc. BM25: giữ nguyên.
# - GenRead: LM tự sinh đoạn nền (≤150 token) rồi dùng làm ngữ cảnh; few-shot của GenRead là tập cố định, 1 ví dụ/loại quan hệ, sinh một lần và cache lại.
# - Chấm điểm: đúng nếu một đáp án chấp nhận được xuất hiện trong output (chuỗi con; thử nguyên bản/chữ thường/viết hoa chữ đầu).

# %% [code]
Q = "Q: {} A:"
GENREAD_TEMPLATE = "Generate a background document from Wikipedia to answer the given question. {}"

def clip_paragraph(text, method):
    if method != "contriever":
        return text
    parts = text.split(". ")
    return text if len(parts) == 1 else ". ".join(parts[:-1]) + "."

def first_line(text):
    if text.startswith("\n\n"):
        text = text[2:]
    return text.split("\n")[0]

def is_correct(pred, answers):
    return any((pa in pred) or (pa.lower() in pred) or (pa.capitalize() in pred) for pa in answers)

def shot_text(row, method):
    base = Q.format(row["question"]) + " " + str(row["obj"])
    if method in RET:
        ctx = get_ctx(method, row)
        if ctx and ctx["text"]:
            return clip_paragraph(ctx["text"], method) + "\n\n" + base
    return base

def select_shots(row, n_shots, method, genread_pool=None):
    if n_shots <= 0:
        return []
    rng = random.Random(f"{SEED}-{row['id']}")
    others = [p for p in PROP_IDS if p != row["prop_id"]]
    k = max(n_shots // len(others), 1)
    rels = others if n_shots >= len(others) else rng.sample(others, n_shots)
    shots = []
    for pid in rels:
        if method == "genread":
            shots += genread_pool[pid][:k]
        else:
            for r in rng.sample(POOL[pid], min(k, len(POOL[pid]))):
                shots.append(shot_text(r, method))
    rng.shuffle(shots)
    return shots

def build_prompt(row, method, n_shots, passage=None, genread_pool=None):
    shots = select_shots(row, n_shots, method, genread_pool)
    head = ("\n\n".join(shots) + "\n\n") if shots else ""
    q = Q.format(row["question"])
    if method == "vanilla":
        body = q
    elif method == "genread":
        body = (passage or "") + "\n\n" + q
    else:
        ctx = get_ctx(method, row)
        txt = clip_paragraph(ctx["text"], method) if ctx else ""
        body = (txt + "\n\n" if txt else "") + q
    return head + body

# %% [markdown]
# ## 5. Backend: model local (HF) hoặc API
# - **HF:** sinh theo batch (sắp xếp theo độ dài prompt để giảm padding), tự giảm batch khi OOM, nạp thẳng theo `DTYPE` (mặc định GPU: fp16, CPU: fp32; không nạp fp32 lên RAM rồi mới chuyển như code gốc).
# - **API:** gọi song song, có retry và **cache trên đĩa** (chạy lại không tốn thêm tiền).

# %% [code]
class HFBackend:
    def __init__(self, name, quant="none", max_ctx=2048, batch_size=8, gpu_index=0, dtype="fp32"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.max_ctx, self.batch_size = max_ctx, batch_size
        dtype = TORCH_DTYPES[dtype]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"; self.tok.truncation_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cuda = torch.cuda.is_available()
        kw = dict(low_cpu_mem_usage=True)
        if quant in ("4bit", "8bit"):
            assert cuda, "Lượng tử hoá 4/8-bit cần GPU"
            kw["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
                if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        if cuda:
            kw["device_map"] = {"": gpu_index}
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        pdt = next(self.model.parameters()).dtype   # dtype thật của trọng số sau khi nạp (4bit: các lớp không lượng tử hoá)
        print(f"Đã nạp {name} | thiết bị: {self.device} | dtype yêu cầu: {dtype} | dtype tham số thực: {pdt} | quant: {quant}")
        if self.device.type == "cuda":
            print(f"  VRAM đang dùng: {torch.cuda.memory_allocated(self.device)/2**30:.2f} GiB")

    def _gen_batch(self, batch, max_new_tokens):
        enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_ctx - max_new_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      num_beams=1, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def _gen_safe(self, batch, max_new_tokens):
        try:
            return self._gen_batch(batch, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            self.batch_size = max(1, mid)
            print(f"  OOM -> giảm batch xuống {self.batch_size}")
            return self._gen_safe(batch[:mid], max_new_tokens) + self._gen_safe(batch[mid:], max_new_tokens)

    def generate(self, prompts, max_new_tokens, kind="answer"):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [None] * len(prompts)
        i = 0
        while i < len(order):
            idx = order[i:i + self.batch_size]
            res = self._gen_safe([prompts[j] for j in idx], max_new_tokens)
            for j, r in zip(idx, res):
                outs[j] = r
            i += len(idx)
        return outs


def get_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    return None


class APIBackend:
    SYSTEM_ANSWER = "Continue the pattern. Reply with only the answer to the final question (a few words), no explanation."

    def __init__(self, provider, model, base_url=None, key_name="OPENAI_API_KEY", workers=4):
        assert model, "Hãy điền API_MODEL"
        key = get_secret(key_name)
        assert key, f"Không tìm thấy khoá API '{key_name}' (Secrets hoặc biến môi trường)"
        self.provider, self.model, self.workers = provider, model, workers
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url=base_url)
        else:
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
        self._tok_param, self._use_temp = "max_tokens", True
        self.lock = threading.Lock()
        self.cache_path = WORK / "cache" / f"api_cache__{provider}__{model.replace('/', '_')}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line); self.cache[d["k"]] = d["v"]
        print(f"API {provider}/{model} | cache: {len(self.cache)} mục")

    def _call(self, prompt, max_tokens, system):
        try:
            if self.provider == "openai":
                kw = {self._tok_param: max_tokens}
                if self._use_temp:
                    kw["temperature"] = 0
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
                r = self.client.chat.completions.create(model=self.model, messages=msgs, **kw)
                return r.choices[0].message.content or ""
            kw = dict(model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            if self._use_temp:
                kw["temperature"] = 0
            if system:
                kw["system"] = system
            r = self.client.messages.create(**kw)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        except Exception as e:
            msg = str(e)
            if self.provider == "openai" and "max_completion_tokens" in msg and self._tok_param == "max_tokens":
                self._tok_param = "max_completion_tokens"; return self._call(prompt, max_tokens, system)
            if "temperature" in msg and self._use_temp:
                self._use_temp = False; return self._call(prompt, max_tokens, system)
            raise

    def generate(self, prompts, max_new_tokens, kind="answer"):
        from concurrent.futures import ThreadPoolExecutor
        system = self.SYSTEM_ANSWER if kind == "answer" else ""
        def one(p):
            k = hashlib.sha256(json.dumps([self.provider, self.model, max_new_tokens, system, p]).encode()).hexdigest()
            with self.lock:
                if k in self.cache:
                    return self.cache[k]
            last = None
            for attempt in range(6):
                try:
                    txt = self._call(p, max_new_tokens, system); break
                except Exception as e:
                    last = e; time.sleep(min(2 ** attempt, 30))
            else:
                raise RuntimeError(f"API lỗi sau 6 lần thử: {last!r}")
            with self.lock:
                self.cache[k] = txt
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": txt}) + "\n")
            return txt
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, prompts))

# %% [code]
if BACKEND == "hf":
    backend = HFBackend(MODEL_NAME, QUANT, MAX_CTX, BATCH_SIZE, GPU_INDEX, DTYPE_USED)
else:
    backend = APIBackend(API_PROVIDER, API_MODEL, API_BASE_URL, API_KEY_NAME, API_WORKERS)

def llm(prompts, max_new_tokens, kind="answer"):
    return [first_line(t) for t in backend.generate(prompts, max_new_tokens, kind)]

# %% [markdown]
# ### (Chỉ API) Ước lượng số token đầu vào trước khi chạy
# Nhân con số này với đơn giá của nhà cung cấp để biết chi phí. GenRead tốn gấp đôi số lần gọi và thêm ~150 token sinh ra mỗi câu.

# %% [code]
if BACKEND == "api":
    sample_rows = eval_df.head(50).to_dict("records")
    for m in [m for m in METHODS_RUN if m != "genread"]:
        lens = [len(build_prompt(r, m, N_SHOTS)) for r in sample_rows]
        est = np.mean(lens) / 4 * len(eval_df)
        print(f"{m:12s} ~{np.mean(lens)/4:7.0f} token/prompt -> ~{est/1e6:.2f}M token đầu vào cho {len(eval_df)} câu")

# %% [markdown]
# ## 6. Chạy thực nghiệm
# Mỗi method ghi ra `results/{model}__{method}__shots{S}.csv`, lưu sau mỗi chunk và **bỏ qua các `id` đã chạy** — nếu session bị ngắt, chỉ cần chạy lại ô này.
# Ô sẽ in tốc độ (câu/giây) và ETA sau mỗi chunk: dùng smoke test để đo tốc độ thật trên GPU của bạn trước khi chạy full.
# Thứ tự khuyến nghị: `vanilla` → `BM25` → `contriever` → `genread` (GenRead chậm nhất vì gọi model 2 lần/câu).

# %% [code]
RESULT_COLS = ["id", "prop", "prop_id", "question", "s_pop", "pred", "is_correct", "has_answer",
               "retrieval_id", "passage", "model", "method", "n_shots"]

def build_genread_pool(n_shots):
    k = max(n_shots // (len(PROP_IDS) - 1), 1)
    path = WORK / "cache" / f"genread_pool__{MODEL_ALIAS}__k{k}.json"
    if path.exists():
        return {int(a): b for a, b in json.load(open(path)).items()}
    rng = random.Random(f"{SEED}-genread-pool")
    pool = {}
    for pid in PROP_IDS:
        rows = rng.sample(POOL[pid], min(k, len(POOL[pid])))
        passages = llm([GENREAD_TEMPLATE.format(r["question"]) for r in rows], GENREAD_TOKENS, "passage")
        pool[pid] = [f"{ps}\n\n{Q.format(r['question'])} {r['obj']}" for r, ps in zip(rows, passages)]
    json.dump(pool, open(path, "w"))
    return pool

def run_method(method, df, n_shots):
    out_path = WORK / "results" / f"{MODEL_ALIAS}__{method}__shots{n_shots}.csv"
    done = set(pd.read_csv(out_path, usecols=["id"])["id"]) if out_path.exists() else set()
    todo = df[~df["id"].isin(done)]
    print(f"\n=== {method}: đã có {len(done)}, còn {len(todo)} câu ===")
    if len(todo) == 0:
        return out_path
    gpool = build_genread_pool(n_shots) if (method == "genread" and n_shots > 0) else None
    t0, shown = time.time(), False
    for start in range(0, len(todo), CHUNK):
        rows = todo.iloc[start:start + CHUNK].to_dict("records")
        passages = [None] * len(rows)
        if method == "genread":
            passages = llm([GENREAD_TEMPLATE.format(r["question"]) for r in rows], GENREAD_TOKENS, "passage")
        prompts = [build_prompt(r, method, n_shots, ps, gpool) for r, ps in zip(rows, passages)]
        if not shown:
            print("--- ví dụ prompt (cuối) ---\n" + prompts[0][-700:] + "\n---------------------------")
            shown = True
        preds = llm(prompts, MAX_NEW_TOKENS, "answer")
        recs = []
        for r, ps, pred in zip(rows, passages, preds):
            answers = json.loads(r["possible_answers"])
            rec = dict(id=r["id"], prop=r["prop"], prop_id=r["prop_id"], question=r["question"], s_pop=r["s_pop"],
                       pred=pred, is_correct=is_correct(pred, answers), has_answer=np.nan, retrieval_id=None,
                       passage=None, model=MODEL_ALIAS, method=method, n_shots=n_shots)
            if method in RET:
                ctx = get_ctx(method, r)
                if ctx:
                    rec.update(has_answer=ctx["hasanswer"], retrieval_id=ctx["id"], passage=ctx["text"][:300])
            elif method == "genread":
                rec.update(has_answer=any(a.lower() in ps.lower() for a in answers), passage=ps[:300])
            recs.append(rec)
        pd.DataFrame(recs)[RESULT_COLS].to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        n_done = start + len(rows)
        qps = n_done / (time.time() - t0)
        print(f"  {n_done}/{len(todo)} | {qps:.2f} câu/s | ETA {(len(todo) - n_done) / qps / 60:.1f} phút")
    return out_path

for m in METHODS_RUN:
    run_method(m, eval_df, N_SHOTS)

# %% [markdown]
# ## 7. Adaptive Retrieval (port đúng từ notebook của tác giả)
# Với mỗi loại quan hệ: chia các câu hỏi thành 10 bin theo `log(s_pop)`; chọn ngưỡng (theo bin) tối đa hoá độ chính xác của hệ lai
# *(dưới ngưỡng dùng kết quả có retrieval, từ ngưỡng trở lên dùng kết quả không retrieval)* **trên tập train (75%)**, rồi đo trên tập test (25%).
# Lặp 100 lần với các lần chia ngẫu nhiên khác nhau và lấy trung bình. Ghép hai bảng kết quả theo `id`.

# %% [code]
def adaptive_eval(df_p, df_r, n_boot=100, split_proportion=0.75, seed=633, min_per_prop=8, sort_key="id"):
    m = df_p[["id", "question", "prop", "s_pop", "is_correct"]].merge(
        df_r[["id", "is_correct"]], on="id", suffixes=("_p", "_r"))
    m = m.sort_values(sort_key).reset_index(drop=True)
    n = len(m)
    props = [p for p in m["prop"].unique() if (m["prop"] == p).sum() >= min_per_prop]
    if not props:
        return None          # quá ít mẫu cho mỗi loại quan hệ (tăng N_QUESTIONS)
    ser_all = np.log(m["s_pop"].values.astype(float))
    c_p_all = m["is_correct_p"].values.astype(bool)
    c_r_all = m["is_correct_r"].values.astype(bool)
    prop_idx = {p: np.where(m["prop"].values == p)[0] for p in props}
    rng = np.random.RandomState(seed)
    hyb, par, ret, retn, ntest_tot = [], [], [], [], []
    for _ in range(n_boot):
        mask = ["train"] * round(n * split_proportion) + ["test"] * round(n * (1 - split_proportion))
        assert len(mask) == n
        rng.shuffle(mask)
        mask = np.array(mask)
        tot_h = tot_p = tot_r = tot_ret = tot_n = 0.0
        for p in props:
            ix = prop_idx[p]
            ser = ser_all[ix]; c = c_p_all[ix]; c_r = c_r_all[ix]; msk = mask[ix] == "test"
            _, edges = np.histogram(ser)
            def counts(sel, correct):
                return np.histogram(ser[sel & correct], bins=edges)[0]
            tr, te = ~msk, msk
            tr_c, tr_c_r = counts(tr, c), counts(tr, c_r)
            te_c, te_c_r = counts(te, c), counts(te, c_r)
            n_tr = int(tr.sum()); n_te = int(te.sum())
            if n_tr == 0 or n_te == 0:
                continue
            nb = len(edges) - 1
            scores = [(tr_c_r[:i].sum() + tr_c[i:].sum()) / n_tr for i in range(nb + 1)]
            thr = int(np.argmax(scores))
            tot_h += te_c_r[:thr].sum() + te_c[thr:].sum()
            tot_p += te_c.sum(); tot_r += te_c_r.sum()
            tot_ret += np.histogram(ser[te], bins=edges)[0][:thr].sum()
            tot_n += n_te
        if tot_n == 0:
            continue
        hyb.append(tot_h / tot_n); par.append(tot_p / tot_n); ret.append(tot_r / tot_n)
        retn.append(tot_ret / tot_n); ntest_tot.append(tot_n)
    # ngưỡng học trên toàn bộ dữ liệu (để tham khảo / dùng thực tế)
    rows = []
    for p in props:
        ix = prop_idx[p]; ser = ser_all[ix]; c = c_p_all[ix]; c_r = c_r_all[ix]
        _, edges = np.histogram(ser)
        a = np.histogram(ser[c], bins=edges)[0]; b = np.histogram(ser[c_r], bins=edges)[0]
        scores = [(b[:i].sum() + a[i:].sum()) / len(ser) for i in range(len(edges))]
        thr = int(np.argmax(scores))
        rows.append(dict(prop=p, n=len(ix), param_acc=a.sum() / len(ser), ret_acc=b.sum() / len(ser),
                         hybrid_acc=max(scores), retrieve_if_s_pop_below=float(np.exp(edges[thr])),
                         frac_retrieve=float((ser < edges[thr]).mean())))
    if not hyb:
        return None
    best = max(np.mean(par), np.mean(ret))
    return dict(n=n, n_props=len(props), param_acc=np.mean(par), ret_acc=np.mean(ret),
                hybrid_acc=np.mean(hyb), hybrid_std=np.std(hyb), gain_vs_best=np.mean(hyb) - best,
                frac_retrieval=np.mean(retn), thresholds=pd.DataFrame(rows).sort_values("n", ascending=False))

def load_results():
    fs = sorted((WORK / "results").glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

RES = load_results()
print(RES.groupby(["model", "method", "n_shots"]).agg(n=("id", "count"), acc=("is_correct", "mean")).round(3).to_string())

# %% [code]
def get(model, method, shots):
    d = RES[(RES.model == model) & (RES.method == method) & (RES.n_shots == shots)]
    return d.drop_duplicates("id") if len(d) else None

summary, adaptive_details = [], {}
for pm in ("vanilla", "genread"):
    for rm in ("BM25", "contriever", "wiki_intro"):
        dp, dr = get(MODEL_ALIAS, pm, N_SHOTS), get(MODEL_ALIAS, rm, N_SHOTS)
        if dp is None or dr is None:
            continue
        r = adaptive_eval(dp, dr)
        if r is None:
            print(f"{pm} + {rm}: quá ít mẫu cho mỗi loại quan hệ (cần >= 8 câu/loại). Tăng N_QUESTIONS rồi chạy lại.")
            continue
        adaptive_details[f"{pm}/{rm}"] = r["thresholds"]
        summary.append({"combo": f"{pm} + {rm}", "n": r["n"], "parametric": r["param_acc"], "retrieval": r["ret_acc"],
                        "adaptive": r["hybrid_acc"], "± std splits": r["hybrid_std"],
                        "gain vs best": r["gain_vs_best"], "frac. retrieving": r["frac_retrieval"]})
summary_df = pd.DataFrame(summary)
print(summary_df.round(3).to_string(index=False) if len(summary_df) else "Chưa đủ kết quả để chạy adaptive (cần ít nhất 1 cặp parametric + retrieval).")
if len(summary_df):
    summary_df.to_csv(WORK / "results" / f"adaptive_summary__{MODEL_ALIAS}.csv", index=False)

# %% [code]
# Ngưỡng theo từng loại quan hệ (học trên toàn bộ dữ liệu): "retrieve nếu s_pop < ngưỡng"
for k, t in adaptive_details.items():
    print(f"\n### {k}")
    print(t.round(3).to_string(index=False))

# %% [markdown]
# ## 8. Biểu đồ: độ chính xác theo độ phổ biến (tương tự Hình 8 của paper)
# Màu gắn cố định theo từng method (không đổi khi bật/tắt method). Vạch lỗi = khoảng tin cậy Wilson 95%.

# %% [code]
import matplotlib.pyplot as plt
COLORS = {"vanilla": "#2a78d6", "BM25": "#eb6834", "contriever": "#1baf7a", "genread": "#eda100", "wiki_intro": "#e87ba4"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def plot_acc_vs_pop(methods, shots, nbins=8, path=None):
    frames = {m: get(MODEL_ALIAS, m, shots) for m in methods}
    frames = {m: d for m, d in frames.items() if d is not None}
    if not frames:
        print("Chưa có kết quả để vẽ."); return
    base = next(iter(frames.values()))
    edges = np.unique(np.quantile(np.log10(base["s_pop"]), np.linspace(0, 1, nbins + 1)))
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for m, d in frames.items():
        lp = np.log10(d["s_pop"].values); b = np.clip(np.digitize(lp, edges[1:-1]), 0, len(edges) - 2)
        xs, ys, los, his = [], [], [], []
        for i in range(len(edges) - 1):
            sel = b == i; n = int(sel.sum())
            if n < 5:
                continue
            k = int(d["is_correct"].values[sel].sum()); lo, hi = wilson(k, n)
            xs.append(np.median(lp[sel])); ys.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
        ax.errorbar(xs, ys, yerr=[los, his], color=COLORS.get(m, INK2), lw=2, marker="o", ms=6,
                    elinewidth=1, capsize=0, label=m)
    ax.set_xlabel("log10(độ phổ biến của chủ thể: lượt xem Wikipedia)", color=INK2)
    ax.set_ylabel("độ chính xác", color=INK2)
    ax.set_ylim(0, 1)
    ax.set_title(f"Độ chính xác theo độ phổ biến — {MODEL_ALIAS} ({shots}-shot, n={len(base)})", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    leg = ax.legend(frameon=False, loc="upper left", ncol=len(frames))
    for t in leg.get_texts():
        t.set_color(INK2)
    if len(frames) <= 4:   # nhãn trực tiếp ở cuối đường (chống chồng chữ), bổ sung cho legend
        ends = sorted(((ln.get_ydata()[-1], ln.get_xdata()[-1], m) for m, ln in
                       zip(frames, [c.lines[0] for c in ax.containers])))
        last_y = -1.0
        xr = ax.get_xlim()[1] - ax.get_xlim()[0]
        for y, x, m in ends:
            ty = max(y, last_y + 0.05); last_y = ty
            ax.text(x + 0.025 * xr, ty, m, color=INK2, fontsize=9, va="center")
        ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * xr)
    fig.tight_layout()
    path = path or (WORK / "figs" / f"acc_vs_pop__{MODEL_ALIAS}.png")
    fig.savefig(path, facecolor=SURFACE); plt.show(); print("Đã lưu", path)

plot_acc_vs_pop(["vanilla", "genread", "BM25", "contriever", "wiki_intro"], N_SHOTS)

# %% [markdown]
# ## 9. Retrieval giúp hay gây hại? (tương tự Bảng 1 của paper)
# Chia câu hỏi theo (LM không retrieval đúng/sai) × (có retrieval đúng/sai), kèm recall@1 của passage. Paper (GPT-3 davinci-003 + Contriever) thấy ~10% câu bị retrieval làm hỏng, với recall@1 rất thấp ở nhóm đó.

# %% [code]
def harm_help_table(df_p, df_r):
    m = df_p[["id", "is_correct"]].merge(df_r[["id", "is_correct", "has_answer"]], on="id", suffixes=("_p", "_r"))
    rows = []
    for p in (True, False):
        for r in (True, False):
            g = m[(m.is_correct_p == p) & (m.is_correct_r == r)]
            rows.append({"không retrieval": "đúng" if p else "sai", "có retrieval": "đúng" if r else "sai",
                         "tỉ lệ câu": len(g) / len(m), "recall@1 của passage": g["has_answer"].astype(float).mean()})
    return pd.DataFrame(rows)

for rm in ("BM25", "contriever", "wiki_intro"):
    dp, dr = get(MODEL_ALIAS, "vanilla", N_SHOTS), get(MODEL_ALIAS, rm, N_SHOTS)
    if dp is not None and dr is not None:
        print(f"\nvanilla vs {rm}"); print(harm_help_table(dp, dr).round(3).to_string(index=False))

# %% [markdown]
# ## 10. Gợi ý mở rộng (để giống paper hơn)
# - Chạy cùng quy trình với nhiều cỡ model (ví dụ 0.5B → 1.5B → 7B 4-bit) để dựng lại biểu đồ scaling (Hình 5) và xem ngưỡng retrieval dịch chuyển theo kích thước model (Hình 10).
# - Thay `MODEL_NAME` rồi chạy lại các ô 1 → 9: kết quả mỗi model nằm trong file riêng, bảng tổng hợp đọc tất cả file trong `results/`.
# - So sánh nhiều model: `RES.groupby(["model","method"]).is_correct.mean().unstack()`.
