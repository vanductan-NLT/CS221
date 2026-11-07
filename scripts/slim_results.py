"""Rút gọn file kết quả để đưa lên git (issue #23).

Đọc file CSV đầy đủ (mặc định mọi file trong `results/raw/`), chỉ giữ các cột cần cho phân tích,
rồi ghi ra `results/slim/` với đúng tên file cũ. Các cột còn lại (văn bản câu hỏi, đoạn văn, prompt,
văn bản mô hình sinh ra, alias, link Wikidata...) bị bỏ; cần thì ghép lại với `data/popQA.tsv` theo `id`.

Đọc được cả hai kiểu file:
- notebook nhóm: id, prop, prop_id, question, s_pop, pred, is_correct, has_answer, retrieval_id, passage,
  model, method, n_shots
- code gốc `run_model.py`: có thêm subj, obj, alias, uri, prompt, generation...; không có cột
  model/method/n_shots vì các thông tin này nằm trong tên file

Cột giữ lại, theo thứ tự (cột nào file có): id, prop, prop_id, s_pop, pred, is_correct, has_answer,
retrieval_id, conf, model, method, n_shots. Thiếu một trong sáu cột đầu thì dừng và báo tên cột thiếu.
`conf` (độ tự tin của câu trả lời, issue #14) chỉ có ở các lượt chạy từ 07/11/2026.
Bản rút gọn quá 10 MB thì không ghi (luật repo, xem CLAUDE.md).

Chạy trên CPU, vài giây mỗi file:
    python scripts/slim_results.py                         # mọi file trong results/raw/
    python scripts/slim_results.py a.csv b.csv --out-dir results/slim
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["id", "prop", "prop_id", "s_pop", "pred", "is_correct"]
OPTIONAL = ["has_answer", "retrieval_id", "conf", "model", "method", "n_shots"]
MAX_BYTES = 10 * 1024 * 1024


def slim_file(path, out_dir):
    """Ghi bản rút gọn của `path` vào `out_dir`, cùng tên file. Trả về đường dẫn file mới."""
    path = Path(path)
    # keep_default_na=False: dự đoán rỗng giữ là chuỗi rỗng, has_answer trống (vanilla) giữ là trống.
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: thiếu cột {missing}; xem quy ước cột ở results/README.md")
    if df["id"].duplicated().any():
        raise SystemExit(f"{path}: có `id` bị trùng")

    keep = REQUIRED + [c for c in OPTIONAL if c in df.columns]
    out = Path(out_dir) / path.name
    df[keep].to_csv(out, index=False)

    size = out.stat().st_size
    if size > MAX_BYTES:
        out.unlink()
        raise SystemExit(f"{out}: {size / 1e6:.1f} MB, quá 10 MB nên không ghi")
    print(f"{out}: {len(df)} dòng, {len(keep)} cột, {size / 1e6:.2f} MB "
          f"(bản đầy đủ {path.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Rút gọn file kết quả để đưa lên git.")
    ap.add_argument("files", nargs="*", help="File CSV đầy đủ; bỏ trống thì lấy mọi file trong results/raw/")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "slim"))
    args = ap.parse_args()

    files = args.files or sorted((ROOT / "results" / "raw").glob("*.csv"))
    if not files:
        raise SystemExit("Không có file CSV nào để rút gọn.")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for f in files:
        slim_file(f, args.out_dir)


if __name__ == "__main__":
    main()
