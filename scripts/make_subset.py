"""Chọn N câu hỏi cố định từ PopQA để mọi lần đối chứng chạy trên đúng cùng một tập.

Cách chọn giống hệt notebook chính khi `N_QUESTIONS = N` và `SEED = 633`: xáo trộn toàn bộ
PopQA bằng `sample(frac=1.0, random_state=seed)` rồi lấy N dòng đầu. Vì vậy tập 300 câu nằm
gọn trong tập 3.000 câu (lấy mẫu lồng nhau), nên kết quả đã chạy trên 300 câu vẫn dùng lại được.

File ra: mỗi dòng một `id` PopQA, giữ nguyên thứ tự sau khi xáo trộn.

Chạy trên CPU, vài giây:
    python scripts/make_subset.py                                  # data/subset_300_ids.txt
    python scripts/make_subset.py --n 3000 --out data/subset_3000_ids.txt
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "upstream" / "data" / "popQA.tsv"), help="file PopQA (.tsv)")
    ap.add_argument("--n", type=int, default=300, help="số câu hỏi cần chọn")
    ap.add_argument("--seed", type=int, default=633, help="seed xáo trộn (giữ 633 như code gốc)")
    ap.add_argument("--out", default=None, help="file ra; mặc định data/subset_<n>_ids.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    df["id"] = df["id"].astype(int)
    if df["id"].duplicated().any():
        raise SystemExit("`id` trong PopQA bị trùng, không thể chọn theo id.")
    if not 0 < args.n <= len(df):
        raise SystemExit(f"--n phải trong khoảng 1..{len(df)}, nhận được {args.n}.")

    perm = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    subset = perm.head(args.n)

    out = Path(args.out) if args.out else ROOT / "data" / f"subset_{args.n}_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": trên Windows vẫn ghi LF để file giống nhau ở mọi máy
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{i}\n" for i in subset["id"])

    per_rel = subset.groupby("prop").size().sort_values()
    print(f"Đã ghi {len(subset)} id vào {out}")
    print(f"Số câu theo loại quan hệ (ít nhất -> nhiều nhất):\n{per_rel.to_string()}")


if __name__ == "__main__":
    main()
