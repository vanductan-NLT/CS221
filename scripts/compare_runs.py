"""So sánh hai file kết quả (vd. code gốc và notebook nhóm) chạy trên cùng một tập câu hỏi, ghép theo `id`.

In ra: số câu, độ chính xác mỗi bên, bảng đúng/sai theo từng câu, kiểm McNemar chính xác,
tỉ lệ dự đoán giống nhau và danh sách các câu hai bên trả lời khác nhau.

Hai file cần có các cột `id`, `pred`, `is_correct` (file của code gốc và của notebook nhóm đều có).
Ghép theo `id`, không ghép theo văn bản câu hỏi, vì PopQA có 881 câu hỏi trùng văn bản.

Chạy trên CPU, vài giây:
    python scripts/compare_runs.py a.csv b.csv
    python scripts/compare_runs.py a.csv b.csv --label-a "code gốc" --label-b "notebook nhóm" --show 15 --out diff.csv
"""
import argparse
import math
import sys

import pandas as pd


def read_run(path):
    """Đọc file kết quả; giữ nguyên dự đoán rỗng (không biến thành NaN)."""
    df = pd.read_csv(path, keep_default_na=False)
    missing = {"id", "pred", "is_correct"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: thiếu cột {sorted(missing)}")
    if df["id"].duplicated().any():
        raise SystemExit(f"{path}: có `id` bị trùng, không ghép được theo id")
    df["id"] = df["id"].astype(int)
    df["is_correct"] = df["is_correct"].map(lambda v: str(v).strip().lower() == "true")
    df["pred"] = df["pred"].astype(str)
    return df


def mcnemar_exact(only_a, only_b):
    """p hai phía của kiểm McNemar chính xác (nhị thức, p = 0,5) trên các câu hai bên chấm khác nhau."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def norm(text):
    """Chuẩn hóa để so 'dự đoán giống nhau': bỏ khoảng trắng đầu cuối, không phân biệt hoa/thường."""
    return text.strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a", help="file kết quả bên A (vd. code gốc)")
    ap.add_argument("file_b", help="file kết quả bên B (vd. notebook nhóm)")
    ap.add_argument("--label-a", default="A", help="tên hiển thị của bên A")
    ap.add_argument("--label-b", default="B", help="tên hiển thị của bên B")
    ap.add_argument("--show", type=int, default=10, help="in tối đa bao nhiêu câu khác nhau cho mỗi nhóm (mặc định 10)")
    ap.add_argument("--out", default=None, help="ghi CSV đầy đủ các câu hai bên dự đoán khác nhau")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows: tránh lỗi in tiếng Việt
    la, lb = args.label_a, args.label_b

    a, b = read_run(args.file_a), read_run(args.file_b)
    only_in_a, only_in_b = set(a["id"]) - set(b["id"]), set(b["id"]) - set(a["id"])
    if only_in_a or only_in_b:
        print(f"!! Cảnh báo: {len(only_in_a)} id chỉ có ở {la}, {len(only_in_b)} id chỉ có ở {lb}; chỉ so các id chung.")
    keep = [c for c in ("prop", "question") if c in a.columns]
    j = a[["id", "pred", "is_correct"] + keep].merge(b[["id", "pred", "is_correct"]], on="id", suffixes=("_a", "_b"))
    if j.empty:
        raise SystemExit("Hai file không có id chung nào.")
    n = len(j)

    print(f"Số câu so sánh (ghép theo id): {n}")
    print(f"Độ chính xác: {la} = {j.is_correct_a.mean() * 100:.2f}% ({int(j.is_correct_a.sum())}/{n}) | "
          f"{lb} = {j.is_correct_b.mean() * 100:.2f}% ({int(j.is_correct_b.sum())}/{n}) | "
          f"chênh {(j.is_correct_b.mean() - j.is_correct_a.mean()) * 100:+.2f} điểm ({lb} trừ {la})")

    both = int((j.is_correct_a & j.is_correct_b).sum())
    only_a = int((j.is_correct_a & ~j.is_correct_b).sum())
    only_b = int((~j.is_correct_a & j.is_correct_b).sum())
    neither = n - both - only_a - only_b
    print(f"\nSo từng câu:  cả hai đúng = {both} | chỉ {la} đúng = {only_a} | chỉ {lb} đúng = {only_b} | cả hai sai = {neither}")
    print(f"Kiểm McNemar chính xác trên {only_a + only_b} câu chấm khác nhau: p = {mcnemar_exact(only_a, only_b):.4f} (chỉ để tham khảo)")

    same_exact = int((j.pred_a.str.strip() == j.pred_b.str.strip()).sum())
    same_norm = int((j.pred_a.map(norm) == j.pred_b.map(norm)).sum())
    print(f"\nDự đoán giống nhau: {same_exact}/{n} ({same_exact / n * 100:.1f}%) khi so chuỗi nguyên văn (bỏ khoảng trắng đầu cuối); "
          f"{same_norm}/{n} ({same_norm / n * 100:.1f}%) khi không phân biệt hoa/thường")

    diff = j[j.pred_a.map(norm) != j.pred_b.map(norm)].copy()
    diff["ket_qua"] = diff.apply(lambda r: f"chỉ {la} đúng" if r.is_correct_a and not r.is_correct_b
                                 else f"chỉ {lb} đúng" if r.is_correct_b and not r.is_correct_a
                                 else "cùng đúng" if r.is_correct_a else "cùng sai", axis=1)
    print(f"Số câu hai bên dự đoán khác nhau: {len(diff)}")
    print("  " + " | ".join(f"{k} = {v}" for k, v in diff["ket_qua"].value_counts().items()))

    for label in (f"chỉ {lb} đúng", f"chỉ {la} đúng"):
        rows = diff[diff.ket_qua == label]
        print(f"\n--- {label}: {len(rows)} câu (in {min(args.show, len(rows))} câu đầu) ---")
        for r in rows.head(args.show).itertuples():
            q = getattr(r, "question", "")
            p = getattr(r, "prop", "")
            print(f"  id {r.id} [{p}] {q}\n      {la}: {r.pred_a.strip()!r} | {lb}: {r.pred_b.strip()!r}")

    if args.out:
        cols = ["id"] + keep + ["pred_a", "pred_b", "is_correct_a", "is_correct_b", "ket_qua"]
        diff[cols].to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"\nĐã ghi {len(diff)} câu khác nhau vào {args.out}")


if __name__ == "__main__":
    main()
