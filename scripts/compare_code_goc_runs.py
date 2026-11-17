"""So lần chạy code gốc nguyên bản của Thịnh với kết quả của nhóm, đủ 14.267 câu (PR #68, `docs/KHAC_BIET.md` dòng 7 và 8).

`results/code_goc/` chứa kết quả GPT-Neo 1.3B và 2.7B chạy bằng `run_model.py` của tác giả KHÔNG sửa (nội dung giống hệt
`upstream/run_model.py`, chỉ khác kiểu xuống dòng): vanilla, và BM25 với hành vi gốc là ví dụ mẫu KHÔNG kèm đoạn văn (khóa tra
không khớp file BM25, xem `upstream/KNOWN_ISSUES.md` mục 1). `results/slim/` là kết quả của nhóm: vanilla (đã đặt lại seed
như dòng 9 của `KHAC_BIET.md`) và BM25 có đoạn văn cho ví dụ mẫu (quyết định #12).

Ghi `results/tables/code_goc_vs_nhom.csv`, mỗi mô hình và cách một dòng: độ chính xác của hai bên, chênh lệch (điểm %),
số câu hai bên chấm giống nhau, số câu chỉ đúng ở một bên, và p-value của kiểm định McNemar chính xác (cặp, hai phía):
có phải chênh lệch lớn hơn mức dao động ngẫu nhiên của các câu hai bên chấm khác nhau không.

    python scripts/compare_code_goc_runs.py
"""
import argparse
from math import comb
from pathlib import Path

import pandas as pd

import aggregate

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267


def mcnemar_exact(only_a, only_b):
    """p-value hai phía của kiểm định McNemar chính xác: only_a, only_b là số câu chỉ đúng ở bên A, chỉ đúng ở bên B."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(folder):
    """{(model, dtype, method): bảng kết quả} cho các lượt đủ câu trong một thư mục rút gọn."""
    runs = {}
    for (model, dtype, shots, n), by_method in aggregate.load_groups(folder).items():
        if n == FULL_N:
            for method, df in by_method.items():
                runs[(model, dtype, method)] = df
    return runs


def compare(code_goc, nhom):
    rows = []
    for key, a in sorted(code_goc.items()):
        if key not in nhom or key[2] not in ("vanilla", "BM25"):
            continue
        b = nhom[key]
        if set(a["id"]) != set(b["id"]):
            raise ValueError(f"Hai bên không cùng tập `id`: {key}")
        m = a[["id", "is_correct"]].merge(b[["id", "is_correct"]], on="id", suffixes=("_goc", "_nhom"))
        only_goc = int((m["is_correct_goc"] & ~m["is_correct_nhom"]).sum())
        only_nhom = int((~m["is_correct_goc"] & m["is_correct_nhom"]).sum())
        rows.append(dict(model=key[0], dtype=key[1], method=key[2], n=len(m),
                         acc_code_goc=m["is_correct_goc"].mean(), acc_nhom=m["is_correct_nhom"].mean(),
                         diff_points=100 * (m["is_correct_nhom"].mean() - m["is_correct_goc"].mean()),
                         same_correct_share=float((m["is_correct_goc"] == m["is_correct_nhom"]).mean()),
                         only_code_goc=only_goc, only_nhom=only_nhom, mcnemar_p=mcnemar_exact(only_goc, only_nhom)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="So code gốc nguyên bản của Thịnh với kết quả của nhóm.")
    ap.add_argument("--code-goc-dir", default=str(ROOT / "results" / "code_goc"))
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "code_goc_vs_nhom.csv"))
    args = ap.parse_args()
    table = compare(load(args.code_goc_dir), load(args.slim_dir))
    if table.empty:
        raise SystemExit("Không có cặp lượt chạy nào để so trong " + args.code_goc_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
