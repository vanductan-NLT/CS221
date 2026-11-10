"""Bảng 2×2 "retrieval giúp hay hại", làm lại Bảng 1 của paper bằng BM25 (issue #29).

Với mỗi mô hình có đủ vanilla và BM25 (14.267 câu), chia câu hỏi thành 4 ô theo hai điều: vanilla đúng hay
sai, BM25 đúng hay sai (cột `is_correct`, ghép theo `id`). Mỗi ô có:
- `pct`: tỉ lệ câu rơi vào ô (4 ô cộng lại 100%).
- `recall_at_1`: trong các câu của ô, bao nhiêu phần có đáp án nằm trong đoạn BM25 top-1 mà mô hình thấy.
  Dùng cột `answer_in_passage` của `results/tables/bm25_top1_has_answer.csv` (tính bằng đúng quy tắc chấm
  của đồ án, xem `scripts/make_bm25_top1_table.py`), không dùng cờ `has_answer` có sẵn vì cờ đó đánh giá thiếu.
- `recall_flag`: cùng phép tính nhưng theo cờ `has_answer` (`hasanswer_flag`), để đối chiếu.

Hai ô đáng chú ý: vanilla đúng mà BM25 sai (tra cứu **hại**) và vanilla sai mà BM25 đúng (tra cứu **giúp**).
Đoạn văn là đoạn mô hình thực sự thấy: script kiểm `retrieval_id` của từng lượt BM25 trùng với bảng top-1.

Ghi `results/tables/retrieval_help_hurt.csv`. Chạy trên CPU, vài giây:
    python scripts/retrieval_help_hurt.py
"""
import argparse
from pathlib import Path

import pandas as pd

import aggregate

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267


def passage_id(series):
    """id đoạn văn dạng chuỗi số nguyên, để so giữa kết quả mô hình (có thể là số) và bảng top-1."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def two_by_two(vanilla, bm25, top1):
    """Bốn ô (vanilla đúng/sai × BM25 đúng/sai): số câu, tỉ lệ câu, recall@1 theo quy tắc chấm và theo cờ."""
    m = vanilla[["id", "is_correct"]].merge(bm25[["id", "is_correct", "retrieval_id"]], on="id", suffixes=("_v", "_b"))
    m = m.merge(top1, on="id", suffixes=("", "_top1"))
    if len(m) != len(vanilla) or len(m) != len(bm25) or len(m) != len(top1):
        raise ValueError("Ba bảng không cùng tập `id`.")
    if (passage_id(m["retrieval_id"]) != passage_id(m["retrieval_id_top1"])).any():
        raise ValueError("`retrieval_id` của lượt BM25 khác bảng top-1: đoạn văn mô hình thấy không phải đoạn đã kiểm đáp án.")
    rows = []
    for vanilla_ok in (True, False):
        for bm25_ok in (True, False):
            g = m[(m["is_correct_v"] == vanilla_ok) & (m["is_correct_b"] == bm25_ok)]
            rows.append(dict(vanilla_correct=vanilla_ok, bm25_correct=bm25_ok, count=len(g), pct=len(g) / len(m),
                             recall_at_1=g["answer_in_passage"].mean() if len(g) else float("nan"),
                             recall_flag=g["hasanswer_flag"].mean() if len(g) else float("nan")))
    return rows


def build_table(slim_dir, top1_path):
    top1 = pd.read_csv(top1_path)
    for col in ("hasanswer_flag", "answer_in_passage"):
        top1[col] = top1[col].astype(str).str.strip().str.lower().eq("true")
    rows = []
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n != FULL_N or not {"vanilla", "BM25"} <= set(runs):
            continue
        for cell in two_by_two(runs["vanilla"], runs["BM25"], top1):
            rows.append(dict(model=model, dtype=dtype, shots=shots, n=n, **cell))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Bảng 2×2 retrieval giúp hay hại (BM25).")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--top1", default=str(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "retrieval_help_hurt.csv"))
    args = ap.parse_args()
    table = build_table(args.slim_dir, args.top1)
    if table.empty:
        raise SystemExit("Không có mô hình nào đủ vanilla và BM25 (14.267 câu) trong " + args.slim_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
