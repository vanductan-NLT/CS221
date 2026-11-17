"""Độ chính xác theo 16 loại quan hệ và tương quan với độ nổi tiếng (issue #36, mục 2 của `docs/PHAN_TICH.md`).

Với mỗi mô hình có kết quả đủ 14.267 câu và mỗi cách (vanilla, BM25, GenRead nếu có), mỗi loại quan hệ:
- `accuracy`: độ chính xác theo cột `is_correct` (cách chấm của paper; chấm chặt hơn xem `scripts/strict_grading.py`);
- `pop_corr`: tương quan Pearson giữa log10(`s_pop`) và việc câu trả lời đúng (0 hoặc 1) trong loại quan hệ đó, như
  Hình 4 của paper. Gần 1 nghĩa là câu càng nổi tiếng càng hay đúng; gần 0 là độ nổi tiếng không liên quan. Để trống
  khi mọi câu trong loại quan hệ cùng đúng hoặc cùng sai.

Ghi hai bảng trong `results/tables/` và một hình:
1. `accuracy_by_relation.csv`: mỗi mô hình, cách, loại quan hệ một dòng: số câu, độ chính xác, tương quan.
2. `accuracy_by_relation_summary.csv`: mỗi cách và loại quan hệ một dòng, gộp qua các mô hình có đủ cả vanilla và BM25
   (trung bình, nhỏ nhất, lớn nhất của độ chính xác và tương quan) và mức tăng của BM25 so với vanilla.
3. `results/figs/acc_corr_by_relation.png`: trên: độ chính xác theo loại quan hệ (vanilla và BM25, trung bình các mô hình,
   vạch là khoảng nhỏ nhất đến lớn nhất); dưới: tương quan với độ nổi tiếng.

    python scripts/analyze_by_relation.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267
METHODS = ["vanilla", "BM25", "genread"]


def pop_correlation(s_pop, correct):
    """Tương quan Pearson giữa log10(s_pop) và đúng/sai; NaN nếu một vector không đổi hoặc có dưới 3 câu."""
    x = np.log10(np.asarray(s_pop, dtype=float))
    y = np.asarray(correct, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_long_table(slim_dir):
    rows = []
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n != FULL_N:
            continue
        for method in METHODS:
            if method not in runs:
                continue
            df = runs[method]
            for prop, g in df.groupby("prop"):
                rows.append(dict(model=model, dtype=dtype, method=method, prop=prop, n=len(g),
                                 accuracy=float(g["is_correct"].mean()),
                                 pop_corr=pop_correlation(g["s_pop"].astype(float), g["is_correct"])))
    return pd.DataFrame(rows)


def summarize(long_table):
    """Gộp qua các mô hình có đủ vanilla và BM25 (9 mô hình), mỗi cách và loại quan hệ một dòng; thêm mức tăng của BM25."""
    both = [m for m, g in long_table.groupby("model") if {"vanilla", "BM25"} <= set(g["method"])]
    t = long_table[long_table["model"].isin(both) & long_table["method"].isin(["vanilla", "BM25"])]
    s = t.groupby(["method", "prop"]).agg(n=("n", "first"), n_models=("model", "nunique"),
                                         mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"),
                                         mean_corr=("pop_corr", "mean"), min_corr=("pop_corr", "min"), max_corr=("pop_corr", "max")).reset_index()
    gain = (s[s.method == "BM25"].set_index("prop")["mean_acc"] - s[s.method == "vanilla"].set_index("prop")["mean_acc"])
    s["bm25_gain_pts"] = s["prop"].map(100 * gain).where(s["method"] == "BM25")
    return s


def draw_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = summary[summary.method == "vanilla"].set_index("prop")
    b = summary[summary.method == "BM25"].set_index("prop")
    order = list(v.sort_values("mean_acc", ascending=False).index)
    x = np.arange(len(order))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    for off, tbl, label, color in ((-w / 2, v, "vanilla", "#4c72b0"), (w / 2, b, "BM25", "#dd8452")):
        mean = tbl.loc[order, "mean_acc"].values
        err = [mean - tbl.loc[order, "min_acc"].values, tbl.loc[order, "max_acc"].values - mean]
        ax1.bar(x + off, 100 * mean, w, yerr=100 * np.array(err), capsize=2, label=label, color=color, error_kw=dict(lw=0.8))
    ax1.set_ylabel("Độ chính xác (%)")
    ax1.set_title("Độ chính xác theo loại quan hệ (trung bình 9 mô hình, vạch = nhỏ nhất đến lớn nhất)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for off, tbl, label, color in ((-w / 2, v, "vanilla", "#4c72b0"), (w / 2, b, "BM25", "#dd8452")):
        mean = tbl.loc[order, "mean_corr"].values
        err = [mean - tbl.loc[order, "min_corr"].values, tbl.loc[order, "max_corr"].values - mean]
        ax2.bar(x + off, mean, w, yerr=np.array(err), capsize=2, label=label, color=color, error_kw=dict(lw=0.8))
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Tương quan với log10(độ nổi tiếng)")
    ax2.set_title("Câu càng nổi tiếng càng hay đúng? (như Hình 4 của paper)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=40, ha="right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Độ chính xác theo loại quan hệ và tương quan với độ nổi tiếng.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--fig", default=str(ROOT / "results" / "figs" / "acc_corr_by_relation.png"))
    args = ap.parse_args()
    long_table = build_long_table(args.slim_dir)
    if long_table.empty:
        raise SystemExit("Không có mô hình nào đủ 14.267 câu trong " + args.slim_dir)
    summary = summarize(long_table)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(out / "accuracy_by_relation.csv", index=False)
    summary.to_csv(out / "accuracy_by_relation_summary.csv", index=False)
    draw_figure(summary, args.fig)
    print(summary.round(3).to_string(index=False))
    print(f"\nĐã ghi 2 bảng vào {out} và hình {args.fig}")


if __name__ == "__main__":
    main()
