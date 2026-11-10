"""Adaptive retrieval theo độ tự tin của mô hình, thay cho độ nổi tiếng (issue #34, hướng C).

Ý tưởng: paper quyết định "có tra cứu không" bằng `s_pop` (lượt xem trang Wikipedia mỗi tháng). Ngoài đời
không ai biết trước `s_pop` của một câu hỏi, nhưng mô hình biết nó chắc đến đâu về câu trả lời của chính
nó. Cột `conf` (issue #14) là trung bình log-xác suất của các token trong câu trả lời vanilla: gần 0 là rất
chắc, càng âm càng không chắc. Chỗ nào mô hình không chắc thì tra cứu, chỗ nào chắc thì tin mô hình.

Quy trình giống hệt adaptive gốc (`aggregate.adaptive_eval`), chỉ đổi "điểm" từ log(s_pop) sang `conf`:
với mỗi loại quan hệ chia 10 bin theo điểm, chọn ngưỡng tốt nhất trên 75% câu (dưới ngưỡng dùng BM25, từ
ngưỡng trở lên dùng vanilla), đo trên 25% còn lại, lặp 100 lần, seed 633. Hai cách điểm dùng đúng cùng
các lần chia, nên so cặp được: `conf_wins` là số lần chia (trên 100) mà cách theo `conf` thắng.

Ghi `results/tables/adaptive_by_confidence.csv` cho mọi mô hình có đủ vanilla (có cột `conf`) và BM25,
và `results/tables/adaptive_by_confidence_bins.csv`: tra cứu giúp hay hại ở từng nhóm điểm (lượt đủ câu).
Hai số giải thích kết quả:
- AUROC: nhặt ngẫu nhiên một câu vanilla đúng và một câu vanilla sai; xác suất điểm của câu đúng cao hơn.
  0,5 là đoán mò, 1 là phân biệt hoàn hảo. So `auroc_conf` với `auroc_pop`.
- `spearman`: conf và log(s_pop) giống nhau đến đâu (0 là không liên quan, 1 là xếp hạng giống hệt).

Chạy trên CPU, khoảng một phút:
    python scripts/adaptive_by_confidence.py
    python scripts/adaptive_by_confidence.py --slim-dir results/slim --out results/tables/adaptive_by_confidence.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate

ROOT = Path(__file__).resolve().parents[1]


def auroc(score, label):
    """Xác suất điểm của một câu đúng cao hơn điểm của một câu sai (dựa trên hạng, điểm bằng nhau tính nửa)."""
    ranks = pd.Series(score).rank().values
    pos = np.asarray(label, dtype=bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def confidence_column(df):
    """Cột `conf` dạng số, hoặc None nếu lượt này không có. Câu trả lời rỗng không có conf: coi là không
    chắc nhất (điểm thấp nhất) để luôn đi tra cứu."""
    if "conf" not in df.columns:
        return None
    conf = pd.to_numeric(df["conf"].replace("", np.nan))
    if conf.isna().all():
        return None
    return conf.fillna(conf.min())


def compare(vanilla, bm25):
    """Adaptive theo độ nổi tiếng và theo độ tự tin trên cùng một cặp lượt vanilla/BM25."""
    vanilla = vanilla.assign(conf=confidence_column(vanilla))
    by_pop = aggregate.adaptive_eval(vanilla, bm25)
    by_conf = aggregate.adaptive_eval(vanilla, bm25, score_col="conf")
    if by_pop is None or by_conf is None:
        return None
    correct = vanilla["is_correct"].values
    log_pop = np.log(vanilla["s_pop"].astype(float).values)
    diff = by_conf["hybrid_runs"] - by_pop["hybrid_runs"]
    return dict(
        adaptive_pop=by_pop["hybrid_acc"], frac_pop=by_pop["frac_retrieval"],
        adaptive_conf=by_conf["hybrid_acc"], adaptive_conf_std=by_conf["hybrid_std"],
        frac_conf=by_conf["frac_retrieval"],
        diff=diff.mean(), conf_wins=int((diff > 0).sum()), n_splits=len(diff),
        auroc_conf=auroc(vanilla["conf"].values, correct), auroc_pop=auroc(log_pop, correct),
        spearman=pd.Series(vanilla["conf"].values).corr(pd.Series(log_pop), method="spearman"))


def gain_by_bins(vanilla, bm25, k=5):
    """Tra cứu giúp hay hại ở từng mức điểm. Với mỗi cách điểm (conf, độ nổi tiếng), xếp mọi câu theo điểm
    tăng dần, chia k nhóm bằng số câu (nhóm 1 = điểm thấp nhất = nên tra cứu nhất). Mỗi nhóm: vanilla đúng
    bao nhiêu, BM25 đúng bao nhiêu, và gain = BM25 - vanilla (dương là tra cứu giúp). Gộp mọi loại quan hệ
    cho dễ đọc; adaptive thật chọn ngưỡng riêng theo từng loại quan hệ."""
    m = vanilla.assign(conf=confidence_column(vanilla), log_pop=np.log(vanilla["s_pop"].astype(float)))
    m = m.merge(bm25[["id", "is_correct"]], on="id", suffixes=("_v", "_b"))
    rows = []
    for name, col in (("conf", "conf"), ("pop", "log_pop")):
        m["bin"] = pd.qcut(m[col].rank(method="first"), k, labels=False) + 1
        for b, g in m.groupby("bin"):
            v, r = g["is_correct_v"].mean(), g["is_correct_b"].mean()
            rows.append(dict(score=name, bin=int(b), n=len(g), vanilla=v, BM25=r, gain=r - v))
    return rows


def build_bins_table(slim_dir, min_n=1000):
    rows = []
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n < min_n or not {"vanilla", "BM25"} <= set(runs) or confidence_column(runs["vanilla"]) is None:
            continue
        rows += [dict(model=model, dtype=dtype, shots=shots, n_total=n, **r)
                 for r in gain_by_bins(runs["vanilla"], runs["BM25"])]
    return pd.DataFrame(rows)


def build_table(slim_dir):
    rows = []
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if not {"vanilla", "BM25"} <= set(runs):
            continue
        if confidence_column(runs["vanilla"]) is None:
            print(f"Bỏ qua {model} n={n}: lượt vanilla chưa có cột conf")
            continue
        result = compare(runs["vanilla"], runs["BM25"])
        if result is None:
            continue
        rows.append(dict(model=model, dtype=dtype, shots=shots, n=n,
                         vanilla=runs["vanilla"]["is_correct"].mean(), BM25=runs["BM25"]["is_correct"].mean(),
                         **result))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Adaptive retrieval theo độ tự tin so với theo độ nổi tiếng.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "adaptive_by_confidence.csv"))
    ap.add_argument("--bins-out", default=str(ROOT / "results" / "tables" / "adaptive_by_confidence_bins.csv"))
    args = ap.parse_args()
    table = build_table(args.slim_dir)
    if table.empty:
        raise SystemExit("Không có mô hình nào đủ vanilla (có conf) và BM25 trong " + args.slim_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nĐã ghi {args.out}")
    bins = build_bins_table(args.slim_dir)
    bins.to_csv(args.bins_out, index=False)
    print(f"Đã ghi {args.bins_out} (tra cứu giúp hay hại theo 5 nhóm điểm, chỉ lượt đủ câu)")


if __name__ == "__main__":
    main()
