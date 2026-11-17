"""Kiểm scripts/analyze_by_relation.py (issue #36). Chạy trên CPU: python scripts/test_analyze_by_relation.py

1. Tương quan với độ nổi tiếng: đúng ở trường hợp biết trước; không đổi hoặc quá ít câu thì để trống.
2. Gộp theo loại quan hệ trên dữ liệu giả: trung bình, nhỏ nhất, lớn nhất và mức tăng của BM25.
3. Trên các file đã commit: độ chính xác theo loại quan hệ cộng lại (có trọng số theo số câu) ra đúng độ chính xác chung
   trong `summary.csv` ở mọi mô hình và cách; số câu mỗi loại quan hệ cộng 14.267 và giống nhau giữa các mô hình;
   các số nêu trong `docs/PHAN_TICH.md` mục 2 (tương quan của BM25 thấp hơn vanilla ở 16/16 loại, chỉ `occupation` có mức tăng
   âm, tương quan vanilla dương ở cả 16 loại và từ 0,2 trở lên ở 11 loại); hình tồn tại và đủ lớn.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_by_relation as abr  # noqa: E402

# 1.
r = abr.pop_correlation([10, 100, 1000, 10000], [0, 0, 1, 1])               # log10 = 1,2,3,4 so với 0,0,1,1
assert np.isclose(r, np.corrcoef([1, 2, 3, 4], [0, 0, 1, 1])[0, 1]) and 0.89 < r < 0.90
assert np.isclose(abr.pop_correlation([10, 100, 1000], [1, 0, 0]), -0.866, atol=1e-3)
assert np.isnan(abr.pop_correlation([10, 100, 1000], [1, 1, 1])) and np.isnan(abr.pop_correlation([5, 5, 5], [0, 1, 0]))
assert np.isnan(abr.pop_correlation([10, 100], [0, 1]))
print("1 ok: tương quan đúng ở trường hợp biết trước; không đổi hoặc quá ít câu thì để trống")

# 2.
long_table = pd.DataFrame([
    dict(model="a", dtype="fp32", method="vanilla", prop="x", n=10, accuracy=0.2, pop_corr=0.5),
    dict(model="b", dtype="fp32", method="vanilla", prop="x", n=10, accuracy=0.4, pop_corr=0.1),
    dict(model="a", dtype="fp32", method="BM25", prop="x", n=10, accuracy=0.5, pop_corr=0.0),
    dict(model="b", dtype="fp32", method="BM25", prop="x", n=10, accuracy=0.7, pop_corr=-0.1),
    dict(model="c", dtype="fp32", method="vanilla", prop="x", n=10, accuracy=0.9, pop_corr=0.9),    # thiếu BM25: không được gộp
])
s = abr.summarize(long_table).set_index(["method", "prop"])
assert s.loc[("vanilla", "x"), "n_models"] == 2 and np.isclose(s.loc[("vanilla", "x"), "mean_acc"], 0.3)
assert np.isclose(s.loc[("BM25", "x"), "mean_acc"], 0.6) and np.isclose(s.loc[("BM25", "x"), "min_acc"], 0.5) and np.isclose(s.loc[("BM25", "x"), "max_corr"], 0.0)
assert np.isclose(s.loc[("BM25", "x"), "bm25_gain_pts"], 30.0) and np.isnan(s.loc[("vanilla", "x"), "bm25_gain_pts"])
print("2 ok: gộp theo loại quan hệ đúng trung bình, nhỏ nhất, lớn nhất và mức tăng; mô hình thiếu BM25 không được gộp")

# 3.
tables = ROOT / "results" / "tables"
long_t = pd.read_csv(tables / "accuracy_by_relation.csv")
summary = pd.read_csv(tables / "summary.csv")
overall = {("vanilla", "vanilla"): "vanilla", ("BM25", "BM25"): "BM25", ("genread", "genread"): "genread"}
for (model, method), g in long_t.groupby(["model", "method"]):
    assert g["n"].sum() == 14267 and len(g) == 16, (model, method)
    expected = summary[(summary.model == model) & (summary.n == 14267)].iloc[0][method]
    assert np.isclose((g["accuracy"] * g["n"]).sum() / 14267, expected), (model, method)
sizes = long_t.groupby(["model", "method"]).apply(lambda g: tuple(g.sort_values("prop")["n"]), include_groups=False)
assert sizes.nunique() == 1                                                     # mọi mô hình và cách cùng số câu mỗi loại quan hệ
sm = pd.read_csv(tables / "accuracy_by_relation_summary.csv")
v, b = sm[sm.method == "vanilla"].set_index("prop"), sm[sm.method == "BM25"].set_index("prop")
assert len(v) == len(b) == 16 and (v.n_models == 9).all() and v.n.sum() == 14267
assert int((b.mean_corr < v.mean_corr).sum()) == 16
assert list(b.index[b.bm25_gain_pts < 0]) == ["occupation"]
assert bool((v.mean_corr > 0).all()) and int((v.mean_corr >= 0.2).sum()) == 11
assert v.mean_acc.idxmax() == "capital" and v.mean_acc.idxmin() == "producer" and 27 < v.mean_acc.max() / v.mean_acc.min() < 28
fig = ROOT / "results" / "figs" / "acc_corr_by_relation.png"
import matplotlib.image as mpimg  # noqa: E402
img = mpimg.imread(fig)
assert img.shape[1] >= 1000 and img.shape[0] >= 800 and fig.stat().st_size > 50_000
print("3 ok: độ chính xác theo loại quan hệ cộng lại ra đúng độ chính xác chung; các số trong tài liệu khớp; hình đủ lớn")
print("ALL OK")
