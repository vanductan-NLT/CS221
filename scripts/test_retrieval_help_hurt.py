"""Kiểm scripts/make_bm25_top1_table.py và scripts/retrieval_help_hurt.py (issue #29). Chạy trên CPU:
    python scripts/test_retrieval_help_hurt.py

1. Quy tắc "đáp án có trong đoạn văn" giống quy tắc chấm của code gốc (nguyên bản, chữ thường, viết hoa chữ đầu).
2. Bảng 2×2 trên dữ liệu giả có số đếm biết trước: 4 ô cộng 100%, recall@1 từng ô đúng, lệch `retrieval_id` thì báo lỗi.
3. Trên dữ liệu thật đã commit: bảng top-1 đủ 14.267 `id`; mọi lượt BM25 trong results/slim dùng đúng đoạn văn của
   bảng top-1; 4 ô của mọi mô hình cộng 100%; hai ô "hại" và "giúp" tính lại độc lập từ `is_correct` ra cùng số.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402
import make_bm25_top1_table as top1_script  # noqa: E402
import retrieval_help_hurt as rhh  # noqa: E402

# 1. quy tắc đáp án trong đoạn văn
f = top1_script.answer_in_text
assert f("He was born in Paris.", ["Paris"]) and f("he was born in paris", ["Paris"]) and f("Berlin is big", ["berlin"])
assert not f("born in PARIS", ["Paris"])           # đúng như quy tắc chấm: không hạ chữ của đoạn văn
assert not f("nothing here", ["Paris", "Rome"]) and f("a Rome b", ["Paris", "Rome"])
print("1 ok: quy tắc đáp án trong đoạn văn giống quy tắc chấm của code gốc")

# 2. bảng 2×2 trên dữ liệu giả: 10 câu, số đếm biết trước
ids = list(range(10))
vanilla = pd.DataFrame({"id": ids, "is_correct": [True] * 4 + [False] * 6})
bm25 = pd.DataFrame({"id": ids, "is_correct": [True, True, False, False, True, True, True, False, False, False],
                     "retrieval_id": [100 + i for i in ids]})
top1 = pd.DataFrame({"id": ids, "retrieval_id": [100 + i for i in ids],
                     "hasanswer_flag": [True] + [False] * 9,
                     "answer_in_passage": [True, True, False, False, True, False, True, False, False, True]})
cells = {(r["vanilla_correct"], r["bm25_correct"]): r for r in rhh.two_by_two(vanilla, bm25, top1)}
assert sum(r["count"] for r in cells.values()) == 10 and np.isclose(sum(r["pct"] for r in cells.values()), 1)
assert cells[(True, True)]["count"] == 2 and np.isclose(cells[(True, True)]["recall_at_1"], 1.0)      # id 0, 1
assert cells[(True, False)]["count"] == 2 and np.isclose(cells[(True, False)]["recall_at_1"], 0.0)    # id 2, 3
assert cells[(False, True)]["count"] == 3 and np.isclose(cells[(False, True)]["recall_at_1"], 2 / 3)  # id 4, 5, 6
assert cells[(False, False)]["count"] == 3 and np.isclose(cells[(False, False)]["recall_at_1"], 1 / 3)  # id 7, 8, 9
assert np.isclose(cells[(True, True)]["recall_flag"], 0.5)                                            # id 0 true, id 1 false
bad = bm25.assign(retrieval_id=bm25["retrieval_id"] + 1)
try:
    rhh.two_by_two(vanilla, bad, top1)
    raise SystemExit("FAIL: expected ValueError for a different retrieval_id")
except ValueError as e:
    assert "retrieval_id" in str(e)
print("2 ok: 2×2 giả đúng số đếm, tỉ lệ và recall@1 từng ô; lệch retrieval_id thì báo lỗi")

# 3. dữ liệu thật đã commit
popqa = pd.read_csv(ROOT / "upstream" / "data" / "popQA.tsv", sep="\t")
top1_real = pd.read_csv(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv")
assert len(top1_real) == 14267 and set(top1_real["id"]) == set(popqa["id"]) and top1_real["id"].is_unique
table = rhh.build_table(ROOT / "results" / "slim", ROOT / "results" / "tables" / "bm25_top1_has_answer.csv")
models = table.groupby(["model", "dtype"]).ngroups
assert models >= 9, models
for key, g in table.groupby(["model", "dtype"]):
    assert len(g) == 4 and np.isclose(g["pct"].sum(), 1) and g["count"].sum() == 14267, key
groups = aggregate.load_groups(ROOT / "results" / "slim")
(model, dtype, shots, n), runs = next((k, v) for k, v in sorted(groups.items()) if k[3] == 14267 and {"vanilla", "BM25"} <= set(v))
v, b = runs["vanilla"].set_index("id")["is_correct"], runs["BM25"].set_index("id")["is_correct"].reindex(runs["vanilla"]["id"].values)
hurt = float((v.values & ~b.values.astype(bool)).mean())
help_ = float((~v.values & b.values.astype(bool)).mean())
g = table[(table.model == model) & (table.dtype == dtype)].set_index(["vanilla_correct", "bm25_correct"])
assert np.isclose(g.loc[(True, False), "pct"], hurt) and np.isclose(g.loc[(False, True), "pct"], help_), model
print(f"3 ok: bảng top-1 đủ 14.267 id; {models} mô hình dùng đúng đoạn văn của bảng top-1, 4 ô cộng 100%, ô hại/giúp khớp tính độc lập")
print("ALL OK")
