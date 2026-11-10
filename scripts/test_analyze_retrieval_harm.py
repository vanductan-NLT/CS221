"""Kiểm scripts/analyze_retrieval_harm.py (issue #37). Chạy trên CPU: python scripts/test_analyze_retrieval_harm.py

1. Nhóm nổi tiếng chia bằng số câu theo log(s_pop) tăng dần; phân loại 4 ô đúng.
2. Trên dữ liệu giả có số đếm biết trước: số câu bị hại theo nhóm, lost_share, hurt_share_of_all, recall@1 và
   pred_in_passage (không phân biệt hoa thường) từng ô.
3. Trên bảng đã commit: tổng số câu bị hại theo nhóm nổi tiếng và 4 ô của mọi mô hình khớp bảng 2×2 của issue #29
   (`retrieval_help_hurt.csv`), mỗi mô hình đủ 14.267 câu; bảng câu hỏi chỉ có câu bị hại từ 6 mô hình trở lên.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_retrieval_harm as arh  # noqa: E402

# 1.
s_pop = pd.Series([1000, 10, 100, 100000, 1, 10000, 5, 50, 500, 5000], index=range(10, 20))
q = arh.popularity_quintile(s_pop)
assert q.loc[14] == 1 and q.loc[16] == 1 and q.loc[13] == 5 and q.loc[15] == 5 and q.value_counts().eq(2).all()
assert [arh.cell_of(*x) for x in [(True, True), (True, False), (False, True), (False, False)]] == ["both_right", "hurt", "help", "both_wrong"]
print("1 ok: nhóm nổi tiếng chia đều theo log(s_pop); phân loại 4 ô đúng")

# 2. 10 câu: id 10..19, nổi tiếng tăng dần theo thứ tự của q (nhóm 1: 14, 16; nhóm 2: 11, 17; ...; nhóm 5: 13, 15)
ids = list(range(10, 20))
vanilla = pd.DataFrame({"id": ids, "is_correct": [True, True, False, True, True, False, True, False, True, False]})
bm25 = pd.DataFrame({"id": ids, "is_correct": [False, True, True, False, False, False, True, True, False, True],
                     "pred": [" Paris", " rome", " X", " Berlin", "", " y", " Madrid", " Z", " Oslo", " Lima"]})
passages = {i: "Paris is big. Madrid city. Berlin wall." if i in (10, 16) else "nothing about it" for i in ids}
has = pd.Series([True, False, False, False, False, False, True, False, False, False], index=ids)
by_pop, cells, hurt_ids = arh.analyze_model(vanilla, bm25, q, passages, has)
cells = {c["cell"]: c for c in cells}
assert hurt_ids == {10, 13, 14, 18}, hurt_ids                                    # vanilla đúng, BM25 sai
assert cells["hurt"]["n"] == 4 and cells["help"]["n"] == 3 and cells["both_right"]["n"] == 2 and cells["both_wrong"]["n"] == 1
assert np.isclose(cells["hurt"]["recall_at_1"], 0.25)                           # chỉ id 10 có đáp án trong đoạn
assert np.isclose(cells["hurt"]["pred_in_passage"], 0.25)                       # chỉ " Paris" của id 10 nằm trong đoạn; " Berlin" của id 13 thì không
assert np.isclose(sum(r["hurt_share_of_all"] for r in by_pop), 1) and sum(r["n"] for r in by_pop) == 10
g = {r["quintile"]: r for r in by_pop}
# nhóm 5 = id 13 (vanilla đúng, BM25 sai) và id 15 (vanilla sai): 1 câu vanilla đúng, câu đó bị hại
assert g[5]["n"] == 2 and g[5]["hurt"] == 1 and g[5]["vanilla_correct"] == 1 and np.isclose(g[5]["lost_share"], 1.0)
# nhóm 2 = id 11 (đúng cả hai) và id 17 (vanilla sai): không câu nào bị hại
assert g[2]["hurt"] == 0 and np.isclose(g[2]["lost_share"], 0.0) and np.isclose(g[2]["hurt_share_of_all"], 0.0)
assert all(np.isclose(g[k]["hurt_share_of_all"], 0.25) for k in (1, 3, 4, 5))                # 4 câu bị hại, mỗi nhóm 1 câu trừ nhóm 2
print("2 ok: dữ liệu giả cho đúng số câu bị hại theo nhóm, lost_share, hurt_share_of_all, recall@1, pred_in_passage")

# 3. bảng đã commit
tables = ROOT / "results" / "tables"
by_pop = pd.read_csv(tables / "retrieval_harm_by_popularity.csv")
cells = pd.read_csv(tables / "retrieval_harm_cells.csv")
help_hurt = pd.read_csv(tables / "retrieval_help_hurt.csv")
questions = pd.read_csv(tables / "retrieval_harm_questions.csv")
name = {(True, False): "hurt", (False, True): "help", (True, True): "both_right", (False, False): "both_wrong"}
assert set(by_pop.model) == set(help_hurt.model) == set(cells.model) and by_pop.model.nunique() >= 9
for model, g in by_pop.groupby("model"):
    assert g.n.sum() == 14267 and np.isclose(g.hurt_share_of_all.sum(), 1), model
    h = help_hurt[help_hurt.model == model]
    for r in h.itertuples():
        assert cells[(cells.model == model) & (cells.cell == name[(r.vanilla_correct, r.bm25_correct)])].n.iloc[0] == r.count, (model, r)
    assert g.hurt.sum() == h[(h.vanilla_correct) & (~h.bm25_correct)]["count"].iloc[0], model
assert questions.n_models_hurt.min() >= 6 and questions.n_models_hurt.max() <= 9 and questions.id.is_unique
popqa = pd.read_csv(ROOT / "upstream" / "data" / "popQA.tsv", sep="\t")
assert set(questions.id) <= set(popqa.id) and (questions.n_models_hurt == 9).sum() == 19
print("3 ok: bảng đã commit khớp bảng 2×2 của #29 ở mọi mô hình; 19 câu bị hại ở cả 9 mô hình")

# 4. đoạn dùng chung: đoạn "p1" chọn cho 5 chủ thể khác nhau; đoạn "p2" chọn cho 3 câu về cùng một chủ thể
toy = pd.DataFrame({"id": range(1, 10), "subj": ["a", "b", "c", "d", "e", "x", "x", "x", "y"],
                    "prop": ["p"] * 5 + ["q"] * 4})
toy_passages = {1: {"id": "p1"}, 2: {"id": "p1"}, 3: {"id": "p1"}, 4: {"id": "p1"}, 5: {"id": "p1"},
                6: {"id": "p2"}, 7: {"id": "p2"}, 8: {"id": "p2"}, 9: {"id": "p3"}}
hub_ids, n_hub = arh.hub_question_ids(toy, toy_passages)
assert hub_ids == {1, 2, 3, 4, 5} and n_hub == 1                                  # p2 chỉ có 1 chủ thể nên không phải đoạn dùng chung
rel = arh.hub_by_relation(toy, hub_ids).set_index("prop")
assert rel.loc["p", "n_hub"] == 5 and np.isclose(rel.loc["p", "hub_share"], 1.0) and rel.loc["q", "n_hub"] == 0
v = pd.DataFrame({"id": range(1, 10), "is_correct": [True, True, False, False, False, True, True, False, False]})
b = pd.DataFrame({"id": range(1, 10), "is_correct": [False, False, False, False, False, True, True, True, True]})
ans = pd.Series([False] * 5 + [True] * 4, index=range(1, 10))
rows = {r["group"]: r for r in arh.hub_summary(v, b, hub_ids, ans)}
assert rows["hub"]["n"] == 5 and rows["hub"]["hurt"] == 2 and np.isclose(rows["hub"]["hurt_share_of_all"], 1.0)
assert np.isclose(rows["hub"]["vanilla_acc"], 0.4) and np.isclose(rows["hub"]["bm25_acc"], 0.0) and np.isclose(rows["hub"]["recall_at_1"], 0.0)
assert rows["non_hub"]["n"] == 4 and rows["non_hub"]["hurt"] == 0 and np.isclose(rows["non_hub"]["recall_at_1"], 1.0)
hub_t = pd.read_csv(tables / "retrieval_harm_hub.csv")
rel_t = pd.read_csv(tables / "retrieval_harm_hub_by_relation.csv")
assert rel_t.n.sum() == 14267 and rel_t.n_hub.sum() == hub_t[hub_t.group == "hub"].n.iloc[0]
for model, g in hub_t.groupby("model"):
    g = g.set_index("group")
    assert g.n.sum() == 14267 and g.hurt.sum() == by_pop[by_pop.model == model].hurt.sum() and np.isclose(g.hurt_share_of_all.sum(), 1), model
    assert g.loc["hub", "bm25_acc"] < g.loc["hub", "vanilla_acc"] and g.loc["non_hub", "bm25_acc"] > g.loc["non_hub", "vanilla_acc"], model
print("4 ok: đoạn dùng chung đúng trên dữ liệu giả; bảng đã commit khớp (số câu, số câu bị hại, hub làm BM25 kém đi ở cả 9 mô hình)")
print("ALL OK")
