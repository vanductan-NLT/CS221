"""Kiểm scripts/analyze_answer_present_wrong.py (issue #38). Chạy trên CPU:
    python scripts/test_analyze_answer_present_wrong.py

1. Chuẩn hóa, khớp cả từ, gần đúng và thứ tự ưu tiên của nhóm nguyên nhân tự động đúng trên ví dụ nhỏ.
2. Trên dữ liệu giả có số đếm biết trước: số câu có đáp án, số câu sai, nhóm nguyên nhân, tỉ lệ theo loại quan hệ.
3. Trên bảng đã commit: mọi mô hình có đúng 3.644 câu có đáp án (bằng tổng `answer_in_passage`); các nhóm nguyên nhân cộng
   đúng số câu sai; số câu sai và số câu vanilla từng đúng khớp bảng 2×2 của issue #29 (số câu × recall@1 của các ô
   hại và sai cả hai); mẫu 20 câu đọc tay đủ 20 câu, đều có đáp án và đều sai, đúng số nhóm (5, 7, 3, 3, 2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402
import analyze_answer_present_wrong as apw  # noqa: E402

# 1.
assert apw.normalize("Sooraj R. Barjatya!") == "sooraj r barjatya"
assert not apw.whole_word_present("Marco Polo wrote", ["pol"]) and apw.whole_word_present("Spring Hill Methodist Church", ["church"])
assert apw.whole_word_present("born in Paris, France", ["Paris"]) and not apw.whole_word_present("Parisian", ["Paris"])
near = [("comedy", ["comedy film"], True), ("pop", ["J-pop", "Japanese pop"], True), ("Rock", ["Christian rock"], True),
        ("Isabelo Florentino de los Reyes", ["Isabelo de los Reyes"], True), ("United States", ["the United States of America"], True),
        ("Sooraj Barjatya", ["Sooraj R. Barjatya"], True), ("Pago Pago", ["Apia"], False), ("", ["Apia"], False),
        ("Germany", ["Lower Saxony", "Niedersachsen"], False)]
for pred, answers, expected in near:
    assert apw.is_near_miss(pred, answers) is expected, (pred, answers)
text = "Hanover is the capital of the federal state of Lower Saxony, Germany."
assert apw.cause_of("Germany", ["Lower Saxony"], text, True) == "other_candidate"      # trong đoạn, không phải đáp án
assert apw.cause_of("San José", ["Lower Saxony"], text, True) == "ignored_passage"     # không nằm trong đoạn
assert apw.cause_of("Saxony", ["Lower Saxony"], text, True) == "near_miss"             # một phần của đáp án
assert apw.cause_of("Saxony", ["Lower Saxony"], text, False) == "substring_only"       # đáp án giả được xét trước
assert apw.cause_of("", ["Lower Saxony"], text, True) == "ignored_passage"
print("1 ok: chuẩn hóa, khớp cả từ, gần đúng và thứ tự ưu tiên của nhóm nguyên nhân")

# 2. 8 câu giả: id 1..8, 6 câu có đáp án trong đoạn (id 1..6)
popqa = pd.DataFrame({"id": range(1, 9), "prop": ["genre"] * 4 + ["capital"] * 2 + ["sport"] * 2,
                      "question": [f"q{i}" for i in range(1, 9)],
                      "possible_answers": ['["comedy film"]', '["comedy film"]', '["Paris"]', '["Marco"]', '["Apia"]', '["Rome"]',
                                           '["football"]', '["football"]']})
texts = {1: "a 1994 comedy film.", 2: "a 1994 comedy film.", 3: "Lyon is in France. Paris is far.", 4: "Marcopolo was here.",
         5: "Pago Pago is the capital of American Samoa, near Apia.", 6: "Rome is old.", 7: "x", 8: "y"}
passages = {i: {"id": f"p{i}", "text": t, "hasanswer": False} for i, t in texts.items()}
top1 = pd.DataFrame({"id": range(1, 9), "answer_in_passage": [True] * 6 + [False] * 2})
vanilla = pd.DataFrame({"id": range(1, 9), "is_correct": [False, False, True, False, False, True, False, False]})
bm25 = pd.DataFrame({"id": range(1, 9), "is_correct": [False, True, False, False, False, True, False, False],
                     "pred": [" comedy", " comedy film", " Lyon", " Marco", " Pago Pago", " Rome", " x", " y"]})
info = apw.question_info(popqa, passages, top1.set_index("id")["answer_in_passage"])
row, by_rel = apw.analyze_model(vanilla, bm25, info)
# có đáp án: id 1..6; BM25 sai: id 1 (near_miss: "comedy"), 3 (other_candidate: "Lyon"), 4 (substring_only: "Marco" trong "Marcopolo"), 5 (other_candidate: "Pago Pago")
assert row["n_with_answer"] == 6 and row["n_wrong"] == 4 and row["vanilla_right_among_wrong"] == 1
assert (row["near_miss"], row["other_candidate"], row["substring_only"], row["ignored_passage"]) == (1, 2, 1, 0)
assert np.isclose(row["wrong_pct"], 4 / 6) and np.isclose(row["other_candidate_share"], 0.5)
rel = {r["prop"]: r for r in by_rel}
assert rel["genre"]["n_with_answer"] == 4 and rel["genre"]["n_wrong"] == 3 and rel["capital"]["n_wrong"] == 1
print("2 ok: dữ liệu giả cho đúng số câu có đáp án, câu sai, nhóm nguyên nhân và tỉ lệ theo loại quan hệ")

# 3. bảng đã commit
tables = ROOT / "results" / "tables"
by_model = pd.read_csv(tables / "answer_present_wrong_by_model.csv")
by_relation = pd.read_csv(tables / "answer_present_wrong_by_relation.csv")
sample = pd.read_csv(tables / "answer_present_wrong_sample.csv")
help_hurt = pd.read_csv(tables / "retrieval_help_hurt.csv")
top1_real = pd.read_csv(tables / "bm25_top1_has_answer.csv")
n_with = int(top1_real.answer_in_passage.astype(str).str.lower().eq("true").sum())
assert n_with == 3644 and by_model.model.nunique() >= 9
for r in by_model.itertuples():
    assert r.n_with_answer == n_with, r.model
    assert r.substring_only + r.near_miss + r.other_candidate + r.ignored_passage == r.n_wrong, r.model
    h = help_hurt[help_hurt.model == r.model].set_index(["vanilla_correct", "bm25_correct"])
    hurt, both_wrong = h.loc[(True, False)], h.loc[(False, False)]
    expected_wrong = hurt["count"] * hurt["recall_at_1"] + both_wrong["count"] * both_wrong["recall_at_1"]
    assert np.isclose(r.n_wrong, expected_wrong, atol=1e-6), (r.model, r.n_wrong, expected_wrong)
    assert np.isclose(r.vanilla_right_among_wrong, hurt["count"] * hurt["recall_at_1"], atol=1e-6), r.model
    g = by_relation[by_relation.model == r.model]
    assert g.n_with_answer.sum() == n_with and g.n_wrong.sum() == r.n_wrong, r.model
assert len(sample) == 20 and set(sample.id) == set(apw.MANUAL_CAUSE) and sample.manual_cause.notna().all()
assert sample.manual_cause.value_counts().to_dict() == {"right_in_meaning": 7, "fake_answer": 5, "narrow_gold": 3, "wrong_pick": 3, "ignored": 2}
gptj = next(v for k, v in aggregate.load_groups(ROOT / "results" / "slim").items() if k[3] == 14267 and "gpt-j" in k[0])
wrong_ids = set(gptj["BM25"].loc[~gptj["BM25"]["is_correct"], "id"])
has_ids = set(top1_real.loc[top1_real.answer_in_passage.astype(str).str.lower().eq("true"), "id"])
assert set(sample.id) <= wrong_ids and set(sample.id) <= has_ids
print("3 ok: bảng đã commit khớp bảng 2×2 của #29; mẫu 20 câu đọc tay đủ 20, đều có đáp án và đều sai, đúng số nhóm")
print("ALL OK")
