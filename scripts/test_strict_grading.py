"""Kiểm scripts/strict_grading.py (issue #35). Chạy trên CPU: python scripts/test_strict_grading.py

1. Quy tắc chấm chặt đúng trên các ca thật: "US" trong "Russia" không còn đúng; "Los Angeles, California" vẫn đúng;
   gạch nối, dấu chấm, hoa thường không làm sai.
2. `lenient_kind` phân loại đúng "inside_word" và "not_leading".
3. `add_strict_column`: giữ nguyên các cột cũ và kiểu xuống dòng, chạy lại cho cùng kết quả.
4. Trên dữ liệu đã commit: công thức chấm cũ tính lại từ cột `pred` ra đúng `is_correct` ở mọi dòng của mọi file; cột
   `is_correct_strict` khớp tính lại từ `pred` và đáp án; bảng tóm tắt khớp (độ chính xác cũ trừ chặt bằng chênh số câu chia
   cho số câu); thứ hạng các mô hình theo vanilla và theo BM25 không đổi khi chấm chặt.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402
import strict_grading as sg  # noqa: E402


def lenient(pred, answers):
    """Quy tắc chấm của code gốc (`upstream/run_model.py` dòng 275)."""
    return any(a in pred or a.lower() in pred or a.capitalize() in pred for a in answers)


# 1.
cases = [("Steven Spielberg", ["Steven Spielberg"], True), ("Russia", ["United States of America", "US"], False),
         ("Australia", ["US"], False), ("United States", ["Italy", "IT"], False), ("Toronto", ["Ontario", "ON"], False),
         ("Los Angeles, California", ["Los Angeles"], True), ("Tikhvinsky", ["Tikhvin"], False),
         ("Giacomo Guardi", ["Francesco Guardi", "Guardi"], False), ("beach soccer", ["football", "soccer"], False),
         ("Porto Novo", ["Porto-Novo"], True), ("Thomas D'Alesandro Jr.", ["Thomas D'Alesandro, Jr."], True),
         ("First Person Shooter", ["first-person shooter"], True), ("A.H.M. Jones", ["A. H. M. Jones"], True),
         ("Apia<|endoftext|>", ["Apia"], True), ("", ["Apia"], False), ("Apia", [], False)]
for pred, answers, expected in cases:
    assert sg.is_strict_correct(pred, answers) is expected, (pred, answers)
assert lenient("Russia", ["US"]) and lenient("Toronto", ["ON"]) and lenient("United States", ["IT"])      # cách cũ nhận cả ba
print("1 ok: chấm chặt đúng trên các ca thật (US trong Russia, ON trong Toronto, IT trong United đều bị loại)")

# 2.
assert sg.lenient_kind("Russia", ["US"]) == "inside_word"
assert sg.lenient_kind("beach soccer", ["soccer"]) == "not_leading"
assert sg.lenient_kind("Roman Catholicism", ["Catholic"]) == "inside_word"
print("2 ok: phân loại đáp án giả: nằm trong một chữ khác, hoặc là cả từ nhưng không đứng đầu")

# 3.
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "run.csv"
    rows = ["id,prop,s_pop,pred,is_correct,conf", "1,country,5, Russia,True,-0.5", "2,country,6, Italy,True,-0.1", "3,country,7, ,False,"]
    f.write_bytes(("\r\n".join(rows) + "\r\n").encode("utf-8"))
    answers = {1: ["US"], 2: ["Italy", "IT"], 3: ["Paris"]}
    assert sg.add_strict_column(f, answers) == 3
    assert sg.add_strict_column(f, answers) == 0                                  # chạy lại: không dòng nào đổi
    raw = f.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\r\n") == 4 and b"\n" not in raw.replace(b"\r\n", b"")
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    assert df["is_correct_strict"].tolist() == ["False", "True", "False"] and df["conf"].tolist() == ["-0.5", "-0.1", ""]
    assert df["pred"].tolist() == [" Russia", " Italy", " "]
print("3 ok: thêm cột giữ nguyên các cột cũ và kiểu xuống dòng; chạy lại cho cùng kết quả")

# 4.
popqa = pd.read_csv(ROOT / "upstream" / "data" / "popQA.tsv", sep="\t")
answers_by_id = {int(r.id): json.loads(r.possible_answers) for r in popqa.itertuples(index=False)}
files = [p for p in sorted((ROOT / "results" / "slim").glob("*.csv")) if aggregate.parse_name(p.name)]
assert len(files) >= 36
rows_checked = 0
for p in files:
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert "is_correct_strict" in d.columns, p.name
    old = d["is_correct"].str.lower().eq("true")
    recomputed_old = pd.Series([lenient(str(pr), answers_by_id[int(i)]) for i, pr in zip(d["id"], d["pred"])])
    assert (recomputed_old.values == old.values).all(), f"chấm cũ tính lại khác is_correct ở {p.name}"
    strict = d["is_correct_strict"].str.lower().eq("true")
    recomputed_strict = pd.Series([sg.is_strict_correct(pr, answers_by_id[int(i)]) for i, pr in zip(d["id"], d["pred"])])
    assert (recomputed_strict.values == strict.values).all(), p.name
    rows_checked += len(d)
summary = pd.read_csv(ROOT / "results" / "tables" / "strict_grading_summary.csv")
for r in summary.itertuples():
    assert np.isclose(r.acc_old - r.acc_strict, (r.lenient_only - r.strict_only) / r.n), (r.model, r.method, r.n)
full = summary[summary.n == 14267]
for method in ("vanilla", "BM25"):
    g = full[full.method == method].set_index("model")
    assert list(g.sort_values("acc_old").index) == list(g.sort_values("acc_strict").index), method
assert full[full.method.isin(["vanilla", "BM25", "genread"])].lenient_only.min() > 0
print(f"4 ok: chấm cũ tính lại khớp is_correct ở cả {rows_checked:,} dòng của {len(files)} file; cột strict khớp tính lại; "
      "bảng tóm tắt khớp; thứ hạng mô hình không đổi")
print("ALL OK")
