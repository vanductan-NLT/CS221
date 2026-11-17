"""Kiểm scripts/compare_code_goc_runs.py và các file kết quả của PR #68. Chạy trên CPU:
    python scripts/test_compare_code_goc_runs.py

1. McNemar chính xác đúng ở các trường hợp đã biết.
2. So hai bên trên dữ liệu giả có số đếm biết trước.
3. Trên các file đã commit: độ chính xác của lượt code gốc khớp summary do Thịnh ghi trong zip; mỗi bên đủ 14.267 `id`;
   GenRead (GPT-Neo 1.3B và OPT 1.3B) có mặt trong `summary.csv`, khớp summary của Thịnh, và không làm đổi các cột khác của dòng đó.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402
import compare_code_goc_runs as cmp  # noqa: E402

# 1.
assert cmp.mcnemar_exact(5, 5) == 1.0 and cmp.mcnemar_exact(0, 0) == 1.0
assert np.isclose(cmp.mcnemar_exact(0, 10), 2 / 1024) and np.isclose(cmp.mcnemar_exact(10, 0), 2 / 1024)
assert np.isclose(cmp.mcnemar_exact(1, 3), 2 * (1 + 4) / 16)                       # n = 4, đuôi = (C(4,0)+C(4,1))/16
print("1 ok: McNemar chính xác đúng ở các trường hợp đã biết")

# 2.
ids = list(range(10))
a = pd.DataFrame({"id": ids, "prop": "p", "s_pop": 5, "is_correct": [True] * 4 + [False] * 6})
b = pd.DataFrame({"id": ids, "prop": "p", "s_pop": 5, "is_correct": [True, True, False, False, True, True, True, False, False, False]})
table = cmp.compare({("m", "fp32", "vanilla"): a}, {("m", "fp32", "vanilla"): b})
r = table.iloc[0]
# chỉ đúng ở A: id 2, 3; chỉ đúng ở B: id 4, 5, 6; giống nhau: 5 câu
assert (r["only_code_goc"], r["only_nhom"], r["n"]) == (2, 3, 10) and np.isclose(r["same_correct_share"], 0.5)
assert np.isclose(r["acc_code_goc"], 0.4) and np.isclose(r["acc_nhom"], 0.5) and np.isclose(r["diff_points"], 10.0)
assert np.isclose(r["mcnemar_p"], cmp.mcnemar_exact(2, 3))
try:
    cmp.compare({("m", "fp32", "vanilla"): a}, {("m", "fp32", "vanilla"): b.assign(id=range(100, 110))})
    raise SystemExit("FAIL: expected ValueError for different ids")
except ValueError:
    pass
print("2 ok: so hai bên trên dữ liệu giả đúng số đếm, chênh lệch, p-value; khác tập id thì báo lỗi")

# 3.
code_goc = cmp.load(ROOT / "results" / "code_goc")
assert len(code_goc) == 4
thinh_summary = {("EleutherAI_gpt-neo-1.3B", "vanilla"): 0.11165626971332446, ("EleutherAI_gpt-neo-2.7B", "vanilla"): 0.12076820635031892,
                 ("EleutherAI_gpt-neo-1.3B", "BM25"): 0.20614004345692857, ("EleutherAI_gpt-neo-2.7B", "BM25"): 0.2294806196116913}
for (model, dtype, method), df in code_goc.items():
    assert len(df) == 14267 and df["id"].is_unique
    assert np.isclose(df["is_correct"].mean(), thinh_summary[(model, method)]), (model, method)
committed = pd.read_csv(ROOT / "results" / "tables" / "code_goc_vs_nhom.csv")
assert len(committed) == 4 and (committed["n"] == 14267).all()
fresh = cmp.compare(code_goc, cmp.load(ROOT / "results" / "slim"))
assert np.allclose(committed.drop(columns=["model", "dtype", "method"]).values, fresh.drop(columns=["model", "dtype", "method"]).values)

summary = pd.read_csv(ROOT / "results" / "tables" / "summary.csv")
genread = {("EleutherAI_gpt-neo-1.3B", 14267): 0.11649260531295998, ("EleutherAI_gpt-neo-1.3B", 300): 0.14,
           ("facebook_opt-1.3b", 14267): 0.1223803182168641, ("facebook_opt-1.3b", 300): 0.12}
for (model, n), acc in genread.items():
    row = summary[(summary.model == model) & (summary.n == n)].iloc[0]
    assert np.isclose(row["genread"], acc), (model, n)
    assert not np.isnan(row["vanilla"]) and not np.isnan(row["BM25"]) and not np.isnan(row["adaptive"]), (model, n)
others = summary[~summary.model.isin(["EleutherAI_gpt-neo-1.3B", "facebook_opt-1.3b"])]
assert others["genread"].isna().all()
print("3 ok: code gốc khớp summary của Thịnh; GenRead có trong summary.csv và khớp summary của Thịnh; mô hình khác không có GenRead")
print("ALL OK")
