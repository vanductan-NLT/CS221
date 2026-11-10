"""Kiểm scripts/adaptive_by_confidence.py (issue #34). Chạy trên CPU: python scripts/test_adaptive_by_confidence.py

1. Thêm tham số `score_col` không làm đổi adaptive gốc: truyền cột chính là log(s_pop) thì ra đúng số cũ.
2. `summary.csv` đang commit vẫn khớp với kết quả tính lại bằng aggregate.py sau khi tách hàm đọc nhóm file.
3. auroc: tách hoàn hảo là 1, đảo ngược là 0, điểm giống nhau hết là 0,5.
4. Trên dữ liệu giả mà conf báo đúng câu nào mô hình trả lời được, adaptive theo conf thắng adaptive theo
   độ nổi tiếng (vốn không liên quan), và hai cách dùng cùng các lần chia (so cặp được).
5. confidence_column: điền câu trả lời rỗng bằng điểm thấp nhất; lượt không có conf trả về None.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402
import adaptive_by_confidence as abc  # noqa: E402


def fake_runs(n=3000, seed=0, conf_informative=True):
    """Vanilla đúng hay sai tùy một 'độ chắc' ẩn; s_pop không liên quan. BM25 đúng đều khoảng 35%."""
    rng = np.random.default_rng(seed)
    props = [f"rel{i}" for i in range(10)]
    base = pd.DataFrame({"id": rng.permutation(10 * n)[:n], "prop": rng.choice(props, n),
                         "s_pop": np.exp(rng.uniform(1, 12, n)).round().astype(int) + 1})
    hidden = rng.uniform(0, 1, n)                      # độ chắc thật của mô hình với câu này
    vanilla = base.assign(is_correct=rng.random(n) < 0.05 + 0.8 * hidden ** 2)
    noise = rng.uniform(0, 1, n)
    vanilla["conf"] = -4 * (1 - (hidden if conf_informative else noise))   # trong khoảng [-4, 0]
    bm25 = base.assign(is_correct=rng.random(n) < 0.35)
    return vanilla, bm25


# 1. score_col bằng log(s_pop) cho ra đúng adaptive gốc.
for seed in range(3):
    v, b = fake_runs(seed=seed)
    default = aggregate.adaptive_eval(v, b)
    same = aggregate.adaptive_eval(v.assign(logpop=np.log(v["s_pop"].astype(float))), b, score_col="logpop")
    for key in ["n", "param_acc", "ret_acc", "hybrid_acc", "hybrid_std", "frac_retrieval"]:
        assert np.isclose(default[key], same[key]), (seed, key, default[key], same[key])
    assert np.allclose(default["hybrid_runs"], same["hybrid_runs"]) and len(default["hybrid_runs"]) == 100
print("1 ok: score_col bằng log(s_pop) cho đúng số của adaptive gốc trên 3 bộ dữ liệu giả")

# 2. summary.csv đang commit khớp với kết quả tính lại.
committed = pd.read_csv(ROOT / "results" / "tables" / "summary.csv")
fresh = aggregate.summarize(ROOT / "results" / "slim")
assert list(committed.columns) == list(fresh.columns) and len(committed) == len(fresh), (len(committed), len(fresh))
for col in committed.columns:
    if pd.api.types.is_numeric_dtype(committed[col]):
        assert np.allclose(committed[col], fresh[col], equal_nan=True, atol=1e-12), col
    else:
        assert (committed[col] == fresh[col]).all(), col
print("2 ok: summary.csv đang commit khớp với kết quả tính lại từ results/slim")

# 3. auroc.
assert np.isclose(abc.auroc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
assert np.isclose(abc.auroc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
assert np.isclose(abc.auroc([5, 5, 5, 5], [0, 1, 0, 1]), 0.5)
print("3 ok: auroc đúng ở ba trường hợp đã biết")

# 4. conf có thông tin thì thắng adaptive theo độ nổi tiếng; conf là nhiễu thì không thắng.
v, b = fake_runs(seed=1)
r = abc.compare(v.drop(columns=["conf"]).assign(conf=v["conf"].astype(str)), b)   # đọc từ file thì conf là chuỗi
assert r["adaptive_conf"] > r["adaptive_pop"] + 0.02 and r["conf_wins"] >= 95 and r["n_splits"] == 100, r
assert r["auroc_conf"] > 0.7 and abs(r["auroc_pop"] - 0.5) < 0.05, r
v, b = fake_runs(seed=1, conf_informative=False)
r = abc.compare(v, b)
assert r["adaptive_conf"] < r["adaptive_pop"] + 0.02 and abs(r["auroc_conf"] - 0.5) < 0.05, r
print("4 ok: conf có thông tin thắng độ nổi tiếng (AUROC cao, thắng gần mọi lần chia); conf nhiễu thì không")

# 5. confidence_column.
frame = pd.DataFrame({"conf": ["-1.5", "", "-0.2"]})
assert abc.confidence_column(frame).tolist() == [-1.5, -1.5, -0.2]
assert abc.confidence_column(pd.DataFrame({"x": [1]})) is None
assert abc.confidence_column(pd.DataFrame({"conf": ["", ""]})) is None
print("5 ok: conf rỗng điền bằng điểm thấp nhất; lượt không có conf thì bỏ qua")
print("ALL OK")
