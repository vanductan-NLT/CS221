"""Kiểm scripts/aggregate.py (issue #24). Chạy trên CPU: python scripts/test_aggregate.py

1. Trên dữ liệu giả, adaptive_eval của script cho cùng số với hàm trong notebook chính
   (notebooks/adaptive_retrieval_repro.py, mục 7).
2. Đọc đúng hai kiểu tên file; summary.csv có adaptive khi đủ vanilla và BM25.
"""
import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402


def notebook_adaptive_eval():
    """Lấy đúng hàm adaptive_eval trong notebook chính, không chạy phần còn lại của notebook."""
    src = (ROOT / "notebooks" / "adaptive_retrieval_repro.py").read_text(encoding="utf-8")
    # File có dòng `%pip` của notebook nên không parse cả file; cắt riêng hàm tới dòng định nghĩa kế tiếp.
    start = src.index("def adaptive_eval(")
    end = src.index("\ndef ", start + 1)
    fn = ast.parse(src[start:end]).body[0]
    assert isinstance(fn, ast.FunctionDef) and fn.name == "adaptive_eval"
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module([fn], []), "notebook", "exec"), ns)
    return ns["adaptive_eval"]


def fake_runs(n=2000, seed=0):
    """Hai bảng kết quả giả: vanilla đúng nhiều ở câu nổi tiếng, BM25 đúng đều hơn."""
    rng = np.random.default_rng(seed)
    props = [f"rel{i}" for i in range(12)]
    s_pop = np.exp(rng.uniform(1, 12, n)).round().astype(int) + 1
    prop = rng.choice(props, n)
    p_vanilla = 1 / (1 + np.exp(-(np.log(s_pop) - 8)))
    base = pd.DataFrame({"id": rng.permutation(10 * n)[:n], "question": [f"q{i}" for i in range(n)],
                         "prop": prop, "s_pop": s_pop})
    vanilla = base.assign(is_correct=rng.random(n) < p_vanilla)
    bm25 = base.assign(is_correct=rng.random(n) < 0.35)
    return vanilla, bm25


reference = notebook_adaptive_eval()
for seed in range(3):
    v, b = fake_runs(seed=seed)
    ours, theirs = aggregate.adaptive_eval(v, b), reference(v, b)
    for key in ["n", "param_acc", "ret_acc", "hybrid_acc", "hybrid_std", "frac_retrieval"]:
        assert np.isclose(ours[key], theirs[key]), (seed, key, ours[key], theirs[key])
print("1 ok: adaptive_eval khớp hàm của notebook trên 3 bộ dữ liệu giả")

v, b = fake_runs(n=600, seed=7)
with tempfile.TemporaryDirectory() as tmp:
    slim = Path(tmp)
    cols = ["id", "prop", "s_pop", "pred", "is_correct"]
    v.assign(pred="x")[cols].to_csv(slim / "model=Org_m-1B-input=popqa-full-method=vanilla-shots=15-n=600.csv", index=False)
    b.assign(pred="y", has_answer=True, retrieval_id=1)[cols + ["has_answer", "retrieval_id"]].to_csv(
        slim / "model=Org_m-1B-input=popqa-shot-passages-full-method=BM25-shots=15-n=600.csv", index=False)
    b.head(300).assign(pred="z")[cols].to_csv(slim / "Org_m-1B__fp16__BM25__shots15.csv", index=False)
    (slim / "notes.csv").write_text("a\n1\n")
    table = aggregate.summarize(slim)
assert len(table) == 2, table
full = table[table.n == 600].iloc[0]
assert full["dtype"] == "fp32" and np.isclose(full["vanilla"], v.is_correct.mean())
assert np.isclose(full["BM25"], b.is_correct.mean()) and np.isnan(full["genread"])
assert np.isclose(full["adaptive"], reference(v, b)["hybrid_acc"])
part = table[table.n == 300].iloc[0]
assert part["dtype"] == "fp16" and np.isnan(part["adaptive"])
print("2 ok: đọc cả hai kiểu tên file; adaptive chỉ có ở nhóm đủ vanilla và BM25")
print("ALL OK")
