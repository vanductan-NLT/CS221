"""Kiểm scripts/compare_paper_and_qwen.py trên dữ liệu giả và dữ liệu thật đã commit.

Chạy trên CPU:
    python scripts/test_compare_paper_and_qwen.py
"""
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_paper_and_qwen as cpq  # noqa: E402

# 1. Kiểm hàm auroc_fast và mcnemar_exact trên ca biết trước
assert np.isclose(cpq.auroc_fast([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
assert np.isclose(cpq.auroc_fast([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
assert np.isclose(cpq.auroc_fast([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

r_same = cpq.mcnemar_exact([True, False, True, False], [True, False, True, False])
assert r_same["diff"] == 0.0 and r_same["p_value"] == 1.0
r_diff = cpq.mcnemar_exact([True] * 20 + [False] * 10, [True] * 5 + [False] * 25)
assert r_diff["n_a_only"] == 15 and r_diff["n_b_only"] == 0 and r_diff["p_value"] < 1e-4
print("1 ok: auroc_fast và mcnemar_exact đúng trên các ca biết trước")

# 2. Kiểm bảng đối chiếu paper_vs_repro.csv
pvr = pd.read_csv(ROOT / "results" / "tables" / "paper_vs_repro.csv")
acc = pvr[pvr["metric"] == "accuracy"]
assert len(acc) == 16, f"Kỳ vọng 16 ô accuracy (6 vanilla + 6 BM25 + 2 genread + 2 adaptive), nhận {len(acc)}"
assert acc["abs_diff_pp"].mean() < 0.30, acc["abs_diff_pp"].mean()
assert acc["abs_diff_pp"].max() < 0.75, acc["abs_diff_pp"].max()
print(f"2 ok: paper_vs_repro.csv đủ {len(acc)} ô accuracy, MAD = {acc['abs_diff_pp'].mean():.2f}pp (< 0.30pp)")

# 3. Kiểm bảng kiểm định Qwen2.5 và cơ chế
mc = pd.read_csv(ROOT / "results" / "tables" / "qwen_vs_paper_mcnemar.csv")
mech = pd.read_csv(ROOT / "results" / "tables" / "qwen_mechanisms_by_model.csv")
concl = pd.read_csv(ROOT / "results" / "tables" / "four_conclusions_check.csv")
assert len(mech) == 9 and len(concl) == 9
assert len(mc) == 36  # 12 cặp × 3 chế độ (vanilla, BM25, BM25_in_passage)

# Xác nhận Qwen2.5 3B thắng GPT-J 6B và OPT 6.7B có ý nghĩa thống kê ở cả 3 chế độ
for setting in ("vanilla", "BM25", "BM25_in_passage"):
    for comp in ("GPT-J 6B", "OPT 6.7B"):
        row = mc[(mc["setting"] == setting) & (mc["model_a"] == "Qwen2.5 3B") & (mc["model_b"] == comp)].iloc[0]
        assert row["diff_pp"] > 1.5 and row["mcnemar_p"] < 1e-15, (setting, comp, row.to_dict())
        assert row["strict_diff_pp"] > 1.5 and row["strict_mcnemar_p"] < 1e-15, (setting, comp, row.to_dict())
print("3 ok: bảng kiểm định McNemar, cơ chế và 4 kết luận khớp kỳ vọng định lượng")

# 4. Kiểm 3 biểu đồ PNG đã sinh và đủ kích thước
for fname in ("paper_vs_repro.png", "acc_vs_pop.png", "qwen_vs_paper_mechanisms.png"):
    fpath = ROOT / "results" / "figs" / fname
    assert fpath.exists() and fpath.stat().st_size > 20_000, f"Thiếu hoặc rỗng: {fpath}"
    img = mpimg.imread(fpath)
    assert img.shape[0] >= 500 and img.shape[1] >= 1000, (fname, img.shape)
print("4 ok: đủ 3 hình PNG hợp lệ trong results/figs/")
print("ALL OK")
