"""Gộp kết quả rút gọn và tính adaptive retrieval cho mọi mô hình (issue #24).

Đọc mọi file trong `results/slim/`, nhóm theo (mô hình, độ chính xác số, số shot, số câu), ghép
vanilla với BM25 theo `id` rồi tính adaptive đúng thuật toán của notebook tác giả (đã port ở
`notebooks/adaptive_retrieval_repro.py`, mục 7):

- Với mỗi loại quan hệ: chia câu hỏi thành 10 bin theo log(s_pop). Chọn ngưỡng (theo bin) để hệ lai
  đúng nhiều nhất trên tập train: dưới ngưỡng dùng kết quả BM25, từ ngưỡng trở lên dùng vanilla.
- Chia train/test 75/25, lặp 100 lần, seed 633; báo trung bình trên tập test.

Ghi `results/tables/summary.csv`: model, dtype, shots, n, vanilla, BM25, genread, adaptive,
adaptive_std (độ lệch giữa 100 lần chia), frac_retrieval (tỉ lệ câu phải tra cứu).
Chưa có đủ vanilla và BM25 của một nhóm thì ô adaptive để trống.

Đọc được hai kiểu tên file (xem results/README.md):
- notebook nhóm: `{mô hình}__{dtype}__{cách}__shots{số shot}.csv`
- code gốc:      `model={mô hình}-input={tên lượt}-method={cách}-shots={số shot}-n={số câu}.csv`
  (fp32; tên lượt có chữ `fp16`, ví dụ `popqa-fp16-full`, thì là fp16)

Chạy trên CPU, vài giây:
    python scripts/aggregate.py
    python scripts/aggregate.py --slim-dir results/slim --out results/tables/summary.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["vanilla", "BM25", "genread"]
UPSTREAM_NAME = re.compile(
    r"^model=(?P<model>.+?)-input=(?P<alias>.+)-method=(?P<method>[^-]+)-shots=(?P<shots>\d+)-n=(?P<n>\d+)\.csv$")
NOTEBOOK_NAME = re.compile(r"^(?P<model>.+?)__(?P<dtype>[^_]+)__(?P<method>[^_]+)__shots(?P<shots>\d+)\.csv$")


def parse_name(name):
    """Thông tin lấy từ tên file: model, dtype, method, shots (n đọc từ số dòng)."""
    m = UPSTREAM_NAME.match(name)
    if m:
        # Code gốc chạy fp32; lượt nạp fp16 (GPT-J 6B, issue #28) có chữ `fp16` trong tên lượt.
        dtype = "fp16" if "fp16" in m["alias"].split("-") else "fp32"
        return dict(model=m["model"], dtype=dtype, method=m["method"], shots=int(m["shots"]))
    m = NOTEBOOK_NAME.match(name)
    if m:
        return dict(model=m["model"], dtype=m["dtype"], method=m["method"], shots=int(m["shots"]))
    return None


def read_run(path):
    df = pd.read_csv(path, keep_default_na=False)
    missing = {"id", "prop", "s_pop", "is_correct"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: thiếu cột {sorted(missing)}")
    if df["id"].duplicated().any():
        raise SystemExit(f"{path}: có `id` bị trùng")
    df["is_correct"] = df["is_correct"].map(lambda v: str(v).strip().lower() == "true")
    return df


def adaptive_eval(df_p, df_r, n_boot=100, split_proportion=0.75, seed=633, min_per_prop=8):
    """Adaptive retrieval như notebook tác giả: df_p = kết quả không tra cứu, df_r = có tra cứu.

    Trả về dict(n, param_acc, ret_acc, hybrid_acc, hybrid_std, frac_retrieval), hoặc None nếu
    không loại quan hệ nào có đủ `min_per_prop` câu.
    """
    if set(df_p["id"]) != set(df_r["id"]):
        raise ValueError("Hai bảng không cùng tập `id`; phải chạy trên cùng tập câu.")
    m = df_p[["id", "prop", "s_pop", "is_correct"]].merge(
        df_r[["id", "is_correct"]], on="id", suffixes=("_p", "_r"))
    m = m.sort_values("id").reset_index(drop=True)
    n = len(m)
    props = [p for p in m["prop"].unique() if (m["prop"] == p).sum() >= min_per_prop]
    if not props:
        return None
    ser_all = np.log(m["s_pop"].values.astype(float))
    c_p_all = m["is_correct_p"].values.astype(bool)
    c_r_all = m["is_correct_r"].values.astype(bool)
    prop_idx = {p: np.where(m["prop"].values == p)[0] for p in props}
    rng = np.random.RandomState(seed)
    hyb, par, ret, retn = [], [], [], []
    for _ in range(n_boot):
        mask = ["train"] * round(n * split_proportion) + ["test"] * round(n * (1 - split_proportion))
        assert len(mask) == n
        rng.shuffle(mask)
        mask = np.array(mask)
        tot_h = tot_p = tot_r = tot_ret = tot_n = 0.0
        for p in props:
            ix = prop_idx[p]
            ser, c, c_r, test = ser_all[ix], c_p_all[ix], c_r_all[ix], mask[ix] == "test"
            _, edges = np.histogram(ser)            # 10 bin theo log(s_pop)

            def counts(sel, correct):
                return np.histogram(ser[sel & correct], bins=edges)[0]

            train = ~test
            tr_c, tr_c_r = counts(train, c), counts(train, c_r)
            te_c, te_c_r = counts(test, c), counts(test, c_r)
            n_tr, n_te = int(train.sum()), int(test.sum())
            if n_tr == 0 or n_te == 0:
                continue
            nb = len(edges) - 1
            # Ngưỡng i: bin < i dùng BM25, bin >= i dùng vanilla. Chọn i tốt nhất trên train.
            scores = [(tr_c_r[:i].sum() + tr_c[i:].sum()) / n_tr for i in range(nb + 1)]
            thr = int(np.argmax(scores))
            tot_h += te_c_r[:thr].sum() + te_c[thr:].sum()
            tot_p += te_c.sum()
            tot_r += te_c_r.sum()
            tot_ret += np.histogram(ser[test], bins=edges)[0][:thr].sum()
            tot_n += n_te
        if tot_n == 0:
            continue
        hyb.append(tot_h / tot_n); par.append(tot_p / tot_n); ret.append(tot_r / tot_n)
        retn.append(tot_ret / tot_n)
    if not hyb:
        return None
    return dict(n=n, param_acc=np.mean(par), ret_acc=np.mean(ret), hybrid_acc=np.mean(hyb),
                hybrid_std=np.std(hyb), frac_retrieval=np.mean(retn))


def summarize(slim_dir):
    """Một dòng cho mỗi (model, dtype, shots, n): độ chính xác từng cách và adaptive."""
    groups = {}
    for path in sorted(Path(slim_dir).glob("*.csv")):
        info = parse_name(path.name)
        if info is None:
            print(f"Bỏ qua (tên file không theo quy ước): {path.name}")
            continue
        df = read_run(path)
        key = (info["model"], info["dtype"], info["shots"], len(df))
        group = groups.setdefault(key, {})
        if info["method"] in group:
            raise SystemExit(f"Hai file cùng {key} và cùng cách {info['method']}: {path.name}")
        group[info["method"]] = df

    rows = []
    for (model, dtype, shots, n), runs in sorted(groups.items()):
        row = dict(model=model, dtype=dtype, shots=shots, n=n)
        for method in METHODS:
            row[method] = runs[method]["is_correct"].mean() if method in runs else np.nan
        r = adaptive_eval(runs["vanilla"], runs["BM25"]) if {"vanilla", "BM25"} <= set(runs) else None
        row.update(adaptive=r["hybrid_acc"] if r else np.nan, adaptive_std=r["hybrid_std"] if r else np.nan,
                   frac_retrieval=r["frac_retrieval"] if r else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", "dtype", "shots", "n", *METHODS,
                                       "adaptive", "adaptive_std", "frac_retrieval"])


def main():
    ap = argparse.ArgumentParser(description="Gộp kết quả rút gọn và tính adaptive retrieval.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "summary.csv"))
    args = ap.parse_args()
    table = summarize(args.slim_dir)
    if table.empty:
        raise SystemExit("Không có file kết quả nào trong " + args.slim_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
