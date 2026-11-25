"""So sánh kết quả tái lập với paper gốc (ACL 2023) và kiểm định giả thiết về Qwen2.5.

Chạy hoàn toàn trên CPU từ dữ liệu đã commit (`results/slim/`, `results/tables/`, `docs/paper_numbers.csv`):
    python scripts/compare_paper_and_qwen.py

Đầu ra:
- results/tables/paper_vs_repro.csv          : đối chiếu từng (mô hình, cách chạy) giữa paper gốc và nhóm
- results/tables/qwen_vs_paper_mcnemar.csv   : kiểm định McNemar ghép cặp (lenient + strict) cho Qwen2.5 vs paper
- results/tables/qwen_mechanisms_by_model.csv: phân rã cơ chế theo mô hình (phổ biến, đọc hiểu BM25, calibration AUROC)
- results/tables/four_conclusions_check.csv  : kiểm chứng định lượng 4 kết luận của paper trên cả 9 mô hình
- results/figs/paper_vs_repro.png            : biểu đồ so sánh số liệu Paper vs Nhóm
- results/figs/acc_vs_pop.png                : biểu đồ độ chính xác theo độ nổi tiếng (Vanilla vs BM25)
- results/figs/qwen_vs_paper_mechanisms.png  : biểu đồ 3 cơ chế lý giải vì sao Qwen2.5 vượt trội
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate  # noqa: E402

MODEL_SPECS = [
    ("EleutherAI_gpt-neo-1.3B", "fp32", "GPT-Neo 1.3B", "GPT-Neo", 1.3),
    ("facebook_opt-1.3b", "fp32", "OPT 1.3B", "OPT", 1.3),
    ("EleutherAI_gpt-neo-2.7B", "fp32", "GPT-Neo 2.7B", "GPT-Neo", 2.7),
    ("facebook_opt-2.7b", "fp32", "OPT 2.7B", "OPT", 2.7),
    ("EleutherAI_gpt-j-6b", "fp16", "GPT-J 6B", "GPT-Neo", 6.0),
    ("facebook_opt-6.7b", "fp16", "OPT 6.7B", "OPT", 6.7),
    ("Qwen_Qwen2.5-0.5B", "fp32", "Qwen2.5 0.5B", "Qwen2.5", 0.5),
    ("Qwen_Qwen2.5-1.5B", "fp32", "Qwen2.5 1.5B", "Qwen2.5", 1.5),
    ("Qwen_Qwen2.5-3B", "fp32", "Qwen2.5 3B", "Qwen2.5", 3.0),
]

MCNEMAR_PAIRS = [
    ("Qwen2.5 0.5B", "GPT-Neo 1.3B"),
    ("Qwen2.5 0.5B", "OPT 1.3B"),
    ("Qwen2.5 1.5B", "GPT-Neo 1.3B"),
    ("Qwen2.5 1.5B", "OPT 1.3B"),
    ("Qwen2.5 1.5B", "GPT-Neo 2.7B"),
    ("Qwen2.5 1.5B", "OPT 2.7B"),
    ("Qwen2.5 1.5B", "OPT 6.7B"),
    ("Qwen2.5 1.5B", "GPT-J 6B"),
    ("Qwen2.5 3B", "GPT-Neo 2.7B"),
    ("Qwen2.5 3B", "OPT 2.7B"),
    ("Qwen2.5 3B", "OPT 6.7B"),
    ("Qwen2.5 3B", "GPT-J 6B"),
]


def auroc_fast(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Tính AUROC chính xác bằng tổng hạng (tương đương Mann-Whitney U, không cần sklearn)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Kiểm định McNemar chính xác 2 phía trên hai mảng boolean ghép cặp cùng độ dài."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if len(a) != len(b):
        raise ValueError("Hai mảng phải có cùng số phần tử.")
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n_disc = n10 + n01
    p_val = 1.0 if n_disc == 0 else float(stats.binomtest(min(n10, n01), n_disc, 0.5, alternative="two-sided").pvalue)
    diff = float(np.mean(a) - np.mean(b))
    d = a.astype(float) - b.astype(float)
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return dict(
        acc_a=float(np.mean(a)),
        acc_b=float(np.mean(b)),
        diff=diff,
        ci95_lo=diff - 1.96 * se,
        ci95_hi=diff + 1.96 * se,
        n_a_only=n10,
        n_b_only=n01,
        p_value=p_val,
    )


def paired_adaptive_splits(df_p: pd.DataFrame, df_r: pd.DataFrame, n_boot=100, seed=633) -> dict:
    """Đánh giá Adaptive vs BM25 trên đúng cùng 100 lần chia 75/25 để kiểm định Wilcoxon ghép cặp."""
    m = (
        df_p[["id", "prop", "s_pop", "is_correct"]]
        .merge(df_r[["id", "is_correct"]], on="id", suffixes=("_p", "_r"))
        .sort_values("id")
        .reset_index(drop=True)
    )
    n_all = len(m)
    props_u = [p for p in m["prop"].unique() if (m["prop"] == p).sum() >= 8]
    ser_all = np.log(m["s_pop"].values.astype(float))
    c_p_all = m["is_correct_p"].values.astype(bool)
    c_r_all = m["is_correct_r"].values.astype(bool)
    prop_idx = {p: np.where(m["prop"].values == p)[0] for p in props_u}
    rng = np.random.RandomState(seed)

    diffs, retn = [], []
    for _ in range(n_boot):
        mask = ["train"] * round(n_all * 0.75) + ["test"] * round(n_all * 0.25)
        rng.shuffle(mask)
        mask = np.array(mask)
        tot_h = tot_r = tot_ret = tot_n = 0.0
        for p in props_u:
            ix = prop_idx[p]
            ser, c, c_r, test = ser_all[ix], c_p_all[ix], c_r_all[ix], mask[ix] == "test"
            _, edges = np.histogram(ser)
            train = ~test
            tr_c = np.histogram(ser[train & c], bins=edges)[0]
            tr_cr = np.histogram(ser[train & c_r], bins=edges)[0]
            te_c = np.histogram(ser[test & c], bins=edges)[0]
            te_cr = np.histogram(ser[test & c_r], bins=edges)[0]
            nb = len(edges) - 1
            thr = int(np.argmax([(tr_cr[:i].sum() + tr_c[i:].sum()) / train.sum() for i in range(nb + 1)]))
            tot_h += te_cr[:thr].sum() + te_c[thr:].sum()
            tot_r += te_cr.sum()
            tot_ret += np.histogram(ser[test], bins=edges)[0][:thr].sum()
            tot_n += test.sum()
        diffs.append((tot_h - tot_r) / tot_n)
        retn.append(tot_ret / tot_n)
    diffs = np.array(diffs)
    wins = int(np.sum(diffs > 0))
    ties = int(np.sum(diffs == 0))
    p_wilcoxon = float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return dict(
        mean_gain_vs_bm25=float(diffs.mean()),
        std_gain_vs_bm25=float(diffs.std()),
        wins_over_bm25=wins,
        ties_with_bm25=ties,
        wilcoxon_p=p_wilcoxon,
        frac_retrieval=float(np.mean(retn)),
    )


def load_runs(slim_dir: Path):
    groups = aggregate.load_groups(slim_dir)
    data = {}
    for model, dtype, label, family, params_b in MODEL_SPECS:
        g = groups[(model, dtype, 15, 14267)]
        van = g["vanilla"].sort_values("id").reset_index(drop=True)
        bm = g["BM25"].sort_values("id").reset_index(drop=True)
        assert (van["id"].values == bm["id"].values).all()
        for df in (van, bm):
            df["is_correct_strict"] = df["is_correct_strict"].map(lambda v: str(v).strip().lower() == "true")
        van["conf_num"] = pd.to_numeric(van["conf"], errors="coerce")
        data[label] = dict(
            model=model, dtype=dtype, label=label, family=family, params_b=params_b,
            vanilla=van, BM25=bm, genread=g.get("genread"),
        )
    return data


def build_paper_vs_repro(paper_csv: Path, summary_csv: Path) -> pd.DataFrame:
    paper = pd.read_csv(paper_csv)
    summ = pd.read_csv(summary_csv)
    summ = summ[summ["n"] == 14267].copy()

    label_map = {m: lbl for m, _, lbl, _, _ in MODEL_SPECS}
    dtype_map = {m: dt for m, dt, _, _, _ in MODEL_SPECS}

    rows = []
    for _, r in paper.iterrows():
        m = r["model"]
        method = r["method"]
        p_val = float(r["accuracy_xap_xi"])
        s_row = summ[summ["model"] == m]
        if s_row.empty:
            continue
        s_row = s_row.iloc[0]
        if method == "adaptive_frac_retrieval":
            g_val = float(s_row["frac_retrieval"])
        elif method in ("vanilla", "BM25", "genread", "adaptive"):
            g_val = float(s_row[method]) if pd.notna(s_row[method]) else np.nan
        else:
            continue
        if np.isnan(g_val):
            continue
        rows.append(dict(
            model=m,
            label=label_map.get(m, m),
            dtype=dtype_map.get(m, ""),
            method=method,
            metric=r["metric"],
            paper_source=r["figure"],
            paper_val=p_val,
            repro_val=g_val,
            diff_pp=(g_val - p_val) * 100.0,
            abs_diff_pp=abs(g_val - p_val) * 100.0,
        ))
    return pd.DataFrame(rows)


def build_qwen_tables(data: dict, top1_csv: Path):
    top1 = pd.read_csv(top1_csv).sort_values("id").reset_index(drop=True)
    top1["answer_in_passage"] = top1["answer_in_passage"].astype(bool)
    ref = data["Qwen2.5 3B"]["vanilla"]
    assert (ref["id"].values == top1["id"].values).all()
    s_pop = ref["s_pop"].values.astype(float)
    log10_pop = np.log10(np.maximum(s_pop, 1.0))
    has_ans = top1["answer_in_passage"].values

    order_pop = np.argsort(s_pop, kind="mergesort")
    bottom_4k = np.zeros(len(s_pop), dtype=bool)
    bottom_4k[order_pop[:4000]] = True
    quintiles = pd.qcut(np.log(s_pop), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    m_q1 = np.asarray(quintiles == "Q1")
    m_q5 = np.asarray(quintiles == "Q5")

    # 1. Paired McNemar table (Vanilla, BM25, and Conditional Reading on 3,644 in-passage)
    mc_rows = []
    for method in ("vanilla", "BM25", "BM25_in_passage"):
        for a_lbl, b_lbl in MCNEMAR_PAIRS:
            if method == "BM25_in_passage":
                a_len = data[a_lbl]["BM25"]["is_correct"].values[has_ans]
                b_len = data[b_lbl]["BM25"]["is_correct"].values[has_ans]
                a_str = data[a_lbl]["BM25"]["is_correct_strict"].values[has_ans]
                b_str = data[b_lbl]["BM25"]["is_correct_strict"].values[has_ans]
                n_eval = int(has_ans.sum())
            else:
                a_len = data[a_lbl][method]["is_correct"].values
                b_len = data[b_lbl][method]["is_correct"].values
                a_str = data[a_lbl][method]["is_correct_strict"].values
                b_str = data[b_lbl][method]["is_correct_strict"].values
                n_eval = len(a_len)
            r_len = mcnemar_exact(a_len, b_len)
            r_str = mcnemar_exact(a_str, b_str)
            mc_rows.append(dict(
                setting=method,
                n=n_eval,
                model_a=a_lbl,
                model_b=b_lbl,
                acc_a=r_len["acc_a"],
                acc_b=r_len["acc_b"],
                diff_pp=r_len["diff"] * 100.0,
                ci95_lo_pp=r_len["ci95_lo"] * 100.0,
                ci95_hi_pp=r_len["ci95_hi"] * 100.0,
                n_a_only=r_len["n_a_only"],
                n_b_only=r_len["n_b_only"],
                mcnemar_p=r_len["p_value"],
                strict_diff_pp=r_str["diff"] * 100.0,
                strict_mcnemar_p=r_str["p_value"],
            ))
    df_mc = pd.DataFrame(mc_rows)

    # 2. Per-model mechanism table & 4 conclusions check
    mech_rows = []
    concl_rows = []
    for _, _, label, family, params_b in MODEL_SPECS:
        d = data[label]
        cv = d["vanilla"]["is_correct"].values
        cb = d["BM25"]["is_correct"].values
        cbs = d["BM25"]["is_correct_strict"].values
        conf = d["vanilla"]["conf_num"].values
        conf_filled = np.where(np.isfinite(conf), conf, np.nanmin(conf) - 1.0)

        r_pop, p_pop = stats.pearsonr(log10_pop, cv.astype(float))
        q_van = [float(cv[np.asarray(quintiles == q)].mean()) for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]]
        q_bm = [float(cb[np.asarray(quintiles == q)].mean()) for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]]

        kl3_q1 = mcnemar_exact(cb[m_q1], cv[m_q1])
        kl3_q5 = mcnemar_exact(cb[m_q5], cv[m_q5])
        kl4 = paired_adaptive_splits(d["vanilla"], d["BM25"])

        read_acc = float(cb[has_ans].mean())
        read_acc_strict = float(cbs[has_ans].mean())
        retain_noans = float(cb[cv & ~has_ans].mean())
        retain_inpas = float(cb[cv & has_ans].mean())
        auroc_c = auroc_fast(cv, conf_filled)
        auroc_p = auroc_fast(cv, log10_pop)

        mech_rows.append(dict(
            label=label, family=family, params_b=params_b, dtype=d["dtype"],
            vanilla_acc=float(cv.mean()),
            bm25_acc=float(cb.mean()),
            bottom_4k_vanilla=float(cv[bottom_4k].mean()),
            q1_vanilla=q_van[0], q2_vanilla=q_van[1], q3_vanilla=q_van[2], q4_vanilla=q_van[3], q5_vanilla=q_van[4],
            q1_bm25=q_bm[0], q5_bm25=q_bm[4],
            pearson_r_pop=float(r_pop), pearson_p_pop=float(p_pop),
            reading_acc_in_passage=read_acc,
            reading_acc_strict=read_acc_strict,
            retain_when_no_ans=retain_noans,
            retain_when_in_passage=retain_inpas,
            auroc_conf=auroc_c,
            auroc_pop=auroc_p,
        ))
        concl_rows.append(dict(
            label=label, family=family, params_b=params_b,
            kl1_pearson_r=float(r_pop),
            kl1_p=float(p_pop),
            kl1_q5_minus_q1_pp=(q_van[4] - q_van[0]) * 100.0,
            kl3_q1_bm25_minus_van_pp=kl3_q1["diff"] * 100.0,
            kl3_q1_p=kl3_q1["p_value"],
            kl3_q5_bm25_minus_van_pp=kl3_q5["diff"] * 100.0,
            kl3_q5_p=kl3_q5["p_value"],
            kl4_adaptive_gain_vs_bm25_pp=kl4["mean_gain_vs_bm25"] * 100.0,
            kl4_wins_over_100=kl4["wins_over_bm25"],
            kl4_wilcoxon_p=kl4["wilcoxon_p"],
            kl4_frac_retrieval=kl4["frac_retrieval"],
        ))
    return df_mc, pd.DataFrame(mech_rows), pd.DataFrame(concl_rows)


def plot_paper_vs_repro(df_pvr: pd.DataFrame, out_path: Path):
    acc = df_pvr[df_pvr["metric"] == "accuracy"].copy()
    models_order = ["GPT-Neo 1.3B", "OPT 1.3B", "GPT-Neo 2.7B", "OPT 2.7B", "GPT-J 6B", "OPT 6.7B"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    for ax, method, title in zip(
        axes,
        ["vanilla", "BM25"],
        ["Vanilla (không tra cứu)", "BM25 (tra cứu top-1 Wikipedia)"],
    ):
        sub = acc[acc["method"] == method].set_index("label").reindex(models_order)
        x = np.arange(len(models_order))
        w = 0.36
        p_vals = sub["paper_val"].values * 100
        g_vals = sub["repro_val"].values * 100
        b1 = ax.bar(x - w / 2, p_vals, w, label="Paper gốc (ACL 2023)", color="#4C78A8", edgecolor="white")
        b2 = ax.bar(x + w / 2, g_vals, w, label="Nhóm tái lập (CS221)", color="#F58518", edgecolor="white")
        for i, (pv, gv) in enumerate(zip(p_vals, g_vals)):
            diff = gv - pv
            ax.text(x[i] - w / 2, pv + 0.4, f"{pv:.1f}%", ha="center", va="bottom", fontsize=8, color="#2B4C7E")
            ax.text(x[i] + w / 2, gv + 0.4, f"{gv:.1f}%", ha="center", va="bottom", fontsize=8, color="#A65300")
            ax.text(x[i], max(pv, gv) + 2.1, f"Δ={diff:+.2f}pp", ha="center", va="bottom", fontsize=7.8,
                    fontweight="bold", color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 31)
    axes[0].set_ylabel("Độ chính xác trên PopQA (%)", fontsize=10)
    axes[0].legend(loc="upper left", frameon=True)
    mad = acc["abs_diff_pp"].mean()
    fig.suptitle(
        f"Đối chiếu độ chính xác: Paper gốc (ACL 2023) vs Nhóm tái lập (n = 14.267, MAD = {mad:.2f} điểm %)",
        fontsize=12.5, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_acc_vs_pop(data: dict, out_path: Path):
    """Vẽ độ chính xác theo độ nổi tiếng tương đối (chuẩn hóa trong từng loại quan hệ, như Hình 8 của paper)."""
    ref = data["Qwen2.5 3B"]["vanilla"].copy()
    log_pop = np.log(np.maximum(ref["s_pop"].values.astype(float), 1.0))
    ref["log_pop"] = log_pop
    rel_pop = ref.groupby("prop")["log_pop"].transform(lambda s: (s - s.mean()) / s.std(ddof=0)).values

    bins = np.quantile(rel_pop, np.linspace(0, 1, 9))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    bin_ids = np.clip(np.digitize(rel_pop, bins) - 1, 0, len(bins) - 2)
    x_centers = np.array([rel_pop[bin_ids == i].mean() for i in range(len(bins) - 1)])

    panels = [
        ("GPT-Neo 1.3B", "GPT-Neo 1.3B (Paper P0)"),
        ("OPT 1.3B", "OPT 1.3B (Paper P0)"),
        ("GPT-J 6B", "GPT-J 6B fp16 (Paper P2)"),
        ("OPT 6.7B", "OPT 6.7B fp16 (Paper P2)"),
        ("Qwen2.5 0.5B", "Qwen2.5 0.5B (Mở rộng A)"),
        ("Qwen2.5 3B", "Qwen2.5 3B (Mở rộng A)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True, sharey=True)
    for ax, (lbl, title) in zip(axes.flat, panels):
        cv = data[lbl]["vanilla"]["is_correct"].values
        cb = data[lbl]["BM25"]["is_correct"].values
        yv = np.array([cv[bin_ids == i].mean() * 100 for i in range(len(x_centers))])
        yb = np.array([cb[bin_ids == i].mean() * 100 for i in range(len(x_centers))])
        ax.plot(x_centers, yv, "-o", color="#4C78A8", linewidth=2.2, markersize=4.5, label="Vanilla (trí nhớ tham số)")
        ax.plot(x_centers, yb, "--s", color="#E45756", linewidth=2.2, markersize=4.5, label="BM25 (truy xuất Wikipedia)")
        ax.fill_between(x_centers, yv, yb, where=(yb >= yv), color="#54A24B", alpha=0.14, interpolate=True, label="BM25 giúp")
        ax.fill_between(x_centers, yv, yb, where=(yb < yv), color="#E45756", alpha=0.18, interpolate=True, label="BM25 hại")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, 56)
    for ax in axes[:, 0]:
        ax.set_ylabel("Độ chính xác (%)", fontsize=9.5)
    fig.supxlabel("Độ nổi tiếng tương đối (z-score log s_pop chuẩn hóa theo từng loại quan hệ, như Hình 8 paper)",
                  fontsize=10, y=0.015)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.945), frameon=True, fontsize=9)
    fig.suptitle("Độ chính xác theo độ nổi tiếng tương đối trên PopQA (n = 14.267): Vanilla vs BM25",
                 fontsize=12.5, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.04, 1, 0.90])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_qwen_mechanisms(df_mech: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    colors = {"GPT-Neo": "#4C78A8", "OPT": "#72B7B2", "Qwen2.5": "#E45756"}
    markers = {"GPT-Neo": "o", "OPT": "s", "Qwen2.5": "^"}

    # Panel A: Parameter Efficiency (Vanilla & BM25 vs Params)
    ax = axes[0]
    for fam in ["GPT-Neo", "OPT", "Qwen2.5"]:
        sub = df_mech[df_mech["family"] == fam].sort_values("params_b")
        ax.plot(sub["params_b"], sub["vanilla_acc"] * 100, f"-{markers[fam]}", color=colors[fam],
                linewidth=2.2, markersize=6.5, label=f"{fam} (Vanilla)")
        ax.plot(sub["params_b"], sub["bm25_acc"] * 100, f"--{markers[fam]}", color=colors[fam],
                linewidth=1.8, markersize=5.5, alpha=0.85, label=f"{fam} (BM25)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.3, 2.7, 6.7])
    ax.set_xticklabels(["0.5B", "1.3–1.5B", "2.7–3B", "6–6.7B"], fontsize=8.5)
    ax.minorticks_off()
    ax.set_ylim(10.0, 30.5)
    ax.set_xlabel("Số tham số (thang log)")
    ax.set_ylabel("Độ chính xác PopQA (%)")
    ax.set_title("(A) Hiệu quả tham số: Qwen2.5 vượt 2–4× cỡ", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.6, ncol=2, loc="upper left", frameon=True)

    # Panel B: Reading Comprehension vs Distractor Retention
    ax = axes[1]
    order = df_mech.sort_values("reading_acc_in_passage").reset_index(drop=True)
    y = np.arange(len(order))
    bar_colors = [colors[f] for f in order["family"]]
    ax.barh(y, order["reading_acc_in_passage"] * 100, color=bar_colors, alpha=0.9, height=0.62)
    for i, r in order.iterrows():
        ax.text(r["reading_acc_in_passage"] * 100 + 0.4, i, f"{r['reading_acc_in_passage']*100:.1f}%",
                va="center", fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(order["label"], fontsize=8.5)
    ax.set_xlim(68, 92)
    ax.set_xlabel("Đọc đúng khi đoạn BM25 có đáp án (%)")
    ax.set_title("(B) Khả năng đọc hiểu ngữ cảnh (n = 3.644)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel C: Confidence Calibration AUROC
    ax = axes[2]
    order_c = df_mech.sort_values("auroc_conf").reset_index(drop=True)
    y = np.arange(len(order_c))
    bar_colors_c = [colors[f] for f in order_c["family"]]
    ax.barh(y, order_c["auroc_conf"], color=bar_colors_c, alpha=0.9, height=0.62)
    for i, r in order_c.iterrows():
        ax.text(r["auroc_conf"] + 0.003, i, f"{r['auroc_conf']:.3f}", va="center", fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(order_c["label"], fontsize=8.5)
    ax.set_xlim(0.70, 0.90)
    ax.set_xlabel("AUROC phân biệt câu đúng/sai bằng conf")
    ax.set_title("(C) Khả năng tự nhận biết (Calibration)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.suptitle("Vì sao Qwen2.5 vượt trội các mô hình trong paper? Ba cơ chế định lượng trên PopQA (n = 14.267)",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Đối chiếu paper gốc và kiểm định giả thiết Qwen2.5.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--paper-csv", default=str(ROOT / "docs" / "paper_numbers.csv"))
    ap.add_argument("--summary-csv", default=str(ROOT / "results" / "tables" / "summary.csv"))
    ap.add_argument("--top1-csv", default=str(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv"))
    ap.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--figs-dir", default=str(ROOT / "results" / "figs"))
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    figs_dir = Path(args.figs_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_pvr = build_paper_vs_repro(Path(args.paper_csv), Path(args.summary_csv))
    df_pvr.to_csv(tables_dir / "paper_vs_repro.csv", index=False)

    data = load_runs(Path(args.slim_dir))
    df_mc, df_mech, df_concl = build_qwen_tables(data, Path(args.top1_csv))
    df_mc.to_csv(tables_dir / "qwen_vs_paper_mcnemar.csv", index=False)
    df_mech.to_csv(tables_dir / "qwen_mechanisms_by_model.csv", index=False)
    df_concl.to_csv(tables_dir / "four_conclusions_check.csv", index=False)

    plot_paper_vs_repro(df_pvr, figs_dir / "paper_vs_repro.png")
    plot_acc_vs_pop(data, figs_dir / "acc_vs_pop.png")
    plot_qwen_mechanisms(df_mech, figs_dir / "qwen_vs_paper_mechanisms.png")

    acc_only = df_pvr[df_pvr["metric"] == "accuracy"]
    print(f"Paper vs Repro ({len(acc_only)} ô accuracy): MAD = {acc_only['abs_diff_pp'].mean():.2f}pp, "
          f"max |diff| = {acc_only['abs_diff_pp'].max():.2f}pp")
    print(f"Đã ghi 4 bảng CSV vào {tables_dir} và 3 biểu đồ PNG vào {figs_dir}")


if __name__ == "__main__":
    main()
