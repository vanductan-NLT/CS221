"""Chấm chặt hơn: điểm còn bao nhiêu khi bỏ cách chấm "chuỗi con" (issue #35, mục 1 của `docs/PHAN_TICH.md`).

Cách chấm của code gốc (cột `is_correct`) coi là đúng nếu một đáp án trong `possible_answers` xuất hiện ở BẤT KỲ chỗ nào
trong câu trả lời, dưới dạng chuỗi con: nguyên bản, chữ thường hoặc viết hoa chữ đầu. Nên "pol" (một đáp án của
"politician") khớp trong "police officer". Cách chấm chặt (`is_correct_strict`, cột MỚI, không thay `is_correct`):

- chuẩn hóa cả câu trả lời và đáp án: chữ thường, bỏ dấu câu, chỉ giữ chữ và số, tách bằng một dấu cách;
- đúng khi câu trả lời CHÍNH LÀ một đáp án, hoặc BẮT ĐẦU bằng một đáp án và đáp án đó là cả từ (đứng đầu câu trả lời).

Việc script làm (chạy trên CPU, khoảng một phút):
1. Thêm cột `is_correct_strict` vào mọi file trong `results/slim/` (tính từ cột `pred` và `possible_answers` của PopQA, ghép
   theo `id`; các cột cũ giữ nguyên từng chữ; chạy lại nhiều lần cho cùng kết quả).
2. Ghi `results/tables/strict_grading_summary.csv`: mỗi mô hình và mỗi cách (vanilla, BM25): độ chính xác theo cách cũ
   và cách chặt, số câu chỉ đúng theo cách cũ (`lenient_only`), số câu chỉ đúng theo cách chặt (`strict_only`); cùng với
   adaptive tính lại bằng cách chặt và tỉ lệ câu bị hại, được giúp (bảng 2×2 của issue #29) theo hai cách chấm.
3. Ghi `results/tables/strict_grading_by_relation.csv`: độ chính xác cũ và chặt theo từng loại quan hệ.

    python scripts/strict_grading.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267


def normalize(text):
    """Chữ thường, chỉ giữ chữ và số (kể cả chữ có dấu), tách bằng một dấu cách."""
    return " ".join(re.findall(r"\w+", str(text).lower()))


def is_strict_correct(pred, answers):
    """Câu trả lời chính là một đáp án, hoặc bắt đầu bằng một đáp án (đứng đầu, nguyên cả từ), sau khi chuẩn hóa."""
    p = normalize(pred)
    if not p:
        return False
    for a in answers:
        n = normalize(a)
        if n and (p == n or p.startswith(n + " ")):
            return True
    return False


def lenient_kind(pred, answers):
    """Vì sao một câu đúng theo cách cũ mà không đúng theo cách chặt: "inside_word" (đáp án chỉ nằm trong một chữ khác,
    như "pol" trong "police"), "not_leading" (đáp án là cả từ nhưng không đứng đầu câu trả lời), hoặc "other"."""
    text = str(pred)
    if any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text, re.I) for a in answers):
        return "not_leading"
    return "inside_word" if any(a.lower() in text.lower() for a in answers) else "other"


def add_strict_column(path, answers_by_id):
    """Thêm (hoặc tính lại) cột `is_correct_strict` vào một file rút gọn, giữ nguyên các cột cũ. Trả về số dòng đổi."""
    raw = Path(path).read_bytes()
    eol = "\r\n" if b"\r\n" in raw[:20000] else "\n"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    strict = ["True" if is_strict_correct(p, answers_by_id[int(i)]) else "False" for i, p in zip(df["id"], df["pred"])]
    changed = int((df["is_correct_strict"] != pd.Series(strict)).sum()) if "is_correct_strict" in df.columns else len(df)
    df["is_correct_strict"] = strict
    df.to_csv(path, index=False, lineterminator=eol)
    return changed


def to_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")


def two_by_two_shares(old_v, old_b):
    """Tỉ lệ câu bị hại (vanilla đúng, BM25 sai) và được giúp (vanilla sai, BM25 đúng) cho hai vector đúng/sai."""
    return float((old_v & ~old_b).mean()), float((~old_v & old_b).mean())


def summarize(slim_dir):
    """(bảng theo mô hình và cách, bảng theo loại quan hệ) từ các file rút gọn đã có cột `is_correct_strict`."""
    rows, rel_rows = [], []
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        for method, df in runs.items():
            if "is_correct_strict" not in df.columns:
                raise SystemExit(f"Thiếu cột is_correct_strict ở {model} {method}; chạy scripts/strict_grading.py")
            old, strict = df["is_correct"].astype(bool), to_bool(df["is_correct_strict"])
            rows.append(dict(model=model, dtype=dtype, shots=shots, n=n, method=method, acc_old=old.mean(), acc_strict=strict.mean(),
                             lenient_only=int((old & ~strict).sum()), strict_only=int((~old & strict).sum())))
            for prop, g in df.assign(strict=strict.values).groupby("prop"):
                rel_rows.append(dict(model=model, dtype=dtype, n_total=n, method=method, prop=prop, n=len(g),
                                     acc_old=g["is_correct"].mean(), acc_strict=g["strict"].mean()))
    summary = pd.DataFrame(rows)
    # Các cột chỉ có ở lượt đủ câu có cả vanilla và BM25: adaptive và bảng 2×2 theo hai cách chấm.
    extra = {}
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n != FULL_N or not {"vanilla", "BM25"} <= set(runs):
            continue
        v, b = runs["vanilla"], runs["BM25"]
        v_s = v.assign(is_correct=to_bool(v["is_correct_strict"]))
        b_s = b.assign(is_correct=to_bool(b["is_correct_strict"]))
        a_old, a_strict = aggregate.adaptive_eval(v, b), aggregate.adaptive_eval(v_s, b_s)
        m = v[["id", "is_correct", "is_correct_strict"]].merge(b[["id", "is_correct", "is_correct_strict"]], on="id", suffixes=("_v", "_b"))
        hurt_old, help_old = two_by_two_shares(m["is_correct_v"].astype(bool), m["is_correct_b"].astype(bool))
        hurt_s, help_s = two_by_two_shares(to_bool(m["is_correct_strict_v"]), to_bool(m["is_correct_strict_b"]))
        extra[(model, dtype, shots, n)] = dict(adaptive_old=a_old["hybrid_acc"], adaptive_strict=a_strict["hybrid_acc"],
                                               frac_retrieval_old=a_old["frac_retrieval"], frac_retrieval_strict=a_strict["frac_retrieval"],
                                               hurt_old=hurt_old, hurt_strict=hurt_s, help_old=help_old, help_strict=help_s)
    for col in ["adaptive_old", "adaptive_strict", "frac_retrieval_old", "frac_retrieval_strict", "hurt_old", "hurt_strict", "help_old", "help_strict"]:
        summary[col] = [extra.get((r.model, r.dtype, r.shots, r.n), {}).get(col, np.nan) if r.method == "vanilla" else np.nan
                        for r in summary.itertuples()]
    return summary, pd.DataFrame(rel_rows)


def main():
    ap = argparse.ArgumentParser(description="Chấm chặt hơn: thêm is_correct_strict và so điểm cũ với điểm mới.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--popqa", default=str(ROOT / "upstream" / "data" / "popQA.tsv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    args = ap.parse_args()
    popqa = pd.read_csv(args.popqa, sep="\t")
    answers_by_id = {int(r.id): json.loads(r.possible_answers) for r in popqa.itertuples(index=False)}
    files = sorted(Path(args.slim_dir).glob("*.csv"))
    changed = {f.name: add_strict_column(f, answers_by_id) for f in files if aggregate.parse_name(f.name)}
    print(f"Đã thêm hoặc cập nhật cột is_correct_strict ở {len(changed)} file; số dòng đổi: {sum(changed.values()):,}")
    summary, by_rel = summarize(args.slim_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "strict_grading_summary.csv", index=False)
    by_rel.to_csv(out / "strict_grading_by_relation.csv", index=False)
    full = summary[summary.n == FULL_N]
    print(full[["model", "method", "acc_old", "acc_strict", "lenient_only", "strict_only"]].round(4).to_string(index=False))
    print(f"\nĐã ghi 2 file vào {out}")


if __name__ == "__main__":
    main()
