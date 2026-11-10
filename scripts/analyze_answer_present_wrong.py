"""Phân tích những câu đoạn BM25 top-1 có đáp án mà BM25 vẫn trả lời sai (issue #38, mục 4 của `docs/PHAN_TICH.md`).

"Có đáp án" tính bằng `answer_in_passage` của `results/tables/bm25_top1_has_answer.csv` (quy tắc chấm của đồ án áp lên
đoạn văn, xem `docs/BANG_TAI_LAP.md`), không dùng cờ `has_answer` có sẵn vì cờ đó đánh giá thiếu. Mọi mô hình dùng
chung một đoạn cho mỗi câu, nên số câu "có đáp án" giống nhau ở mọi mô hình (3.644 trên 14.267).

Ghi ba bảng trong `results/tables/`:

1. `answer_present_wrong_by_model.csv`: mỗi mô hình, trong các câu có đáp án trong đoạn: số câu BM25 sai (`n_wrong`,
   `wrong_pct`), số câu trong đó vanilla từng đúng, và chia câu sai thành 4 nhóm bằng quy tắc tự động, xét theo thứ tự:
   - `substring_only`: đáp án chỉ khớp như một phần của chữ khác ("pol" trong "Polo"), không phải cả từ: có đáp án giả;
   - `near_miss`: câu trả lời của BM25 là một phần của một đáp án chuẩn hoặc ngược lại, theo tập từ đã chuẩn hóa
     ("comedy" với "comedy film"; "Isabelo Florentino de los Reyes" với "Isabelo de los Reyes"): đúng hoặc gần đúng về
     nghĩa nhưng không khớp chuỗi chấm;
   - `other_candidate`: câu trả lời của BM25 nằm trong đoạn văn nhưng không phải đáp án: mô hình chọn nhầm một ứng viên khác;
   - `ignored_passage`: câu trả lời của BM25 không nằm trong đoạn văn: mô hình bỏ qua đoạn hoặc tự bịa.
   Đây là xấp xỉ tự động; nó không bắt được "có đáp án giả" khi đáp án khớp cả từ nhưng nói về thực thể khác.
2. `answer_present_wrong_by_relation.csv`: mỗi mô hình và loại quan hệ: số câu có đáp án, số câu sai, tỉ lệ sai.
3. `answer_present_wrong_sample.csv`: 20 câu GPT-J chọn ngẫu nhiên (seed 633) trong số câu có đáp án mà BM25 sai, kèm
   nhóm nguyên nhân đọc tay (`MANUAL_CAUSE`, một người đọc) và nhóm tự động để đối chiếu.

Cần file BM25 (42 MB, không đưa lên git; xem `data/README.md`) để đọc đoạn văn. Chạy trên CPU, vài chục giây:
    python scripts/analyze_answer_present_wrong.py --bm25 data/raw/wikidata_bm25_14282_retrieved_docs.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

import aggregate
import make_bm25_top1_table as top1_script

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267
CAUSES = ["substring_only", "near_miss", "other_candidate", "ignored_passage"]

# Nhóm nguyên nhân đọc tay cho 20 câu GPT-J (id -> nhóm). Một người đọc (Claude Code), chỉ có đoạn văn và đáp án để xét.
#   fake_answer: "có đáp án" giả, đáp án khớp nhưng đoạn nói về thực thể khác hoặc nói ngược lại
#   right_in_meaning: đúng hoặc gần đúng về nghĩa nhưng không khớp chuỗi chấm (thiếu chữ, tên đầy đủ hơn, tên rút gọn)
#   narrow_gold: câu trả lời hợp lý theo đoạn nhưng đáp án chuẩn chọn một nhãn khác
#   wrong_pick: mô hình chọn nhầm một thực thể hoặc cụm khác trong chính đoạn có đáp án
#   ignored: mô hình bỏ qua đoạn dù đoạn có đáp án
MANUAL_CAUSE = {
    3381798: "fake_answer", 4582156: "fake_answer", 308097: "fake_answer", 4272302: "fake_answer", 900521: "fake_answer",
    3337761: "right_in_meaning", 144090: "right_in_meaning", 505960: "right_in_meaning", 3205878: "right_in_meaning",
    1303963: "right_in_meaning", 3521531: "right_in_meaning", 6092302: "right_in_meaning",
    2576512: "narrow_gold", 3237038: "narrow_gold", 4097458: "narrow_gold",
    2849350: "wrong_pick", 1188844: "wrong_pick", 1830760: "wrong_pick",
    4830188: "ignored", 6370830: "ignored",
}


def normalize(text):
    """Chữ thường, chỉ giữ chữ và số, tách bằng một dấu cách."""
    return " ".join(re.findall(r"\w+", str(text).lower()))


def whole_word_present(text, answers):
    """Một đáp án xuất hiện như cả từ hoặc cả cụm (không phân biệt hoa thường), không chỉ là một phần của chữ khác."""
    return any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text, re.I) for a in answers)


def is_near_miss(pred, answers):
    """Tập từ của câu trả lời nằm trong tập từ của một đáp án chuẩn, hoặc ngược lại."""
    p = set(normalize(pred).split())
    if not p:
        return False
    for a in answers:
        t = set(normalize(a).split())
        if t and (p <= t or t <= p):
            return True
    return False


def cause_of(pred, answers, text, whole_word):
    """Nhóm nguyên nhân tự động cho một câu BM25 sai dù đoạn có đáp án (xét theo thứ tự trong CAUSES)."""
    if not whole_word:
        return "substring_only"
    if is_near_miss(pred, answers):
        return "near_miss"
    if str(pred).strip() and str(pred).strip().lower() in text.lower():
        return "other_candidate"
    return "ignored_passage"


def question_info(popqa, passages, answer_in_passage):
    """{id: dict(prop, question, answers, text, has, whole_word)} cho cả PopQA."""
    info = {}
    for r in popqa.itertuples(index=False):
        text = passages[r.id]["text"]
        answers = json.loads(r.possible_answers)
        info[r.id] = dict(prop=r.prop, question=r.question, answers=answers, text=text,
                          has=bool(answer_in_passage[r.id]), whole_word=whole_word_present(text, answers))
    return info


def cohort(vanilla, bm25, info):
    """Các câu có đáp án trong đoạn, kèm kết quả vanilla và BM25 (giữ thứ tự file vanilla)."""
    m = vanilla[["id", "is_correct"]].merge(bm25[["id", "is_correct", "pred"]], on="id", suffixes=("_v", "_b"))
    return m[m["id"].map(lambda i: info[i]["has"])].reset_index(drop=True)


def analyze_model(vanilla, bm25, info):
    """(dòng theo mô hình, danh sách dòng theo loại quan hệ) cho một mô hình."""
    m = cohort(vanilla, bm25, info)
    wrong = m[~m["is_correct_b"]]
    counts = {c: 0 for c in CAUSES}
    for r in wrong.itertuples():
        d = info[r.id]
        counts[cause_of(r.pred, d["answers"], d["text"], d["whole_word"])] += 1
    row = dict(n_with_answer=len(m), n_wrong=len(wrong), wrong_pct=len(wrong) / len(m),
               vanilla_right_among_wrong=int(wrong["is_correct_v"].sum()))
    for c in CAUSES:
        row[c] = counts[c]
        row[c + "_share"] = counts[c] / len(wrong) if len(wrong) else float("nan")
    props = m["id"].map(lambda i: info[i]["prop"])
    by_rel = [dict(prop=p, n_with_answer=len(g), n_wrong=int((~g["is_correct_b"]).sum()),
                   wrong_pct=float((~g["is_correct_b"]).mean())) for p, g in m.groupby(props)]
    return row, by_rel


def sample_cases(vanilla, bm25, info, n=20, seed=633):
    """20 câu GPT-J ngẫu nhiên trong số câu có đáp án mà BM25 sai."""
    m = cohort(vanilla, bm25, info)
    wrong = m[~m["is_correct_b"]].sample(n=n, random_state=seed)
    rows = []
    for r in wrong.itertuples():
        d = info[r.id]
        rows.append(dict(id=r.id, prop=d["prop"], question=d["question"], gold_first=d["answers"][0], bm25_pred=str(r.pred).strip(),
                         vanilla_correct=bool(r.is_correct_v), whole_word=d["whole_word"],
                         auto_cause=cause_of(r.pred, d["answers"], d["text"], d["whole_word"]),
                         manual_cause=MANUAL_CAUSE.get(r.id, "")))
    return pd.DataFrame(rows)


def build_tables(slim_dir, popqa, passages, top1):
    answer_in_passage = top1.set_index("id")["answer_in_passage"]
    info = question_info(popqa, passages, answer_in_passage)
    model_rows, rel_rows, sample = [], [], None
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n != FULL_N or not {"vanilla", "BM25"} <= set(runs):
            continue
        row, by_rel = analyze_model(runs["vanilla"], runs["BM25"], info)
        model_rows.append(dict(model=model, dtype=dtype, **row))
        rel_rows += [dict(model=model, dtype=dtype, **r) for r in by_rel]
        if "gpt-j" in model:
            sample = sample_cases(runs["vanilla"], runs["BM25"], info)
    return pd.DataFrame(model_rows), pd.DataFrame(rel_rows), sample


def main():
    ap = argparse.ArgumentParser(description="Đoạn BM25 có đáp án mà BM25 vẫn sai.")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--popqa", default=str(ROOT / "upstream" / "data" / "popQA.tsv"))
    ap.add_argument("--bm25", default=str(ROOT / "data" / "raw" / "wikidata_bm25_14282_retrieved_docs.jsonl"))
    ap.add_argument("--top1", default=str(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    args = ap.parse_args()
    popqa = pd.read_csv(args.popqa, sep="\t")
    top1 = pd.read_csv(args.top1)
    top1["answer_in_passage"] = top1["answer_in_passage"].astype(str).str.strip().str.lower().eq("true")
    by_model, by_rel, sample = build_tables(args.slim_dir, popqa, top1_script.top1_passages(popqa, args.bm25), top1)
    if by_model.empty:
        raise SystemExit("Không có mô hình nào đủ vanilla và BM25 (14.267 câu) trong " + args.slim_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_model.to_csv(out / "answer_present_wrong_by_model.csv", index=False)
    by_rel.to_csv(out / "answer_present_wrong_by_relation.csv", index=False)
    sample.to_csv(out / "answer_present_wrong_sample.csv", index=False)
    print(by_model.round(3).to_string(index=False))
    print(f"\nĐã ghi 3 file vào {out}")


if __name__ == "__main__":
    main()
