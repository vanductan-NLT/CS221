"""Phân tích những lần retrieval gây hại: vanilla đúng mà BM25 sai (issue #37, mục 3 của `docs/PHAN_TICH.md`).

Với mỗi mô hình có đủ vanilla và BM25 (14.267 câu), ghi các bảng trong `results/tables/`:

1. `retrieval_harm_by_popularity.csv`: chia mọi câu thành 5 nhóm bằng số câu theo log(`s_pop`) (nhóm 1 = ít nổi
   tiếng nhất, giống bảng nhóm điểm ở `docs/BANG_MO_RONG.md`). Mỗi nhóm: số câu, số câu vanilla đúng, số câu bị hại,
   `hurt_pct` (bị hại trên mọi câu của nhóm), `lost_share` (bị hại trên số câu vanilla đang đúng), và
   `hurt_share_of_all` (nhóm này chứa bao nhiêu phần của mọi câu bị hại; đều thì mỗi nhóm 20%).
2. `retrieval_harm_cells.csv`: với 4 ô hại / giúp / đúng cả hai / sai cả hai: số câu, `recall_at_1` (đáp án có trong
   đoạn top-1, lấy từ `bm25_top1_has_answer.csv`) và `pred_in_passage` (câu trả lời của BM25 xuất hiện nguyên văn
   trong đoạn văn, không phân biệt hoa thường; là cận dưới của "mô hình chép từ đoạn văn").
3. `retrieval_harm_questions.csv`: các câu bị hại ở ít nhất `--min-models` mô hình (mặc định 6) và số mô hình bị hại,
   để chọn ví dụ đọc tay (id, loại quan hệ, `s_pop`, câu hỏi).
4. `retrieval_harm_hub.csv` và `retrieval_harm_hub_by_relation.csv`: đoạn "dùng chung". Một đoạn Wikipedia về một thực
   thể có thể hợp với nhiều loại quan hệ của chính thực thể đó, nhưng không thể hợp với câu hỏi về nhiều chủ thể khác
   nhau; đoạn nào được chọn làm top-1 cho câu hỏi về ít nhất 5 chủ thể (`subj`) khác nhau thì gọi là đoạn dùng chung.
   Bảng đầu so các câu có đoạn dùng chung với các câu còn lại (độ chính xác vanilla, BM25, recall@1, số câu bị hại);
   bảng sau cho biết mỗi loại quan hệ có bao nhiêu câu rơi vào đoạn dùng chung.

Cần file BM25 (42 MB, không đưa lên git; xem `data/README.md`) để đọc đoạn văn. Chạy trên CPU, vài chục giây:
    python scripts/analyze_retrieval_harm.py --bm25 data/raw/wikidata_bm25_14282_retrieved_docs.jsonl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate
import make_bm25_top1_table as top1_script

ROOT = Path(__file__).resolve().parents[1]
FULL_N = 14267
CELLS = ["hurt", "help", "both_right", "both_wrong"]
HUB_MIN_SUBJECTS = 5


def popularity_quintile(s_pop, k=5):
    """Nhóm 1..k theo log(s_pop) tăng dần, chia bằng số câu (Series cùng chỉ số với `s_pop`)."""
    ranks = np.log(s_pop.astype(float)).rank(method="first")
    return pd.qcut(ranks, k, labels=False) + 1


def cell_of(vanilla_ok, bm25_ok):
    if vanilla_ok:
        return "both_right" if bm25_ok else "hurt"
    return "help" if bm25_ok else "both_wrong"


def hub_question_ids(popqa, passages, min_subjects=HUB_MIN_SUBJECTS):
    """(tập id câu có đoạn top-1 dùng chung, số đoạn dùng chung). Đoạn dùng chung = được chọn làm top-1 cho câu hỏi
    về ít nhất `min_subjects` chủ thể (`subj`) khác nhau."""
    passage_id = popqa["id"].map(lambda i: passages[i]["id"])
    subjects = popqa.groupby(passage_id)["subj"].nunique()
    hub_passages = set(subjects[subjects >= min_subjects].index)
    return set(popqa.loc[passage_id.isin(hub_passages), "id"]), len(hub_passages)


def hub_by_relation(popqa, hub_ids):
    """Mỗi loại quan hệ: số câu, số câu có đoạn dùng chung, tỉ lệ (xếp giảm dần)."""
    t = popqa.assign(hub=popqa["id"].isin(hub_ids)).groupby("prop")["hub"].agg(n="size", n_hub="sum").reset_index()
    t["hub_share"] = t["n_hub"] / t["n"]
    return t.sort_values("hub_share", ascending=False).reset_index(drop=True)


def hub_summary(vanilla, bm25, hub_ids, answer_in_passage):
    """Một mô hình: câu có đoạn dùng chung (hub) so với các câu còn lại (non_hub)."""
    m = vanilla[["id", "is_correct"]].merge(bm25[["id", "is_correct"]], on="id", suffixes=("_v", "_b"))
    m["hub"] = m["id"].isin(hub_ids)
    m["has_answer"] = m["id"].map(answer_in_passage)
    total_hurt = int((m["is_correct_v"] & ~m["is_correct_b"]).sum())
    rows = []
    for is_hub, g in m.groupby("hub"):
        hurt = int((g["is_correct_v"] & ~g["is_correct_b"]).sum())
        rows.append(dict(group="hub" if is_hub else "non_hub", n=len(g), vanilla_acc=g["is_correct_v"].mean(),
                         bm25_acc=g["is_correct_b"].mean(), recall_at_1=g["has_answer"].mean(), hurt=hurt,
                         hurt_share_of_all=hurt / total_hurt if total_hurt else float("nan")))
    return rows


def analyze_model(vanilla, bm25, quintile, passage_text, answer_in_passage):
    """(bảng theo nhóm nổi tiếng, bảng 4 ô, tập id bị hại) cho một mô hình.

    `quintile`: Series nhóm nổi tiếng theo id; `passage_text`: {id: đoạn top-1}; `answer_in_passage`: Series bool theo id."""
    m = vanilla[["id", "is_correct"]].merge(bm25[["id", "is_correct", "pred"]], on="id", suffixes=("_v", "_b"))
    m["cell"] = [cell_of(v, b) for v, b in zip(m["is_correct_v"], m["is_correct_b"])]
    m["quintile"] = m["id"].map(quintile)
    m["has_answer"] = m["id"].map(answer_in_passage)
    m["pred_in_passage"] = [bool(str(p).strip()) and str(p).strip().lower() in passage_text[i].lower()
                            for i, p in zip(m["id"], m["pred"])]
    total_hurt = int((m["cell"] == "hurt").sum())
    by_pop = []
    for q, g in m.groupby("quintile"):
        hurt, correct = int((g["cell"] == "hurt").sum()), int(g["is_correct_v"].sum())
        by_pop.append(dict(quintile=int(q), n=len(g), vanilla_correct=correct, hurt=hurt, hurt_pct=hurt / len(g),
                           lost_share=hurt / correct if correct else float("nan"),
                           hurt_share_of_all=hurt / total_hurt if total_hurt else float("nan")))
    cells = []
    for name in CELLS:
        g = m[m["cell"] == name]
        cells.append(dict(cell=name, n=len(g), recall_at_1=g["has_answer"].mean() if len(g) else float("nan"),
                          pred_in_passage=g["pred_in_passage"].mean() if len(g) else float("nan")))
    return by_pop, cells, set(m.loc[m["cell"] == "hurt", "id"])


def build_tables(slim_dir, popqa, passages, top1, min_models=6):
    """(theo nhóm nổi tiếng, 4 ô, câu bị hại nhiều mô hình, đoạn dùng chung theo mô hình, theo loại quan hệ, số đoạn dùng chung)."""
    quintile = popularity_quintile(popqa.set_index("id")["s_pop"])
    passage_text = {i: p["text"] for i, p in passages.items()}
    answer_in_passage = top1.set_index("id")["answer_in_passage"]
    hub_ids, n_hub_passages = hub_question_ids(popqa, passages)
    by_pop_rows, cell_rows, hub_rows, hurt_counts = [], [], [], {}
    for (model, dtype, shots, n), runs in sorted(aggregate.load_groups(slim_dir).items()):
        if n != FULL_N or not {"vanilla", "BM25"} <= set(runs):
            continue
        by_pop, cells, hurt_ids = analyze_model(runs["vanilla"], runs["BM25"], quintile, passage_text, answer_in_passage)
        by_pop_rows += [dict(model=model, dtype=dtype, **r) for r in by_pop]
        cell_rows += [dict(model=model, dtype=dtype, **r) for r in cells]
        hub_rows += [dict(model=model, dtype=dtype, **r) for r in hub_summary(runs["vanilla"], runs["BM25"], hub_ids, answer_in_passage)]
        for i in hurt_ids:
            hurt_counts[i] = hurt_counts.get(i, 0) + 1
    info = popqa.set_index("id")
    questions = pd.DataFrame(
        [dict(id=i, prop=info.loc[i, "prop"], s_pop=int(info.loc[i, "s_pop"]), question=info.loc[i, "question"], n_models_hurt=c)
         for i, c in hurt_counts.items() if c >= min_models]
    ).sort_values(["n_models_hurt", "id"], ascending=[False, True]).reset_index(drop=True)
    return (pd.DataFrame(by_pop_rows), pd.DataFrame(cell_rows), questions, pd.DataFrame(hub_rows),
            hub_by_relation(popqa, hub_ids), n_hub_passages)


def main():
    ap = argparse.ArgumentParser(description="Những lần retrieval gây hại (vanilla đúng, BM25 sai).")
    ap.add_argument("--slim-dir", default=str(ROOT / "results" / "slim"))
    ap.add_argument("--popqa", default=str(ROOT / "upstream" / "data" / "popQA.tsv"))
    ap.add_argument("--bm25", default=str(ROOT / "data" / "raw" / "wikidata_bm25_14282_retrieved_docs.jsonl"))
    ap.add_argument("--top1", default=str(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--min-models", type=int, default=6)
    args = ap.parse_args()
    popqa = pd.read_csv(args.popqa, sep="\t")
    top1 = pd.read_csv(args.top1)
    top1["answer_in_passage"] = top1["answer_in_passage"].astype(str).str.strip().str.lower().eq("true")
    passages = top1_script.top1_passages(popqa, args.bm25)
    by_pop, cells, questions, hub, hub_rel, n_hub_passages = build_tables(args.slim_dir, popqa, passages, top1, args.min_models)
    if by_pop.empty:
        raise SystemExit("Không có mô hình nào đủ vanilla và BM25 (14.267 câu) trong " + args.slim_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_pop.to_csv(out / "retrieval_harm_by_popularity.csv", index=False)
    cells.to_csv(out / "retrieval_harm_cells.csv", index=False)
    questions.to_csv(out / "retrieval_harm_questions.csv", index=False)
    hub.to_csv(out / "retrieval_harm_hub.csv", index=False)
    hub_rel.to_csv(out / "retrieval_harm_hub_by_relation.csv", index=False)
    print(hub.round(4).head(4).to_string(index=False))
    print(f"\n{by_pop.model.nunique()} mô hình; {len(questions)} câu bị hại ở ít nhất {args.min_models} mô hình; "
          f"{n_hub_passages} đoạn dùng chung phủ {int(hub_rel.n_hub.sum())} câu. Đã ghi 5 file vào {out}")


if __name__ == "__main__":
    main()
