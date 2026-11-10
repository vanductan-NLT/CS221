"""Bảng "đoạn BM25 top-1 có chứa đáp án không" cho cả 14.267 câu PopQA (issue #29).

Vì sao cần: cờ `hasanswer` có sẵn trong file BM25 của tác giả chỉ báo 7,86% câu có đáp án trong đoạn top-1,
nhưng tính lại bằng đúng cách chấm của đồ án thì ra khoảng 25%. Cờ đó đánh giá thiếu rất nhiều (xem
`docs/BANG_TAI_LAP.md`), nên bảng recall@1 dùng cột `answer_in_passage` của bảng này.

`answer_in_passage` dùng đúng quy tắc chấm của code gốc (`upstream/run_model.py`, dòng 275), chỉ đổi chỗ so:
một đáp án trong `possible_answers` xuất hiện trong đoạn văn (chuỗi con), thử nguyên bản, chữ thường và
viết hoa chữ đầu. Đoạn văn là đoạn mô hình thực sự thấy: tra bằng nguyên văn câu hỏi (quyết định #12 trong
`docs/QUYET_DINH.md`), `retrieval_id` ghi lại id đoạn văn đó để đối chiếu với kết quả từng mô hình.

Ghi `results/tables/bm25_top1_has_answer.csv`: id, retrieval_id, hasanswer_flag (cờ của file BM25),
answer_in_passage. File BM25 (42 MB) không đưa lên git; lấy từ dataset Kaggle, xem `data/README.md`.

Chạy trên CPU, vài giây:
    python scripts/make_bm25_top1_table.py --bm25 data/raw/wikidata_bm25_14282_retrieved_docs.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def answer_in_text(text, answers):
    """Quy tắc chấm của code gốc áp lên một đoạn văn: một đáp án xuất hiện nguyên bản, chữ thường hoặc viết hoa chữ đầu."""
    return any(a in text or a.lower() in text or a.capitalize() in text for a in answers)


def build(popqa_path, bm25_path):
    popqa = pd.read_csv(popqa_path, sep="\t")
    retrieval = {}
    with Path(bm25_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                retrieval[record["question"]] = record
    rows = []
    for r in popqa.itertuples(index=False):
        ctxs = retrieval.get(r.question, {}).get("ctxs")
        if not ctxs or not ctxs[0].get("text", "").strip():
            raise SystemExit(f"Thiếu đoạn BM25 cho câu id={r.id}: {r.question}")
        top1 = ctxs[0]
        rows.append(dict(id=r.id, retrieval_id=top1["id"], hasanswer_flag=bool(top1["hasanswer"]),
                         answer_in_passage=answer_in_text(top1["text"], json.loads(r.possible_answers))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Đoạn BM25 top-1 có chứa đáp án không, cho cả PopQA.")
    ap.add_argument("--popqa", default=str(ROOT / "upstream" / "data" / "popQA.tsv"))
    ap.add_argument("--bm25", default=str(ROOT / "data" / "raw" / "wikidata_bm25_14282_retrieved_docs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "bm25_top1_has_answer.csv"))
    args = ap.parse_args()
    table = build(args.popqa, args.bm25)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} câu. recall@1 theo cờ của file BM25: {table.hasanswer_flag.mean():.4f}; "
          f"tính lại bằng quy tắc chấm của đồ án: {table.answer_in_passage.mean():.4f}")
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
