# HalluGuard MiniCheck full benchmark — run v2

This directory contains the completed LLM-AggreFact test-split artifacts produced by
`halluguard_minicheck_full_benchmark_run_v2.ipynb`.

## Completion summary

- Total rows: 29,320
- Successful predictions: 29,319
- Parse failures: 0
- Inference failures: 1
- Not run: 0
- Accuracy on successful rows: 0.8451175005968826
- F1 on successful rows: 0.9046869424679387
- Balanced Accuracy on successful rows: 0.7206754926758265
- Macro Balanced Accuracy on successful rows: 0.6714861385248813

The single failed row exceeded the frozen `max_model_len=4096`: its decoder prompt
contained 4,312 tokens. It has no fabricated prediction or support score. Metrics exclude
that failed row, as recorded in `manifest.json`.

## Files

- `predictions_final.csv.gz`: complete row-level result table, losslessly gzip-compressed.
- `predictions_final.csv.sha256`: SHA-256 of the uncompressed CSV.
- `predictions_checkpoint.csv`: compact resumable checkpoint.
- `dataset_metrics.csv`: per-dataset metrics for all 11 datasets.
- `failure_error_summary.csv`: explicit failure count and error.
- `parse_failures_raw_output.csv`: parse-failure export; header-only because none occurred.
- `manifest.json`: frozen experiment configuration and environment summary.

Decompress the primary output with `gzip -dk predictions_final.csv.gz`, then verify it
against `predictions_final.csv.sha256`.
