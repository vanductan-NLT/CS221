# Version 2 experiment workspace

This directory is the clean starting point for the next experiment. It contains no new model implementation yet; HalluGuard is intentionally not introduced here.

## Version 1 is immutable

`../test_nb_ver1/` is the raw Version 1 baseline. It is retained exactly as recorded and must not be modified, overwritten, cleaned, or refactored. The Version 2 baseline snapshot was transcribed from the executed output of `bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb`, rather than inferred from prose or copied from a publication.

## Benchmark snapshot and comparison target

The benchmark snapshot comprises 11 LLM-AggreFact test datasets and 29,320 examples. The primary metric is Balanced Accuracy (BAcc), reported as an unweighted macro average over the 11 datasets.

The baseline a future model must beat or compare against is **our reproduced Bespoke-MiniCheck-7B run**: **76.05% macro BAcc**, 29,320 examples, and **15,901.3 seconds (265.0 minutes)** measured runtime. Per-dataset sample counts, BAcc values, and runtimes are in [bespoke_minicheck_7b_reproduced_baseline.json](bespoke_minicheck_7b_reproduced_baseline.json).

## Result labels: keep these distinct

| Label | Meaning | How to use it |
| --- | --- | --- |
| Original MiniCheck paper results | Results reported in the EMNLP 2024 MiniCheck paper (for example, the Version 1 table reports Bespoke-MiniCheck-7B at 77.41% macro BAcc). | Historical published reference only. |
| Later/current LLM-AggreFact reference results | Results from a later or current LLM-AggreFact source or evaluation release. | Cite the exact release, model, and protocol; do not call them paper results. |
| Our Kaggle reproduced results | This repository's executed Version 1 Kaggle run, captured here. | Primary local baseline: Bespoke-MiniCheck-7B = 76.05% macro BAcc. |

Do not label 76.05% as a paper result. Do not silently substitute a later/current LLM-AggreFact reference value for either the original paper value or our reproduced baseline.

## Credentials

Version 2 must read Hugging Face credentials from an environment variable (for example, `HF_TOKEN`) or Kaggle Secrets only. Do not place credentials in notebooks, source files, snapshots, logs, or documentation.

## Methodology

See [methodology_notes.md](methodology_notes.md) for the corrected interpretation of the Version 1 comparison.
