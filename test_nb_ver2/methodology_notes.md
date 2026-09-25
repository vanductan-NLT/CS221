# Methodology notes for Version 2

## Corrected interpretation of the main comparison

The Version 1 Kaggle reproduction is 76.05% macro BAcc, while the Version 1 paper-reference table gives 77.41% for Bespoke-MiniCheck-7B. It is **not valid** to explain this main difference by claiming that the paper used a per-dataset tuned threshold and the reproduction used a fixed 0.5 threshold.

The executed Bespoke notebook records a call to `scorer_7b.score(...)` and evaluates its returned labels. It does not show a controlled per-dataset dev-threshold study, and that absence cannot establish thresholding as the cause of the paper-versus-reproduction difference. Threshold behavior must therefore not be presented as the causal explanation for the main comparison.

The recorded run did use a Kaggle-specific setup, including `max_model_len=4096`, `chunk_size=500`, 2x T4 hardware, and the installed MiniCheck/vLLM stack. These are protocol and environment differences worth testing, but they are hypotheses—not confirmed causes—until Version 2 runs controlled ablations with a fixed dataset snapshot and explicitly recorded scoring protocol.

## Reporting rule

For every future experiment, record the dataset revision/split, model revision, inference library revision, context/chunking configuration, prediction-to-label rule, hardware, measured runtime, and whether any threshold was selected using a development split. Report original-paper, later/current-reference, and local-reproduction values in separate labeled columns.
