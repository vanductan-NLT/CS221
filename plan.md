Chốt \*\*HalluGuard-Qwen3-4B\*\*. Tôi đề xuất đúng \*\*4 step\*\*, không thêm. Mục tiêu cuối: có một notebook HalluGuard chạy được trên Kaggle, dùng đúng benchmark 29,320 mẫu hiện tại, rồi so trực tiếp với kết quả Bespoke cũ.



Có một nguyên tắc xuyên suốt: \*\*không sửa `test\_nb\_ver1`\*\*. Giữ nó làm raw baseline. HalluGuard đi vào `test\_nb\_ver2`.



HalluGuard khác Bespoke ở chỗ nó trả \*\*3 class\*\*: `GROUNDED`, `HALLUCINATED\_INTRINSIC`, `HALLUCINATED\_EXTRINSIC`; khi so với LLM-AggreFact binary thì map `GROUNDED → 1`, hai hallucination class → `0`. Model card chính thức cũng yêu cầu XML output và justification. :chatgpt-content-reference{index="0"}



\## Step 1 — Đóng băng baseline + sửa những lỗi methodology hiện tại



Mục tiêu: trước khi thay model, phải khóa lại chính xác “kết quả cũ” để sau này không so nhầm.



Agent cần làm:



\- Không đụng `test\_nb\_ver1`.

\- Tạo `test\_nb\_ver2/`.

\- Đọc lại 3 notebook + `readme.md` + `analysis.md`.

\- Tạo một file baseline machine-readable, ví dụ `baseline\_bespoke\_v1.json` hoặc CSV, lấy trực tiếp từ output notebook:

&#x20; - 11 dataset.

&#x20; - BAcc từng dataset.

&#x20; - average = \*\*76.05\*\*.

&#x20; - runtime total = \*\*\~265 phút\*\*.

&#x20; - sample count = \*\*29,320\*\*.

\- Ghi rõ `77.41` là \*\*reference result\*\*, còn `76.05` mới là \*\*kết quả Bespoke mà nhóm đã thực sự chạy trên Kaggle\*\*.

\- Sửa claim sai trong documentation về việc “paper tune threshold theo dataset”. Main comparison hiện tại không được giải thích bằng lý do đó.

\- Ghi rõ benchmark hiện tại là phiên bản \*\*11 datasets có RAGTruth\*\*, không được mô tả mơ hồ như chính xác cùng snapshot benchmark của paper EMNLP ban đầu.

\- Xóa hard-coded Hugging Face token khỏi notebook mới và docs.



\### Master prompt cho Agent



```text

You are working on this repository:



https://github.com/vanductan-NLT/CS221



Focus on:

test\_nb\_ver1/



Goal:

Prepare a clean Version 2 experiment workspace before introducing a new model. DO NOT modify, overwrite, clean, or refactor anything inside test\_nb\_ver1. Version 1 must remain an immutable raw baseline.



Tasks:



1\. Read all files in test\_nb\_ver1 carefully, especially:

&#x20;  - bespoke-minicheck-7b-full-benchmark-on-kaggle-2x.ipynb

&#x20;  - minicheck-llm-aggrefact-benchmark-evaluation-on-k.ipynb

&#x20;  - deepseek-api-fact-checking-full-benchmark-on-ka.ipynb

&#x20;  - readme.md

&#x20;  - analysis.md



2\. Create:

&#x20;  test\_nb\_ver2/



3\. Extract the ACTUAL executed Bespoke-MiniCheck-7B baseline from the notebook outputs into a machine-readable baseline file.

&#x20;  Preserve:

&#x20;  - all 11 dataset names

&#x20;  - sample count for each dataset

&#x20;  - reproduced BAcc for each dataset

&#x20;  - macro average across the 11 datasets

&#x20;  - total benchmark sample count

&#x20;  - total measured runtime



&#x20;  The reproduced Bespoke average currently visible in Version 1 is 76.05%, over 29,320 samples, with approximately 265 minutes total runtime. Verify these values from the notebook instead of blindly copying them.



4\. Clearly distinguish three concepts in Version 2 documentation:

&#x20;  A. original MiniCheck paper results

&#x20;  B. later/current LLM-AggreFact reference results

&#x20;  C. our own Kaggle reproduced results



&#x20;  Never label our reproduced value as a paper value.



5\. Correct the methodology mistake currently present in analysis.md:

&#x20;  do NOT claim that the main MiniCheck benchmark result differs because the paper tuned a per-dataset threshold while our run used 0.5.

&#x20;  That explanation is not valid for the main comparison.



6\. Keep the current Version 1 analysis untouched.

&#x20;  Put corrected methodology notes only in Version 2.



7\. Security:

&#x20;  - Do not copy any hard-coded Hugging Face token into Version 2.

&#x20;  - Read HF tokens only from environment variables / Kaggle Secrets.

&#x20;  - If any credential is found, do not print it.



8\. Create a concise test\_nb\_ver2/README.md explaining:

&#x20;  - purpose of Version 2

&#x20;  - immutable Version 1 baseline

&#x20;  - benchmark snapshot: 11 datasets / 29,320 test examples

&#x20;  - main evaluation metric: Balanced Accuracy

&#x20;  - old baseline to beat/compare: our reproduced Bespoke run, NOT only the published result



Do not implement HalluGuard yet.

Do not add unnecessary architecture, scripts, packages, or abstractions.

At the end, show exactly which files were created and what baseline values were extracted.

```



\### Mày cần làm thủ công



Chỉ \*\*1 việc\*\*: revoke Hugging Face token đã bị commit public và tạo token mới nếu còn cần. Sau này đưa token mới vào \*\*Kaggle Secrets\*\*, không paste vào notebook.



\---



\# Step 2 — Thay Bespoke bằng HalluGuard + smoke test



Mục tiêu: làm notebook chạy đúng trước, \*\*chưa chạy 29K mẫu\*\*.



Checkpoint chính thức:



`lrsbrgrn/HalluGuard-Qwen3-4B`



Nó là 4B, Qwen3 backbone, LoRA + ORPO, Apache 2.0 và model card hiện tại có sẵn prompt/output protocol. :chatgpt-content-reference{index="1"}



Paper ACL 2026 báo \*\*77.1 BAcc toàn LLM-AggreFact\*\*, nhưng chưa được phép lấy 77.1 thay cho kết quả của chúng ta; notebook mới vẫn phải tự chạy. :chatgpt-content-reference{index="2"}



Agent cần fork logic benchmark từ Bespoke notebook, nhưng \*\*không reuse `MiniCheck.score()`\*\*, vì HalluGuard có inference protocol riêng.



Flow:



```text

doc + claim

&#x20;   ↓

official HalluGuard prompt

&#x20;   ↓

HalluGuard-Qwen3-4B

&#x20;   ↓

<classification>...</classification>

&#x20;   ↓

GROUNDED                    → 1

HALLUCINATED\_INTRINSIC      → 0

HALLUCINATED\_EXTRINSIC      → 0

&#x20;   ↓

Balanced Accuracy

```



Smoke test chỉ dùng một subset nhỏ từ \*\*dev\*\*, không test set, chủ yếu kiểm tra:



\- model load được trên 2×T4;

\- prompt đúng;

\- XML parse đúng;

\- 3 class map đúng;

\- không OOM;

\- checkpoint/resume hoạt động;

\- output parse failure gần 0.



Không optimize generation config dựa vào labels của test set.



\### Master prompt cho Agent



```text

Continue working in:

test\_nb\_ver2/



We are replacing the old Bespoke-MiniCheck-7B experiment with:



lrsbrgrn/HalluGuard-Qwen3-4B



Before coding, inspect the latest official HalluGuard model card and ACL 2026 paper:

"HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation."



Use the authors' documented inference protocol as the source of truth.



Goal:

Create ONE Kaggle notebook for HalluGuard that follows the same benchmark structure as our old Bespoke notebook, but uses HalluGuard's native inference format.



Requirements:



1\. Create a new notebook under test\_nb\_ver2.

&#x20;  Use a clear name such as:

&#x20;  halluguard-qwen3-4b-llm-aggrefact-kaggle.ipynb



2\. Target environment:

&#x20;  Kaggle Notebook

&#x20;  2x NVIDIA T4 16GB

&#x20;  Internet enabled.



3\. Dataset:

&#x20;  lytang/LLM-AggreFact



&#x20;  For the smoke-test stage, use the DEV split only.

&#x20;  Do NOT evaluate the final test split yet.



4\. Model:

&#x20;  lrsbrgrn/HalluGuard-Qwen3-4B



5\. Do NOT route HalluGuard through MiniCheck.score().

&#x20;  HalluGuard has its own generation + classification protocol.



6\. Use the official HalluGuard document/claim prompt structure from the model card.

&#x20;  Do not invent or simplify the task prompt.



7\. Parse the official XML classification output:

&#x20;  <classification>...</classification>



&#x20;  Map HalluGuard's three labels to LLM-AggreFact binary labels exactly as:



&#x20;  GROUNDED -> 1

&#x20;  HALLUCINATED\_INTRINSIC -> 0

&#x20;  HALLUCINATED\_EXTRINSIC -> 0



8\. Preserve the original 3-class prediction as a separate column.

&#x20;  Also preserve the generated raw response so parsing failures can be audited.



9\. Add robust parsing.

&#x20;  Never silently map malformed output to 0 or 1.

&#x20;  Parsing failures must be recorded explicitly as PARSE\_ERROR.



10\. Use a small, stratified DEV smoke test only.

&#x20;   The purpose is engineering validation, not model tuning.

&#x20;   Do not select decoding parameters based on test-set accuracy.



11\. Verify:

&#x20;   - model loads

&#x20;   - inference succeeds

&#x20;   - no OOM

&#x20;   - all three valid labels are parsed correctly

&#x20;   - binary mapping is correct

&#x20;   - checkpoint/resume works

&#x20;   - results retain the original dataset row/index

&#x20;   - parse-error rate is reported



12\. Reuse only the necessary evaluation structure from Version 1:

&#x20;   - dataset name

&#x20;   - label

&#x20;   - predicted label

&#x20;   - Balanced Accuracy

&#x20;   - checkpoint saving



13\. Also measure:

&#x20;   - wall-clock runtime

&#x20;   - samples/sec

&#x20;   - peak GPU memory if reliably available



These are necessary because the research goal is cost-efficient factuality evaluation.



14\. Do NOT add large error-analysis sections, decorative charts, or unrelated metrics yet.

&#x20;   First make inference correct and reproducible.



15\. Use Kaggle Secrets/environment variables for any Hugging Face credential.

&#x20;   No secret may appear directly in notebook source.



At the end of the notebook, print a concise smoke-test summary:

\- number of examples

\- valid predictions

\- parse errors

\- runtime

\- throughput

\- BAcc for the DEV sample (clearly labeled as smoke-test only)



Do not run or fabricate the full benchmark results.

```



\### Mày cần làm thủ công



Trên Kaggle:



1\. bật \*\*GPU T4 ×2 + Internet\*\*;

2\. add `HF\_TOKEN` trong Kaggle Secrets nếu model/dataset cần;

3\. chạy notebook smoke test;

4\. chỉ kiểm tra nó \*\*run hết, không OOM, parse error không bất thường\*\*.



Không chỉnh prompt vì thấy vài sample sai.



\---



\# Step 3 — Chạy full benchmark HalluGuard



Khi smoke test ổn, Agent chuyển notebook sang full run.



Không tune gì thêm dựa trên test result.



Full run:



```text

LLM-AggreFact test

29,320 samples

11 datasets

&#x20;       ↓

HalluGuard

&#x20;       ↓

Binary mapping

&#x20;       ↓

BAcc / dataset

&#x20;       ↓

Macro average

```



Quan trọng nhất: phải có checkpoint/resume vì reasoning model có thể chạy lâu.



\### Master prompt cho Agent



```text

The HalluGuard DEV smoke test is now working.



Convert the existing HalluGuard Version 2 notebook into the FINAL full benchmark notebook.



Do not redesign the experiment and do not change the prompt/model configuration merely to improve accuracy.



Requirements:



1\. Switch evaluation from DEV smoke-test mode to:

&#x20;  lytang/LLM-AggreFact TEST split



2\. Evaluate ALL 29,320 test examples across all 11 datasets.



3\. Preserve exactly the label mapping already validated:

&#x20;  GROUNDED -> 1

&#x20;  HALLUCINATED\_INTRINSIC -> 0

&#x20;  HALLUCINATED\_EXTRINSIC -> 0



4\. Keep HalluGuard's original 3-class prediction and raw response.



5\. Implement reliable checkpoint/resume.

&#x20;  At minimum save progress frequently enough that a Kaggle session interruption does not require restarting the complete benchmark.



6\. Do not mark a dataset complete unless every expected row has a valid stored result or an explicitly recorded failure.



7\. Parsing/API/runtime failures must NEVER silently become negative predictions.

&#x20;  Keep them as explicit errors and report their count.



8\. Calculate the exact same primary metric as Version 1:

&#x20;  Balanced Accuracy for each of the 11 datasets.



9\. Calculate final benchmark score as the unweighted macro mean of the 11 dataset BAcc values, matching the Version 1 comparison method.



10\. Record:

&#x20;   - total samples

&#x20;   - BAcc per dataset

&#x20;   - macro-average BAcc

&#x20;   - total runtime

&#x20;   - throughput in samples/sec

&#x20;   - parse-error count/rate

&#x20;   - peak GPU memory if reliably measurable

&#x20;   - model name and inference configuration



11\. Export:

&#x20;   - full per-example predictions CSV

&#x20;   - per-dataset metrics CSV

&#x20;   - final summary CSV or JSON



12\. Keep the notebook focused.

&#x20;   Do not add speculative root-cause explanations or tune against the test labels.



13\. Add a final validation cell that checks:

&#x20;   - expected sample count == 29320

&#x20;   - exactly 11 datasets

&#x20;   - no duplicate benchmark rows

&#x20;   - binary predictions only contain 0/1 for successfully parsed examples

&#x20;   - number of processed rows matches expected rows

&#x20;   - average is computed from the 11 dataset BAcc values



Do not fabricate any missing values.

```



\### Mày cần làm thủ công



Chỉ việc \*\*Run All trên Kaggle\*\*.



Sau khi xong:



\- download executed `.ipynb`;

\- download 3 result files CSV/JSON;

\- bỏ chúng vào `test\_nb\_ver2/results/`.



Đừng copy số bằng tay.



\---



\# Step 4 — So HalluGuard với Bespoke cũ + update documentation



Đây mới là step kết luận.



Primary comparison phải là:



```text

Bespoke old run

7B

29,320 samples

Kaggle 2x T4

BAcc 76.05



VS



HalluGuard new run

4B

29,320 samples

Kaggle 2x T4

BAcc = actual result vừa chạy

```



Ngoài ra có thể ghi HalluGuard paper \*\*77.1\*\* làm external reference riêng, không trộn với số mình chạy. HalluGuard authors report 77.1 BAcc across the benchmark and 84.4 on RAGTruth. :chatgpt-content-reference{index="3"}



Chỉ cần compare:



\- BAcc từng dataset;

\- macro BAcc;

\- model size: 7B vs 4B;

\- runtime;

\- throughput;

\- peak VRAM nếu cả hai bên thực sự có dữ liệu;

\- parse/error rate HalluGuard.



\*\*Không được chế VRAM của Bespoke\*\* vì Version 1 không lưu một con số peak-VRAM chuẩn. Nếu thiếu → `N/A`.



\### Master prompt cho Agent



```text

The full HalluGuard benchmark has completed.



Now perform the FINAL Version 2 comparison.



Inputs:

\- immutable Version 1 Bespoke notebook/results

\- test\_nb\_ver2 baseline file created in Step 1

\- executed HalluGuard notebook

\- HalluGuard full result files



Goal:

Produce a factual apples-to-apples comparison between our OLD Bespoke run and our NEW HalluGuard run.



Rules:



1\. Primary comparison must use OUR reproduced values:

&#x20;  Bespoke Version 1 vs HalluGuard Version 2.



2\. Do not substitute published paper scores for our measured scores.



3\. HalluGuard's published 77.1 BAcc may be shown separately as an external reference only.



4\. Build one comparison table containing:

&#x20;  - 11 dataset names

&#x20;  - Bespoke Version 1 BAcc

&#x20;  - HalluGuard Version 2 BAcc

&#x20;  - HalluGuard minus Bespoke difference



5\. Add final summary:

&#x20;  - macro-average BAcc

&#x20;  - parameter count: Bespoke \~7B vs HalluGuard 4B

&#x20;  - total runtime

&#x20;  - throughput

&#x20;  - peak GPU VRAM only when actually measured

&#x20;  - parsing failure rate for HalluGuard



6\. Do not invent missing cost metrics.

&#x20;  If Version 1 did not record a reliable metric, use N/A.



7\. Update test\_nb\_ver2/README.md with:

&#x20;  - experiment goal

&#x20;  - model replacement rationale

&#x20;  - exact benchmark/evaluation protocol

&#x20;  - HalluGuard 3-class -> binary mapping

&#x20;  - final measured comparison

&#x20;  - limitations



8\. Keep methodology accurate:

&#x20;  - do not claim the old Bespoke gap was caused by missing per-dataset threshold tuning

&#x20;  - distinguish original MiniCheck publication, later benchmark references, and our own runs

&#x20;  - explicitly state when a number comes from our experiment versus an external paper



9\. Add only evidence-supported interpretation.

&#x20;  Example acceptable conclusions:

&#x20;  - HalluGuard used fewer parameters

&#x20;  - HalluGuard was faster/slower by measured runtime

&#x20;  - BAcc increased/decreased by measured amount



&#x20;  Do NOT speculate about why accuracy changed unless supported by a dedicated experiment.



10\. Keep Version 1 untouched.



At the end, give me:

\- final comparison table

\- files changed

\- 3–5 evidence-based findings suitable for the project report

\- any remaining experimental limitation that materially affects the comparison

```



\### Mày cần làm thủ công



Không cần tính toán gì.



Chỉ review final table xem các số có đúng output Kaggle rồi commit.



\---



\## Flow cuối cùng



```text

Step 1

Freeze Bespoke baseline + correct methodology

&#x20;            ↓

Step 2

Implement HalluGuard + DEV smoke test

&#x20;            ↓

&#x20;    \[Mày chạy Kaggle]

&#x20;            ↓

Step 3

Full 29,320 benchmark

&#x20;            ↓

&#x20;    \[Mày chạy Kaggle]

&#x20;            ↓

Step 4

Agent đọc results → compare → update README/report

```



Có một chỗ tôi cố tình \*\*không đưa vào plan\*\*: threshold tuning. HalluGuard hiện là generative 3-class classifier, không phải probability classifier giống MiniCheck cũ; ép thêm threshold tuning lúc này vừa không cần thiết vừa làm comparison bẩn. :chatgpt-content-reference{index="4"}

