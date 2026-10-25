# OR-ShARC: Experiment B

Current status (2026-10-23): **Experiment B Phase 4 complete**.
See the [full results](EXPERIMENT_B_REPORT.md), [presentation insights](PRESENTATION_INSIGHTS.md),
[interactive threshold dashboard](artifacts/reports/phase4/test_threshold_dashboard.html),
and [matrix verification receipt](artifacts/smoke/phase4-preflight-20261023T1355Z/verified-phase4-matrix.json).
All 14 unique conditions have complete dev calibration and test reports. Qwen LoRA
and Elephant-specific quantization remain blocked. B5 uses StartLux Q4_K_M; it is
not an Elephant quantization result. The Phase 1 and Phase 2 text below is retained
for historical context.

Phase 1 implements the pipeline and checks CPU behavior. GPU paths are **candidates
for Phase 2 validation**. Run commands from the project root. CPU commands need
Python 3.11+ and its standard library. GPU environments use Python 3.12.

## Prepare and check

```bash
python3 experiments/orsharc/run.py prepare --download
python3 experiments/orsharc/run.py validate --prepared experiments/orsharc/artifacts/prepared
python3 experiments/orsharc/run.py check --prepared experiments/orsharc/artifacts/prepared
python3 experiments/orsharc/run.py bootstrap --sources onejev kev startlux unsloth
uvx ruff check experiments/orsharc
uvx ruff format --check experiments/orsharc
```

`prepare` downloads four small official data files at the revision and SHA-256
hashes in `configs/sources.json`. Existing files must match. It produces:

- `inputs/{train,dev,test}.jsonl`: oracle rule, initial question, scenario and history;
  no gold answer or `evidence` field.
- `sealed/{train,dev,test}.jsonl`: original answers and mapped labels; mode 0600.
- `partitions.json`: seed-42 nested 256/1,024/4,096 subsets, fixed four train
  demonstrations and disjoint dev selection/calibration groups.
- `manifest.json`, `audit.json`: hashes, counts, rule/tree overlap and label audit.

The official split sizes are 17,936 / 1,105 / 2,373. Seen means present in the
**full official training split**, even when a training subset does not include
that rule. Official rule/tree overlap is retained. The dev split is partitioned
by tree, with 456 selection and 649 calibration examples. This does not change
the official split. Report each dev partition by its name; do not present either
as a complete official dev score.

The source README says `gole_snippet_id`; the actual files use `gold_snippet_id`.
The official evaluator maps Yes/No case-insensitively and every other follow-up
answer to `more`. This pipeline displays those classes as Yes / No / Inquire.
It audits follow-ups without question marks rather than removing them. It
rejects empty answers and unexpected classification labels. Source bytes are
pinned, so a new label distribution cannot pass without a new data review.

## Models and training

`configs/models.json` selects native adapters and pinned model revisions.
Weights live in `/root/model/orsharc-{elephant,startlux,kev,qwen}`. A supplied
`--checkpoint` replaces that directory and hashes every checkpoint file.
`--model-config path.json` accepts a complete alternative profile. Inputs,
tasks and metrics remain common to all adapters.

| Model | Inference | Adaptation | Quantization candidates |
|---|---|---|---|
| Elephant-1 9B | Official OneJev + released `serve.patch` | Official patched `train.sft`, LoRA | BF16; other precisions unsupported here |
| StartLux v1.1 9B Instant | Official Transformers or vLLM engine | Unsupported: published trainer is for v1 | Matched vLLM BF16 / FP8 |
| Kev-9B | Official checkpoint/API/model code, T=1 | Official `kev.train --init_from`, LoRA + existing pointer head | BF16; CUDA int8/4bit unsupported here |
| Qwen3.5-9B | Transformers next-letter likelihood, thinking off | Optional Unsloth + TRL, LoRA only | BF16 / bitsandbytes int8; optional NF4 PTQ |

Use [setup_phase2.sh](setup_phase2.sh) in Phase 2. It creates **separate**
environments; StartLux reference requires Transformers 5.8, while the inspected
Kev revision requires 5.17+. This is not a verified combined dependency lock.
The script records the resolved package freeze. CUDA/toolchain installation
still needs validation on the execution host. See [upstream evidence](UPSTREAM.md).

Training plans and export can be checked without a GPU:

```bash
python3 experiments/orsharc/run.py export --size 256 --recipe onejev --out experiments/orsharc/artifacts/exports/elephant-256
python3 experiments/orsharc/run.py train --model elephant --size 256 --out experiments/orsharc/artifacts/training/elephant-256
```

After Phase 2 gates pass, run the same `train` command with `--execute` in the
model's environment. Registered hyperparameters are in `configs/training.json`.
The primary recipe is a one-epoch domain adaptation, not a reproduction of
upstream pretraining. Native heads, losses, rendering and quantization code
come from upstream. Elephant's attention/readout settings are read from the
checkpoint and explicitly passed to its trainer. The wrapper copies native
readout metadata into the merged directory because upstream omits that copy.

All LLM adaptation trains LoRA only. Qwen uses completion-only next-letter SFT,
with the same prompt as its inference adapter. It has no new decision head.
Native training preflight rejects over-context records before the trainer can
filter them. Score has no five-level gold, so Score training is rejected.

## Matrices and inference

Generate each matrix independently; generation does not execute its commands:

```bash
for B_MATRIX in b1 b2 b3 b4 b5; do
  python3 experiments/orsharc/run.py matrix --name "$B_MATRIX" --model elephant
done
python3 experiments/orsharc/run.py matrix --name b1 --model qwen --include-optional
python3 experiments/orsharc/run.py matrix --name b5 --model qwen --include-optional
python3 experiments/orsharc/run.py matrix --name b5 --model startlux
```

Scripts and JSON command plans go to `artifacts/plans/`. Unsupported cells are
listed in the JSON plan; they are not silently replaced. Add `--checkpoint PATH`
to reuse any matrix with an existing checkpoint, including a fine-tuned one.
Checkpoint paths get distinct run names. Use `--run-root NEW_DIRECTORY` for new
inference attempts after a code, environment or recipe change.
For example, B2/B3 can use the B1 4,096-example checkpoint without new training.
Execute generated scripts only in Phase 2, after the required GPU gates.

| Matrix | Comparison |
|---|---|
| B1 | Original vs. adapted 256/1,024/4,096; Seen/Unseen; Qwen zero/four-shot and optional LoRA |
| B2 | Native Noul, binary Choice and exploratory Score on one binary sufficiency target |
| B3 | Main 3-way Choice, hierarchy and parallel Noul; conflicts, propagation and costs |
| B4 | Same nested training grid as B1; identical output IDs allow safe reuse/skip |
| B5 | Same source checkpoint, examples and engine across precision conditions |

Example inference (Phase 2):

```bash
python experiments/orsharc/run.py infer --model elephant --partition dev_select --out experiments/orsharc/artifacts/runs/elephant-original-single_choice-bf16-0shot/dev_select
```

Each inference manifest binds model/file hashes, source runtime receipt,
library versions, code hash, seed, inputs, demonstration IDs and task settings.
Each saved row includes all option probabilities, native outputs, option order,
rule split, IDs, per-call timing, tokens and composition decisions. Gold is joined
only by the report command. An exclusive writer lock protects each run. Resume
checks the full run identity, ordered IDs and saved row identity; a partial final
JSON write is repaired. Complete runs are skipped only after hash verification.

## Calibration and reports

Generated scripts run dev selection, dev calibration, freeze calibration, and
then test inference. Reports can be regenerated from saved files with no model:

```bash
B_RUN=experiments/orsharc/artifacts/runs/elephant-original-single_choice-bf16-0shot
python3 experiments/orsharc/run.py calibrate --select-run "$B_RUN/dev_select" --calibration-run "$B_RUN/dev_calibrate" --out "$B_RUN/calibration.json"
python3 experiments/orsharc/run.py report --run "$B_RUN/test" --calibration "$B_RUN/calibration.json" --out "$B_RUN/reports/test"
```

Outputs include scored JSONL with gold/prediction, `metrics.json`, standalone
reliability SVGs and `dashboard.html`. Open the HTML locally to filter by rule
split, prediction version and confidence. It uses no external scripts or CDN.
The slider is exploratory; it does not modify frozen test decisions.

Accuracy, Macro-F1, official-style macro accuracy, 10-bin ECE, multiclass Brier
(class sum), NLL, tied risk–coverage curves and AURC use the saved complete
probabilities. Raw and calibrated outputs are separate. Temperature fits NLL
on `dev_calibrate` only, after task projection/composition. Sufficiency thresholds
1–99% are selected on `dev_select` only; ties prefer 50%, then the smaller value.
The frozen classifier's labels remain fixed after temperature scaling.

Score probabilities are preserved over all five native options. Expected level
divided by four is the exploratory binary sufficient probability. Binary gold
is Yes/No → sufficient and Inquire → insufficient. No five-level accuracy,
five-level gold labels or comparison against the main benchmark is invented.

Hierarchy computes two separate calls: sufficiency, then conditional Yes/No.
Both stages are measured on every example to retain complete joint probabilities.
The gate chooses Inquire below 0.5; otherwise the terminal choice wins. Parallel
Noul asks three independent questions together. Exactly one score ≥0.5 wins;
zero or multiple positives fall back to Inquire. Normalized support scores are
a heuristic distribution; all original Noul probabilities remain available.

For B5, also run `compare-quantization --runs BF16_RUN QUANTIZED_RUN --out comparison.json`.
It rejects changed checkpoint bytes, input IDs, option order, task or engine.
Never use an unrelated prequantized checkpoint for this comparison.

Native OneJev and Kev LoRA trainers lack full-state resume. Their interrupted
attempts must be retained and restarted in a new directory. Qwen exposes Trainer
resume; exact optimizer/RNG/sampler parity is a Phase 2 gate. Changed code,
environment, data or recipe requires a new run path. No formal test run may use
`--limit`; short runs are train/dev smoke artifacts only.
