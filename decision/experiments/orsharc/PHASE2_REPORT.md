# Phase 2 OR-ShARC smoke-test report

Subsequent status: [Phase 3](PHASE3_REPORT.md) verifies B1–B4 admission and
provides the [Phase 4 handoff](PHASE4.md). This report and its gate remain the
unchanged historical Phase 2 evidence.

**Date:** 2026-10-23 UTC  
**Scope:** bounded Phase 2 smoke tests. No full matrix, final benchmark, or official test inference ran.

## Outcome

Elephant-1 is the recommended primary model. Its BF16 inference and a two-step native LoRA training smoke passed after a local Triton version adjustment. The merged checkpoint loaded and returned predictions. The smoke gate remains `formal_experiment_admission: false`; Phase 3 must complete the remaining parameter, merge-parity, and long-input checks before formal runs.

All predictions and metrics in this report are pipeline diagnostics. They are not research results.

## Test matrix

| Area | Status | Evidence and limit |
|---|---|---|
| Data preparation, hashes, labels, nested subsets, oracle rule, Seen/Unseen metadata, and leakage checks | **PASS** | `prepare --download`, `validate`, and `check` passed. The prepared manifest, audit, and partitions hashes did not change after preparation. The check confirms inference does not open gold. |
| Elephant Choice inference and repeat load | **PASS** | Original and reload runs each saved 8 valid train rows. All probabilities were finite and summed to 1. The smoke gate reports a maximum original/reload probability difference of 0.0. |
| Elephant Noul, Choice2, and Score5 | **PASS** | Each returned 4 complete train rows with valid probabilities. Score5 retained its five native options and projected the expected score to the binary sufficiency decision. |
| Elephant native LoRA training and adapted load | **PASS** | Two optimizer steps used 16 examples from the 256-row training subset. The trainer reported finite loss, 43.28M trainable parameters (0.481%), saved an adapter and merged checkpoint, and the adapted checkpoint returned 8 predictions. |
| Trainable/frozen tensor audit and adapted save/merge parity | **NOT TESTED** | The smoke gate leaves these as admission checks. The logged parameter count does not prove optimizer membership or frozen backbone integrity. |
| B1/B4 Original vs. FT and budgets | **PASS for smoke scope** | A matched 4-row `dev_select` Original and FT256 run produced reports. Both generated plans contain Original and FT conditions at 256, 1,024, and 4,096 rows; their command lists match. The full budgets and matrices did not run. |
| B2 formats and Score threshold mapping | **PASS** | Noul, binary Choice2, and Score5 all ran. The five native Score probabilities were finite; the expected score was projected to a binary probability and the 0.5 rule selected the binary label. |
| B2 dev-only threshold sweep | **NOT TESTED** | The calibration command rejected the 8-row smoke slices because it requires all 456 `dev_select` and all 649 `dev_calibrate` rows. No full development calibration run was made. |
| B3 single, hierarchical, parallel Noul, aggregation, and conflict logging | **PASS** | Single Choice ran in the main smoke. Hierarchical and parallel Noul each returned 4 rows. Recomputed aggregation matched saved outputs within floating-point tolerance. The 4 parallel rows logged conflict/no-positive fields; 2 had no positive vote and none had multiple positive votes. The CPU check also passed its conflict fixture. |
| B5 BF16 and quantized inference | **PASS for tested paths** | Qwen BF16 and BNB8 each ran on the same 4 train rows and checkpoint. `compare-quantization` accepted the identities; all 4 labels matched and the maximum probability difference was 0.1134. This is a smoke check only. |
| Calibration metrics and dashboard | **PASS for raw report; calibration fit NOT TESTED** | The 8-row dev report wrote ECE, Brier, NLL, risk–coverage/AURC, reliability plots, and an interactive dashboard. Its JavaScript passed `node --check`. Temperature fitting and threshold selection were not run on complete dev partitions. The Phase 1 temperature-NLL fixture passed. |
| Manifests, predictions, logs, timing, VRAM, and checkpoint metadata | **PASS for smoke artifacts** | Fourteen inference runs saved 76 rows in total. Runs have manifests, completion hashes, unique IDs, finite normalized probabilities, and positive per-call latencies. Training logs and adapter/merged checkpoint hashes are present. |
| StartLux and Kev fallbacks | **NOT TESTED** | Elephant completed the preferred train/inference path, so fallback model setup was not needed. StartLux v1.1 adaptation remains unsupported by the inspected upstream trainer. |
| Qwen Unsloth LoRA training | **FAIL at import setup; not trained** | The documented Qwen setup installed the pinned Unsloth source, but `import unsloth` failed because `unsloth_zoo` is missing. This did not block the Transformers BNB8 inference path. |
| Official final test evaluation | **NOT RUN by design** | Every smoke prediction manifest names `train_256`, `dev_select`, or `dev_calibrate`. No smoke manifest uses `test`. |

## Data fairness and leakage

The source files match the pinned OR-ShARC revision and hashes. Preparation produced 17,936 train, 1,105 dev, and 2,373 test rows. The 256, 1,024, and 4,096 training subsets are nested, use seed 42, and retain four fixed demonstrations. Dev is split by tree into 456 selection and 649 calibration rows.

The official mapping is Yes/No to those labels and every other non-empty answer to Inquire. The pipeline rejects empty or unexpected answers. Inputs contain the oracle rule, question, scenario, and history; they omit the gold answer and evidence. The input allowlist and inference file-open guard passed.

Official rule/tree overlap remains in the source split. The audit reports 257 shared train/dev rules and trees, and 373 shared train/test rules with 377 shared train/test trees. Seen means present in the full official training split, not necessarily in the sampled subset. Dev has 500 Seen and 605 Unseen examples. These overlaps are documented; this smoke did not claim a leak-free disjoint rule split.

## Environment and resource observations

- Host: five idle NVIDIA H100 80 GB GPUs; tests used GPU 0 only. Driver 570.86.10 reports CUDA 12.8. PyTorch 2.8.0+cu128 saw all five GPUs and BF16 support.
- Disk had about 12 TB free before setup. The pinned Elephant and Qwen checkpoints downloaded to 18 GB and 19 GB model directories.
- Elephant environment: Python 3.12.14, Transformers 5.19.0, Flash Linear Attention 0.5.2, Triton 3.7.1. The smoke training log reports 79.9 seconds total, 17.6 GB peak VRAM, 16 examples, and 2 steps. Step 1 took 76.6 seconds during kernel setup; step 2 took 2.7 seconds.
- The initial Torch install resolved Triton 3.4.0. Flash Linear Attention stopped the first H100 backward pass because that version is known to produce incorrect results for the gated chunk backward kernel on Hopper. Installing Triton 3.7.1 in the isolated environment allowed the same bounded training smoke to pass. The successful freeze is [elephant.freeze.txt](artifacts/env/elephant.freeze.txt). Triton’s official release matrix lists Python 3.12 wheels for 3.7.1 ([release matrix](https://github.com/triton-lang/triton/blob/main/RELEASE.md)).
- `causal-conv1d` did not build because no `nvcc` executable is available. The build then raised `NameError: bare_metal_version is not defined`. Inference and training used the slower PyTorch fallback and completed. The installed `nvidia-cuda-nvcc-cu12` wheel supplied `ptxas`, but local inspection still found no `nvcc`.
- Elephant model loads used about 23.6 GB on GPU 0. Qwen BNB8 used about 11.0 GB. Qwen used PyTorch 2.8.0+cu128, Transformers 5.20.2, and bitsandbytes 0.50.2. BNB8 logged that it casts BF16 inputs to FP16 for MatMul8bitLt.
- Jobs ran one at a time. No job approached the two-hour Phase 2 ceiling. The trainer records elapsed time and VRAM; inference manifests record per-call latency. There is no single aggregate GPU-hour counter in the artifacts.

## Ranked issues and Phase 3 handoff

1. **High — Hopper Triton compatibility is not encoded in setup.** The default Torch install supplied Triton 3.4.0, which blocked Elephant training on H100. The environment-only update to 3.7.1 passed the smoke. Pin a compatible Triton/TileLang path in the runtime setup and keep the resolved freeze with the run.
2. **Medium — Qwen Unsloth setup is incomplete.** `unsloth` import stops on missing `unsloth_zoo`. Add a matching companion dependency and test the Qwen LoRA recipe only if Qwen training remains a required path. Do not use this result to claim Unsloth training compatibility.
3. **Medium — Native convolution extension is unavailable on this host.** `causal-conv1d` cannot compile without `nvcc`; inference and the smoke trainer fell back to PyTorch. Provide a development toolkit or accept the slower fallback after measuring the longest inputs.
4. **Environment — PyTorch dependency downloads from `pypi.nvidia.com` fail TLS on this host.** The documented setup stopped after three retries for NVIDIA CUDA wheels. Adding `https://pypi.org/simple` as an extra index completed the Torch install. Both the failed and successful commands are logged.
5. **Calibration scope — The calibration CLI requires both complete dev partitions.** A bounded 8+8 run is rejected by design. Keep this requirement for formal calibration; use CPU fixtures for smoke checks, or plan the full dev-only calibration as a separate, budgeted action.

**Recommendation:** continue to Phase 3 before Phase 4. Keep Elephant-1 as primary. First make the successful Triton environment reproducible, then complete the LoRA/frozen-weight audit, adapted merge parity, and longest-input memory/throughput profile. Do not admit formal runs while `formal_experiment_admission` remains false.

## Commands and artifact paths

The exact shell commands and tracebacks are in the linked console logs. The key commands were:

```bash
python3 experiments/orsharc/run.py prepare --download
python3 experiments/orsharc/run.py validate --prepared experiments/orsharc/artifacts/prepared
python3 experiments/orsharc/run.py check --prepared experiments/orsharc/artifacts/prepared
bash experiments/orsharc/setup_phase2.sh elephant
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  B_ATTEMPT=20261023T065300Z bash experiments/orsharc/phase2_smoke.sh elephant
# After the recorded Triton 3.7.1 environment adjustment:
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  bash experiments/orsharc/phase2_smoke.sh elephant
```

The successful and failed Elephant smoke artifacts are in [elephant-20261023T065538Z](artifacts/smoke/elephant-20261023T065538Z/), [first attempt](artifacts/smoke/elephant-20261023T065300Z/), and the [console/setup logs](artifacts/smoke/phase2-preflight-20261023T0645Z/elephant-smoke-triton371.log). The final gate is [gate.json](artifacts/smoke/elephant-20261023T065538Z/gate.json); training logs and checkpoint metadata are in its `training/` directory.

B2, B3, calibration, B1 dev, and B5 smoke outputs are in [functional-20261023T0701Z](artifacts/smoke/functional-20261023T0701Z/). The B1/B4 plans are [b1-elephant.json](artifacts/plans/b1-elephant.json) and [b4-elephant.json](artifacts/plans/b4-elephant.json). The Qwen precision comparison is [b5-qwen-comparison.json](artifacts/smoke/functional-20261023T0701Z/b5-qwen-comparison.json). The raw dev metrics and dashboard are in [calibration/raw-report](artifacts/smoke/functional-20261023T0701Z/calibration/raw-report/).

Setup failures and environment adjustments are recorded in [setup-elephant.log](artifacts/smoke/phase2-preflight-20261023T0645Z/setup-elephant.log), [causal-conv1d.log](artifacts/smoke/phase2-preflight-20261023T0645Z/causal-conv1d.log), [triton-3.7.1.log](artifacts/smoke/phase2-preflight-20261023T0645Z/triton-3.7.1.log), [setup-qwen.log](artifacts/smoke/phase2-preflight-20261023T0645Z/setup-qwen.log), and [qwen-compatibility.log](artifacts/smoke/phase2-preflight-20261023T0645Z/qwen-compatibility.log). The resolved Qwen package list is [qwen.freeze.txt](artifacts/env/qwen.freeze.txt).
