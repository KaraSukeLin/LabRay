# Phase 4 handoff to Luna Max — archived

> **Status (2026-10-23): Phase 4 is complete.** All supported Experiment B conditions have full dev calibration and official test reports. See [EXPERIMENT_B_REPORT.md](EXPERIMENT_B_REPORT.md), [PRESENTATION_INSIGHTS.md](PRESENTATION_INSIGHTS.md), and the [interactive dashboard](artifacts/reports/phase4/test_threshold_dashboard.html). This file preserves the original pre-run plan for reproduction; its future-tense status text describes the state before execution.

The formal matrix used Elephant BF16 and StartLux Q4_K_M. Qwen LoRA and Elephant-native quantization remain blocked. No Experiment C work was started.

Elephant BF16 **B1–B4 are admitted** by [admission.json](artifacts/smoke/phase3-20261023/admission.json).
Read [PHASE3_REPORT.md](PHASE3_REPORT.md) for measurements, numerical limits and accepted constraints.
This is an execution handoff. No command below that starts Phase 4 was run during Phase 3.

Qwen LoRA and Elephant B5 quantization are BLOCKED independently. Keep them out of the formal training matrix. Qwen zero/four-shot and its Phase 2 BF16/BNB8 smoke remain separate capabilities. Full dev calibration and final evaluation remain future work.

## 1. Check the handoff and environment

```bash
cd /root/projects/decision
export B_PYTHON="$PWD/experiments/orsharc/artifacts/env/elephant-repro/bin/python"
B_ENV_ONLY=1 B_ENV="$PWD/experiments/orsharc/artifacts/env/elephant-repro" bash experiments/orsharc/setup_phase2.sh elephant
bash experiments/orsharc/artifacts/smoke/phase3-20261023/admission-command.sh
python3 experiments/orsharc/run.py validate --prepared experiments/orsharc/artifacts/prepared
```

Keep Torch 2.8.0/CUDA 12.8, Triton 3.7.1 and the exact lock. The single Torch/Triton metadata conflict is deliberate and checked explicitly by `environment-verify`. Do not run a dependency-resolving reinstall over it. Keep the working convolution fallback and single-GPU execution. Leave `CUDA_VISIBLE_DEVICES` consistent with the receipts (commands use `cuda:0`). Do not enable native full-state resume or distributed training.

Admission checks the pinned final code/configuration hashes and the actual native source and recipe. Changes require new relevant verification; do not edit the evidence index merely to accept changed code.

## 2. Generate and inspect the fixed plans

```bash
for B_MATRIX in b1 b2 b3 b4; do
  python3 experiments/orsharc/run.py matrix --name "$B_MATRIX" --model elephant \
    --run-root experiments/orsharc/artifacts/runs/phase4 \
    --out experiments/orsharc/artifacts/plans/phase4
done
```

The plans are already generated and syntax checked. They contain commands, not results. B1 and B4 command lists are identical. The scripts include **official test inference**, preceded by both complete dev passes and frozen calibration for each fixed condition. Execute them only as Phase 4 work. The matrix is predefined; do not choose models, formats, thresholds or checkpoints from test results.

## 3. Execute B1, then reuse its results for B4

```bash
# Starts formal work; trains all three budgets, then runs the fixed conditions.
bash experiments/orsharc/artifacts/plans/phase4/b1-elephant.sh
```

This uses the unchanged registered one-epoch native LoRA recipe. The individual training commands, for scheduling or recovery, are:

```bash
for B_SIZE in 256 1024 4096; do
  "$B_PYTHON" experiments/orsharc/run.py train --model elephant --size "$B_SIZE" \
    --out "experiments/orsharc/artifacts/training/elephant-$B_SIZE" --execute
done
```

Do not use `--max-steps` for formal runs. The saved checkpoint is `native/merged`; keep its adapter, complete readout metadata and `ft_exposure.json`. Inspect finite native logs, preflight zero-over-limit counts and `complete.json` before using a checkpoint. A full checkpoint must have `complete_budget: true` and rule IDs for its exact subset. The three training jobs should need about 40–55 minutes plus possible cold compilation. Reserve 2–4 H100 hours for the initial primary B1–B4 scope including evaluation/reporting.

Do not rerun B4 as a new grid. Its plan names the same checkpoints and output identities, so B1 artifacts supply B4. Existing completion files are accepted only after their hashes and identities are checked.

## 4. Run B2 and B3 in the same fixed environment

```bash
bash experiments/orsharc/artifacts/plans/phase4/b2-elephant.sh
bash experiments/orsharc/artifacts/plans/phase4/b3-elephant.sh
```

These plans use the released model. If the formal protocol uses the B1 FT4096 checkpoint for B2/B3, generate those fixed plans with the existing override:

```bash
for B_MATRIX in b2 b3; do
  python3 experiments/orsharc/run.py matrix --name "$B_MATRIX" --model elephant \
    --checkpoint experiments/orsharc/artifacts/training/elephant-4096/native/merged \
    --run-root experiments/orsharc/artifacts/runs/phase4 \
    --out experiments/orsharc/artifacts/plans/phase4-adapted
done
# Execute only the conditions required by the fixed formal protocol:
bash experiments/orsharc/artifacts/plans/phase4-adapted/b2-elephant.sh
bash experiments/orsharc/artifacts/plans/phase4-adapted/b3-elephant.sh
```

No new training is needed for these checkpoint overrides. Hierarchy uses two calls; parallel Noul retains all three probabilities, conflict and no-positive fields. Score5 remains an exploratory binary sufficiency projection; it has no five-level gold training task.

## 5. Keep dev selection, calibration and final evaluation separate

The generated scripts enforce the order for each fixed condition:

1. Complete `dev_select` (456 rows), then complete `dev_calibrate` (649 rows).
2. Select sufficiency thresholds only on `dev_select`. Fit temperature only on `dev_calibrate`.
3. Save calibration bound to the exact condition, then run the complete official test partition.
4. Generate raw and calibrated reports from the saved predictions.

For a manual or resumed condition:

```bash
B_RUN=experiments/orsharc/artifacts/runs/phase4/elephant-original-single_choice-bf16-0shot
"$B_PYTHON" experiments/orsharc/run.py calibrate \
  --select-run "$B_RUN/dev_select" --calibration-run "$B_RUN/dev_calibrate" \
  --out "$B_RUN/calibration.json"
# Only after successful complete dev calibration for this exact condition:
"$B_PYTHON" experiments/orsharc/run.py infer --model elephant --partition test --out "$B_RUN/test"
"$B_PYTHON" experiments/orsharc/run.py report --run "$B_RUN/test" \
  --calibration "$B_RUN/calibration.json" --out "$B_RUN/reports/test"
```

The manual example is for Original single Choice only. For adapted checkpoints or B2/B3, use every model/checkpoint/task/strategy/quantization argument from that plan. Never mix conditions. Never apply `--limit` to formal test inference, or weaken the complete-dev calibration guard. Do not change a frozen condition after seeing test outcomes.

## Recovery and provenance

- Keep all failed attempts and logs. Native OneJev LoRA does not restore optimizer/scheduler/RNG state: restart interrupted training in a **new** directory, then regenerate checkpoint-override evaluation plans. Do not use `--resume`.
- Inference supports matching-identity resume, tail repair and verified completion skip. An identity change requires a new `--run-root`; a lock filename alone does not prove a live process. Check the actual process before restarting.
- Preserve manifests, checkpoint hashes, environment receipts, native loss logs, events, all native probabilities, usage/timings and calibration JSON. A partial smoke checkpoint must never supply a formal condition.
- `snippet_seen` remains official full-train rule membership. `ft_exposed` in rows gives full nested-budget membership separately. Use the budget matching the completed FT checkpoint; smoke exposure is explicitly unknown unless its actual rows were audited. Do not use these fields as model input.
- Native active/reload/merged full-forward tolerances are 1e-6 / 0.02 with no changed labels. Production cached BF16 + FP32-head comparison permits 0.05 with no changed labels, as documented upstream. Large or label-changing discrepancies need investigation; do not overwrite evidence or silently widen tolerances.
- Keep Qwen LoRA and Elephant quantization blocked until separately verified. Do not spend the remaining Experiment B/C budget on speculative packaging, toolkit installs or quantization integrations.
