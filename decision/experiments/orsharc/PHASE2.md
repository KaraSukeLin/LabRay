# Phase 2 handoff for Luna Max

Phase 1 is complete only for source, configuration, data preparation and CPU
fixtures. **No GPU gate has passed yet.** Start here on the GPU execution host.
This document is an execution plan, not a record of completed experiments.

## Exact next commands

```bash
cd /root/projects/decision
python3 experiments/orsharc/run.py prepare --download
python3 experiments/orsharc/run.py check --prepared experiments/orsharc/artifacts/prepared
bash experiments/orsharc/setup_phase2.sh elephant
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  bash experiments/orsharc/phase2_smoke.sh elephant
```

The setup script installs packages and downloads pinned weights; it does not run
GPU inference. It needs `uv`, network access, disk space under `/root/model`, a
CUDA toolchain compatible with the selected packages, and an adequate GPU.
Elephant's release specifies a PyTorch 2.8 / CUDA 12.8 development image.
Installation on this host has not been tested. Keep environments separate.

The smoke script makes eight train-only predictions twice, runs two optimizer
steps on the 256-example subset, and reloads the adapted checkpoint for eight
predictions. Kev uses 12 steps because its upstream loss log is emitted every
10 steps. StartLux performs inference/reload checks only; its v1.1 training
path is unsupported. No official test examples are used in these smoke commands.

For fallback smoke tests, run the matching setup and smoke scripts:

```bash
bash experiments/orsharc/setup_phase2.sh startlux
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/startlux/bin/python \
  bash experiments/orsharc/phase2_smoke.sh startlux
bash experiments/orsharc/setup_phase2.sh kev
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/kev/bin/python \
  bash experiments/orsharc/phase2_smoke.sh kev
bash experiments/orsharc/setup_phase2.sh qwen
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/qwen/bin/python \
  bash experiments/orsharc/phase2_smoke.sh qwen
```

## Smoke artifacts and acceptance

Each attempt writes `artifacts/smoke/MODEL-UTC/` with:

- `original/` and `reload/`: manifests, incremental predictions, events and
  completion hashes. Eight IDs, full finite option distributions, raw T=1,
  correct option order and positive latency are required.
- `training/`: exact command and source/data/checkpoint hashes, train/dev exports,
  context audit, console and native loss logs, checkpoint and completion metadata.
  No dropped/truncated records, nonfinite loss or unexplained head replacement
  is acceptable. Qwen must report only LoRA trainable parameters.
- `adapted/`: eight predictions from the completed smoke checkpoint.
- `gate.json`: reload agreement with zero changed labels and max absolute
  probability difference ≤0.005; finite training-loss evidence and successful
  adapted reload where training is supported.

`gate.json` intentionally leaves `formal_experiment_admission: false`. Before
formal runs, Luna Max must also record these measured checks in a separate
`admission.json` tied to code/source/recipe/environment hashes:

1. Audit the actual trainable names and optimizer membership before/after real
   steps. Elephant/Qwen must update only LoRA; Kev must retain its native head
   and update only that head plus LoRA. Check frozen backbone tensor hashes.
2. Compare adapted unmerged/merged or saved/reloaded native outputs on the same
   train panel. Set a numerical tolerance before measurement; record differences
   and label agreement. The current smoke proves original reload and adapted
   loadability, not adapted merge parity or frozen-weight integrity.
3. Profile longest real inputs, peak GPU memory, compile/kernel path, records and
   tokens per second, load/merge/save time. Use measured cost to plan all sizes.
   Do not lower context limits or silently drop records to make a run pass.
4. If Qwen resume is enabled, compare its optimizer/scheduler/RNG/sampler state
   and next updates with an uninterrupted train-only reference. Native
   OneJev/Kev LoRA must restart in a new attempt directory instead.

Keep failures and their elapsed cost. An OOM, API mismatch, incompatible adapter
or changed decision head fails that path. A successful load alone does not
validate training, calibration or quantization. Do not add distributed training
until a separate native synchronization gate exists; this pipeline is single GPU.

## Formal commands after admission

Generate the independent plans. B1 and B4 share run paths, so completed artifacts
are reused only when their identities and hashes match.

```bash
for B_MATRIX in b1 b2 b3 b4; do
  python3 experiments/orsharc/run.py matrix --name "$B_MATRIX" --model elephant
done
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  bash experiments/orsharc/artifacts/plans/b1-elephant.sh
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  bash experiments/orsharc/artifacts/plans/b2-elephant.sh
B_PYTHON=/root/projects/decision/experiments/orsharc/artifacts/env/elephant/bin/python \
  bash experiments/orsharc/artifacts/plans/b3-elephant.sh
```

To run B2/B3 with the 4,096-example adapted checkpoint:

```bash
python3 experiments/orsharc/run.py matrix --name b2 --model elephant \
  --checkpoint experiments/orsharc/artifacts/training/elephant-4096/native/merged --out experiments/orsharc/artifacts/plans/adapted
python3 experiments/orsharc/run.py matrix --name b3 --model elephant \
  --checkpoint experiments/orsharc/artifacts/training/elephant-4096/native/merged --out experiments/orsharc/artifacts/plans/adapted
```

Qwen B1 includes zero/four-shot without `--include-optional`. That switch adds
LoRA at all three sizes. Use the Qwen environment to run its generated script.
Calibration and reporting commands are already in every plan. For model choice
or hyperparameter selection, use **only dev_select**. Keep dev_calibrate reserved
for the fixed condition's temperature. Do not select models or thresholds by
their test results. Each predefined condition may have a separate final test report.

## B5 precision gates

```bash
python3 experiments/orsharc/run.py matrix --name b5 --model qwen --include-optional
bash experiments/orsharc/setup_phase2.sh startlux-vllm
python3 experiments/orsharc/run.py matrix --name b5 --model startlux
```

Before running either full script, use `infer --partition train_256 --limit 8`
on the same checkpoint for each precision. Qwen uses `--quantization bf16`,
`bnb8`, and optional `bnb4`. StartLux uses `--engine vllm` for **both** BF16
and FP8; FP8 adds `--quantization fp8`. The native launcher owns and stops its
vLLM processes and records launch settings/logs. Check for surviving server
processes after a host interruption before starting a replacement.

Run `compare-quantization --runs BF16_DIR QUANT_DIR --out comparison.json`.
It must pass checkpoint/input/engine identity checks; record accuracy and
calibration changes after the common gold join. Record observed numerical
differences rather than imposing unmeasured quality claims. Do not switch
backend, checkpoint or evaluation examples across precision conditions.

For StartLux, the BF16 reference engine and vLLM have separate environments.
Use `artifacts/env/startlux-vllm/bin/python` for the entire B5 script. Other
matrices may use the native Transformers reference environment. A failure in
the vLLM bidirectional plugin or FP8 kernels marks FP8 unsupported on that host;
BF16 reference inference and Qwen int8 remain available.

## Risks and fallback order

- First try Elephant native inference and official LoRA. Its source path is
  inspected, but CUDA kernels, merge behavior and memory are unverified.
- Next use StartLux v1.1 Instant for inference. Keep its native masking. Its
  v1.1 adaptation is unavailable; move adaptation to Kev rather than running
  the old v1 trainer on the v1.1 checkpoint.
- Kev is the adaptation fallback. Strict warm-start compatibility may reject
  a changed checkpoint; preserve the error. Do not partially load its adapter/head.
- Qwen uses Unsloth LoRA; its exact Unsloth/TRL Trainer API and save/resume path
  still need runtime evidence. Its zero/four-shot Transformers baselines are
  independently executable. No generic LLM full fine-tuning is allowed.
- Native Elephant/Kev quantization and Unsloth loading of their released heads
  are unsupported. Use matched StartLux vLLM FP8 or Qwen bitsandbytes PTQ.

All formal outputs remain under `experiments/orsharc/artifacts/`. Preserve source
receipts, package freezes, training logs/checkpoints, calibration files and full
predictions. Reports can be regenerated without inference. Phase 1 must stop
before executing any command above that loads a GPU model.
