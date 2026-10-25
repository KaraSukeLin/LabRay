# Terminal-Bench 2.1 three-seed formal analysis

- Matrix slots: 801/801.
- Environment decisions: 89/89; gate counts `{"CONTROLS_PASS": 60, "ORACLE_CONTROL_FAIL": 28, "SCORING_GATE_FAIL": 1}`.
- All matrix slots terminal: True; all Gate-passed slots have a Trial attempt: True.
- Common fully scored tasks across all models and all three seeds: 53/60 Gate-passed tasks.
- Five-GPU allocated time: 144.58 GPU-hours; sum of seed-run wall time: 29.78 hours; allocated fraction: 0.9709544473592178.

| Model | Seed pass rates | Three-seed mean | Seed SD | All-scored rate | Mean retention vs BF16 | Mean seed retention | Paired difference | Scored slots | GPU hours | Reported cost USD | Tokens in/out | Tools | MTP accepted/draft |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 20261020: 0.472, 20261021: 0.604, 20261022: 0.528 | 0.5346 [0.4151, 0.6541] | 0.06626197328838199 | 0.550561797752809 | 1.0 | 1.0 | 0.0 | 178 | 46.27 | NA | 26117897/3565319 | 4166 | 4104692.0/5800602.0 |
| NaviQuant | 20261020: 0.415, 20261021: 0.566, 20261022: 0.491 | 0.4906 [0.3711, 0.6101] | 0.07547169811320753 | 0.5141242937853108 | 0.9176470588235295 | 0.9153571428571429 | -0.04402515723270434 | 177 | 49.11 | NA | 23028313/3238634 | 3930 | 3537767.0/5044116.0 |
| Aqua | 20261020: 0.528, 20261021: 0.453, 20261022: 0.566 | 0.5157 [0.3962, 0.6352] | 0.05764246157177156 | 0.5449438202247191 | 0.9647058823529412 | 0.9804761904761904 | -0.018867924528301883 | 178 | 49.20 | NA | 25809971/3366349 | 4186 | 3695345.0/5192592.0 |
USD cost is `NA` when any attempt has no reported amount; GPU-hours and token counts remain measured resource usage.

## Status and failure distribution

Each row covers 89 task slots for one model and seed. Shared Gate failures appear for all models and seeds; trial-level verifier and environment failures have no score.

| Seed | Model | ENVIRONMENT_FAIL | MODEL_FAIL | ORACLE_CONTROL_FAIL | PASS | SCORING_GATE_FAIL | TIMEOUT | VERIFIER_FAIL |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20261020 | Qwen-Qwen3.8-27B | 0 | 11 | 28 | 28 | 1 | 20 | 1 |
| 20261021 | Qwen-Qwen3.8-27B | 0 | 5 | 28 | 36 | 1 | 19 | 0 |
| 20261022 | Qwen-Qwen3.8-27B | 0 | 9 | 28 | 30 | 1 | 20 | 1 |
| 20261020 | eNeural-Qwen3.8-27B-fast | 1 | 12 | 28 | 26 | 1 | 20 | 1 |
| 20261021 | eNeural-Qwen3.8-27B-fast | 1 | 8 | 28 | 35 | 1 | 16 | 0 |
| 20261022 | eNeural-Qwen3.8-27B-fast | 0 | 9 | 28 | 27 | 1 | 24 | 0 |
| 20261020 | rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | 0 | 7 | 28 | 31 | 1 | 21 | 1 |
| 20261021 | rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | 0 | 6 | 28 | 29 | 1 | 25 | 0 |
| 20261022 | rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | 0 | 7 | 28 | 35 | 1 | 17 | 1 |

## Paired uncertainty

Intervals resample whole tasks and keep each task’s three seeds together while holding the fixed Seed list. Accuracy difference, harmful flips, and recovery flips use the six-comparison family adjustment fixed in the protocol. Retention has a 95% task-cluster bootstrap interval. Seed-to-seed spread is reported in `model_summary.csv`.

| Candidate | Measure | Estimate | 95% interval | Family-adjusted interval |
|---|---|---:|---:|---:|
| eNeural-Qwen3.8-27B-fast | accuracy_delta | -0.0440 | [-0.1006, 0.0126] | [-0.1195, 0.0377] |
| eNeural-Qwen3.8-27B-fast | harmful_flip | 0.1006 | [0.0566, 0.1509] | [0.0440, 0.1698] |
| eNeural-Qwen3.8-27B-fast | recovery_flip | 0.0566 | [0.0252, 0.0943] | [0.0126, 0.1132] |
| eNeural-Qwen3.8-27B-fast | retention ratio | 0.9176 | [0.8111, 1.0274] | NA |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | accuracy_delta | -0.0189 | [-0.0881, 0.0503] | [-0.1069, 0.0818] |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | harmful_flip | 0.0943 | [0.0503, 0.1447] | [0.0377, 0.1635] |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | recovery_flip | 0.0755 | [0.0314, 0.1321] | [0.0189, 0.1572] |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | retention ratio | 0.9647 | [0.8427, 1.1059] | NA |

## Paired task differences

See `paired_task_seed_differences.csv` for each task and seed, including the exact BF16 reward, candidate reward, score delta, harmful flip, and recovery flip.

## Status and audit files

Runner audit: live-release simulation 570 s vs batch-barrier 600 s; same-model service batch 3; 21 targeted regressions passed. The simulation is scheduler evidence, not measured GPU speed; see `RUNNER_AUDIT.json`.
Slot outcomes across the 801-cell ledger: `{"TIMEOUT": 182, "ORACLE_CONTROL_FAIL": 252, "MODEL_FAIL": 74, "PASS": 277, "VERIFIER_FAIL": 5, "ENVIRONMENT_FAIL": 2, "SCORING_GATE_FAIL": 9}`.
Integrity checks passed for all seeds: True. See `integrity_audit.json` for row-level issues.
Infrastructure, verifier, and PRoot failures remain explicit in `matrix_801.csv`; they have no score and are excluded from paired ability estimates. `attempts.csv` preserves every attempt, including a retry. `paired_intervals.csv` gives the uncertainty intervals.

This is an internal comparison in the shared PRoot environment, not an official Docker leaderboard result.

## Agentic target-model Fidelity

- Prepared from first-seed Low BF16 trajectories; 59 tasks, 234 windows, 7488 scored positions.
- This teacher-forced full-vocabulary result is a Fidelity check, not a benchmark score.

```json
[
  {
    "model": "Qwen-Qwen3.8-27B",
    "stratum": "all",
    "tasks": 59,
    "scoring_positions": 7488,
    "task_equal_weight": {
      "ppl": 1.2209724435005438,
      "reference_ppl": 1.2209724435005438,
      "ppl_delta": 0.0,
      "kl_bf16_to_candidate": 0.0,
      "top1_flip_rate": 0.0
    },
    "task_cluster_95ci": {
      "ppl": [
        1.1917999337945329,
        1.2526088983111159
      ],
      "ppl_delta": [
        0.0,
        0.0
      ],
      "delta_nll": [
        0.0,
        0.0
      ],
      "ppl_ratio": [
        1.0,
        1.0
      ],
      "kl_bf16_to_candidate": [
        0.0,
        0.0
      ],
      "top1_flip_rate": [
        0.0,
        0.0
      ]
    }
  },
  {
    "model": "eNeural-Qwen3.8-27B-fast",
    "stratum": "all",
    "tasks": 59,
    "scoring_positions": 7488,
    "task_equal_weight": {
      "ppl": 1.2328448147962061,
      "reference_ppl": 1.2209724435005438,
      "ppl_delta": 0.01187237129566232,
      "kl_bf16_to_candidate": 0.01049958395809095,
      "top1_flip_rate": 0.024540960451977404
    },
    "task_cluster_95ci": {
      "ppl": [
        1.2028140475931495,
        1.2656075944947378
      ],
      "ppl_delta": [
        0.007968278645607752,
        0.01588238676670886
      ],
      "delta_nll": [
        0.006530194770593027,
        0.01286535305232226
      ],
      "ppl_ratio": [
        1.0065515629801958,
        1.0129484677577818
      ],
      "kl_bf16_to_candidate": [
        0.009237489783180857,
        0.011907679334444305
      ],
      "top1_flip_rate": [
        0.020789194915254237,
        0.02842514124293785
      ]
    }
  },
  {
    "model": "rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm",
    "stratum": "all",
    "tasks": 59,
    "scoring_positions": 7488,
    "task_equal_weight": {
      "ppl": 1.236822790429625,
      "reference_ppl": 1.2209724435005438,
      "ppl_delta": 0.01585034692908116,
      "kl_bf16_to_candidate": 0.011010673455004815,
      "top1_flip_rate": 0.02484992937853107
    },
    "task_cluster_95ci": {
      "ppl": [
        1.2062142902074746,
        1.2699016657884454
      ],
      "ppl_delta": [
        0.012001034071150973,
        0.019763667651191683
      ],
      "delta_nll": [
        0.009834976595723332,
        0.015946529016199158
      ],
      "ppl_ratio": [
        1.0098834989195578,
        1.0160743534582817
      ],
      "kl_bf16_to_candidate": [
        0.00962319597895231,
        0.012426119126335115
      ],
      "top1_flip_rate": [
        0.019906426553672314,
        0.030146539548022596
      ]
    }
  }
]
```
