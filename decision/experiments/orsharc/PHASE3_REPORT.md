# Phase 3 — Experiment B admission

2026-10-23 UTC. Targeted debugging and verification only. **Elephant BF16 B1–B4: PASS.**
The authoritative [admission.json](artifacts/smoke/phase3-20261023/admission.json)
extends the existing `smoke-verify` command. The Phase 2 gate is retained as historical evidence.
No formal matrix, full-budget training, full-dev calibration, research ablation, or official test inference ran.

| Capability | Status | Evidence / limit |
|---|---|---|
| B1–B4 Elephant | PASS | All six required admission checks passed; see admission artifact and checks below |
| Qwen LoRA | BLOCKED | Companion installed; import FAIL; tiny training/reload NOT TESTED |
| B5 Elephant quantization | BLOCKED | No validated supported path in the pinned release/runtime; BF16 B1–B4 remain available |
| B5 Qwen BF16/BNB8 | PASS | Retained Phase 2 four-row smoke only; no formal accuracy claim |
| Complete dev calibration / threshold selection | NOT TESTED | Complete 456/649-row guard retained; Phase 4 work |
| Official final test evaluation | NOT TESTED | No Phase 3 model call used test inputs |

## Changes and root causes

- `elephant-runtime.lock`, `setup_phase2.sh`, `gates.py`, `cli.py`: replay the 69-package successful runtime with `--no-deps`, Python 3.12.14, Torch 2.8.0/CUDA 12.8, Triton 3.7.1, Transformers 5.19.0, FLA 0.5.2 and PEFT 0.21.2. Torch metadata requires Triton 3.4.0, which the prior H100 FLA backward rejected. This deliberate override is explicit; `environment-verify` checks every locked version and all other active dependency requirements. A normal `uv pip check` still reports the single accepted Torch/Triton conflict. No claim of resolver-level compatibility is made.
- Preserve the Phase 2 PyPI extra-index workaround in its **successful index order**. The reverse order either hides newer pinned packages or selects NVIDIA-hosted wheels that fail TLS. Fresh replay, repeated replay, and the original environment all passed. Repeat setup reported `Checked 69 packages`; it did not change installed versions. `B_ENV_ONLY=1` skips model download. Environments remain separate. The optional convolution build no longer prevents Elephant setup.
- `common.py`: include Triton and FLA versions in run provenance.
- `phase3.py`, `gates.py`, `cli.py`: add bounded measurements to the existing smoke gate. The native trainer, loss, heads, optimizer, hyperparameters, inference backend and merged-checkpoint route remain in use. No upstream source was changed.
- `data.py`, `runner.py`: derive `ft_exposed` separately for budgets 256/1024/4096 and save it in predictions. Official `snippet_seen`, rule splits, source data, prepared files and partition hashes are unchanged. State rendering uses the existing explicit allowlist. `answer_label` handles Yes/No case-insensitively and rejects empty, Irrelevant, Unknown and Maybe classification values.
- `training.py`: add `ft_exposure.json` to completed checkpoints, recording budget, partition hash and rule IDs for a complete one-epoch run. A partial smoke checkpoint has `complete_budget: false` and `rule_ids: null`; its actual exposure must not be inferred from a nominal budget.
- `configs/models.json`, `training.py`, `matrix.py`: mark only Qwen training unavailable. Direct training fails clearly; generated plans list the blocked FT cells and retain zero/four-shot inference.
- Documentation: this report, `PHASE4.md`, and current-status links in the existing handoff/readme.

## Measured integrity and parity

[measure-2](artifacts/smoke/phase3-20261023/measure-2/) used 24 distinct real **train** examples, including representative upper-tail lengths and the train maximum. It ran three unchanged native optimizer steps. Dev examples were used only for predictions.

- Optimizer membership exactly matched all **496 LoRA tensors / 43,278,336 parameters**, with no duplicates or frozen tensors. All 496 had finite, nonzero gradients and actual nonzero updates. Maximum per-tensor changes ranged from 5.21e-6 to 3.00e-5. All **427 frozen parameter hashes** stayed identical.
- All three losses were finite: 0.90573, 1.19489, 0.52287. Saved adapter tensors matched active tensors exactly, including names. Adapter rank/alpha and readout metadata matched the registered recipe. See `optimizer.json`, `integrity.json`, `native/args.json`, and `native/log.jsonl`.
- Ten identical train/dev inputs covered p50/p90/p95/p99/max for each split. Full-forward active vs reloaded adapter: maximum absolute probability difference **0.0**, no label changes; tolerance 1e-6. Active vs BF16 merged/reloaded: **0.016395**, no label changes; premeasurement tolerance **0.02**, relative tolerance 0. Complete distributions are in `active.jsonl`, `reloaded.jsonl`, `merged.jsonl` and `parity.json`.
- Production cached serving with its FP32 LM head: maximum difference **0.041476**, no label changes. Final tolerance is **0.05**, relative tolerance 0, following the pinned `qev/engine.py` BF16 cached/batched rounding guidance. The initial 0.02 serving check failed and is retained in `serving-strict-002.json`. This is a numerical limit, not a claim of exact production probability equality. The native full-forward check retains its stricter 0.02 tolerance.
- Diagnostic FP32 merge/reference after correct metadata copying: maximum probability difference **0.000185**, no label changes. The original exploratory 1e-4 threshold was too strict; this diagnostic is not used to admit a new FP32 formal path.

The first parity verifier copied the source's `readout.json` alone. The source's noncausal attention is stored in `config.json`; that copy reloaded with causal attention and caused false differences up to 0.233. The verifier now copies the **adapter's complete readout metadata**, as the existing production wrapper already did. A gate check verifies the copy. All failed arrays, logs, weights and probes remain available, including `missing-attention-*`, `fp32-probe*`, `reload-probe*` and `correct-parity*`. Provisional adapter-routing changes were removed. BF16 weight rounding is real, but the corrected existing merge passes the declared native tolerance.

## Lengths, memory and execution budget

The released tokenizer and exact native renderer measured all 17,936 train and 1,105 dev inputs, without reading dev gold for the length or prediction pass. See `lengths.json`.

| Split | p50 | p90 | p95 | p99 | Max tokens |
|---|---:|---:|---:|---:|---:|
| Train | 257 | 329 | 355 | 405 | 494 |
| Dev | 247 | 294 | 318 | 376 | 439 |

The 16,384-token limit is unchanged. No record was dropped or truncated. Production usage counts matched full rendered token counts on all panel rows; a real request with a deliberately insufficient limit was rejected before execution.

Native train steps took 4.33 / 2.75 / 3.00 seconds for eight examples each. The last two measured **2.66–2.91 examples/s**, **941–1,034 tokens/s**, and **17.69 GiB peak allocated VRAM**. The native training portion took 10.08 seconds; the 108.04-second instrumented procedure also included full hashes, save, parity loads and merge. Side-by-side verification loads peaked at about 34.2 GiB. Adapter save took 0.36 seconds. The final production-wrapper three-step attempt, including preflight, eight-row before/after dev diagnostics, merge/save and hashes, took **46.90 seconds**. Native console timestamps and shard-write progress retain merge/save timings.

Merged production inference averaged **97.5 ms/request**, at most **105.8 ms** on the panel, about **10.3 examples/s / 3,604 tokens/s**. Load took **4.23 seconds**, peak allocated memory **22.39 GiB**. Adapter comparison was slower (151.5 ms/request); formal inference keeps the existing merged path. Keep the PyTorch convolution fallback. No CUDA toolkit or convolution-extension installation was pursued.

Planning estimates for one H100, one registered epoch, based conservatively on measured examples/s rather than assuming perfect token scaling:

| Budget | Exact rendered train tokens | Estimated train compute | Reserve including dev checks, load, save and hashes |
|---|---:|---:|---:|
| 256 | 68,273 | 1.5–1.6 min | 4–6 min |
| 1,024 | 272,682 | 5.9–6.4 min | 9–12 min |
| 4,096 | 1,084,553 | 23.4–25.6 min | 27–35 min |

Reserve another 1–2 minutes for cold kernel compilation on a fresh cache. The measured panel is an upper-tail sample, not a production timing guarantee. Budget about 2–3 minutes per complete dev pass and 4–6 minutes per future single-choice test pass. The latter is an estimate; test inference was not run. Hierarchy uses two calls and parallel Noul has three branches: reserve 2–3 times single-choice cost. A practical initial reserve for the primary B1–B4 work is **2–4 H100 hours**, including training, multiple formats, reporting and interruptions; adapted B2/B3 additions can increase it. B4 reuses B1 exactly. Do not run both as independent new grids.

[resource.json](artifacts/smoke/phase3-20261023/resource.json) records a conservative continuous-GPU-0 reservation upper bound from Phase 3 artifact creation through verification. This is below one GPU-hour; it is not an integrated utilization counter. No GPU work is needed after this admission.

## Labels and generalization

[metadata.json](artifacts/smoke/phase3-20261023/metadata.json) checks every pinned raw answer against sealed gold, every official Seen label, and every budget's rule membership. No Irrelevant/Unknown/Maybe category occurs. There are 29 distinct train follow-ups without question marks (208 rows), 4 dev follow-ups (4 rows), and 12 test follow-ups (14 rows). They remain traceable verbatim in the audit and map to Inquire under the documented official decision task. Test inspection here was static data validation only.

Official Seen means a rule occurs in the **complete official train split**. In dev, 250 / 76 / 4 officially Seen rows are not exposed by the 256 / 1,024 / 4,096 subsets. Both meanings remain available in saved outputs. Per-budget `ft_exposed` describes full nested budgets; it does not assert exposure by a partial smoke checkpoint. The CLI report preserves these fields in `scored_raw.jsonl`. Inference's sealed-file guard passed. Prepared data and partitions still match the original manifest hashes.

## Optional limits

`unsloth-zoo==2026.10.7`, the pinned Unsloth source's minimum companion, installed without changing existing package versions. Import then stopped on missing `requests`. Its requirements also constrain Transformers to <=5.17.0 and Datasets to <5, conflicting with preserved Qwen 5.20.2/5.1.0. The constrained resolver confirmed no solution. No downgrade, custom trainer or speculative retry followed. Training/reload remains NOT TESTED; Qwen LoRA is BLOCKED. Install/import/resolver logs and the separate `qwen.phase3.freeze.txt` are retained. Existing BF16/BNB8 inference evidence remains separate.

The pinned Elephant release and OneJev engine have no validated compatible quantized path in this pipeline. B5 Elephant remains BLOCKED; existing rejection behavior is preserved. No unrelated prequantized model was substituted.

## Reproduction and handoff

From `/root/projects/decision`:

```bash
B_ENV_ONLY=1 B_ENV="$PWD/experiments/orsharc/artifacts/env/elephant-repro" bash experiments/orsharc/setup_phase2.sh elephant
B_PYTHON="$PWD/experiments/orsharc/artifacts/env/elephant-repro/bin/python"
# New measurement attempts require a NEW directory.
"$B_PYTHON" experiments/orsharc/run.py phase3-measure --out experiments/orsharc/artifacts/smoke/NEW-ATTEMPT
"$B_PYTHON" experiments/orsharc/run.py phase3-serving --measurement experiments/orsharc/artifacts/smoke/NEW-ATTEMPT
python3 experiments/orsharc/run.py metadata-audit --out experiments/orsharc/artifacts/smoke/NEW-METADATA.json
```

The executed measurement is `measure-2`; its corrected parity-only replay is `correct-parity.log` and `correct-parity.py`. Final serving is `serving-final.log`. The real wrapper command used `train --model elephant --size 256 --max-steps 3 --execute --out .../production-training`. The exact inference/isolation/repeat/report command is saved in `cli-check.py`, with `cli-check.log`. No output directory from an interrupted or changed attempt was reused for training.

`admission-command.sh` reruns the existing smoke verifier against the preserved evidence, including the new production training/checkpoint and eight-row CLI run. `admission.log` records the successful result. Targeted Ruff and shell checks passed. `plan-checks.json` verifies identical B1/B4 commands, calibration-before-test ordering, optional Qwen blocking and retained zero/four-shot commands; generated matrices were not executed.

**Luna Max:** use [PHASE4.md](PHASE4.md) for exact execution order, commands and safeguards. Phase 4 has not started. Preserve this evidence and all Phase 2 artifacts.
