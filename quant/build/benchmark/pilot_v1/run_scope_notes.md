# Pilot run scope and decoding note

Recorded 2026-10-17 after the user's clarification.

## Scope

- Finish the current core pilot: 9 models × 160 questions = 1,440 question-model cases.
- After the core run succeeds, run only the separate IFEval supplement: 40 additional questions per model, 360 question-model cases total, scheduled across five GPUs.
- Do not launch the 8,192-token follow-up stage. The original background launchers were stopped; a new IFEval-only controller waits for core completion.
- After IFEval completes, pause. Do not prepare the round's final report until the user gives the next instruction.

## Decoding configuration actually used

The active jobs read `assets/benchmark_config.json`: `enable_thinking=false`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`, `repetition_penalty=1`, `seed=20261016`, `mtp=false`, and BF16 KV cache. The 4,096-token probes keep these settings and change the output cap. This is Qwen3.8's documented non-thinking sampling recipe; it is not greedy. Greedy has a separate config and separate smoke records.

`REPORT.md` section “下一階段 IFEval／MMMU-Pro” still says the formal run is greedy. That statement conflicts with the active run and the 2026-10-17 official review; correct it when the final report is authorized. Do not merge results from the greedy checks with this sampling run.

## Live count snapshot

At the time this note was added, the core runner had written 1,295 of the planned 1,440 main rows. It had also written 418 extension rows. Of the main MMMU rows, 376 ended at the 1,024-token cap. Of the completed 4,096-token extension rows, 151 still ended with `finish_reason=length` and are candidates for an 8,192-token check. These are live counts, not final totals; no 8,192-token generation was started.

The next read-only refresh found 1,296 main rows, 418 extension rows, and 377 main MMMU rows ending at 1,024. The 151 4,096-token rows still ending with `length` span 39 distinct question IDs. No 8,192-token follow-up is authorized.

## Updated user instruction

The user then authorized the IFEval supplement after the core run, followed by a pause for further direction. This does not authorize the 8,192-token follow-up, analysis, or final report.

## Final completion and report authorization

Core and IFEval supplement finished successfully: 1,800 main answers and 501 saved 4,096-token probes across nine models, all 18 worker processes exit 0. There were 451 primary MMMU truncations at 1,024 tokens and 187 remaining at 4,096, across 40 question IDs. No 8,192-token or full benchmark inference was run.

The user's later instruction explicitly authorized an integrated analysis report and time budget for model selection. The reporting deadline is Monday 2026-10-19 at 09:00 Asia/Taipei; the user accepts later data updates. This authorizes CPU analysis and report preparation. It does not request a new full GPU run. REPORT.md now states the actual non-greedy settings; the earlier conflicting sentence has been corrected.

## Unexpected retained-controller stage, stopped

The final check found old controller PID9718 started an out-of-scope8192 stage after the authorized merge. It was stopped along with17 descendants; all five GPUs were confirmed free. The17 partial answers and logs are quarantined under interrupted_unexpected_8192, excluded from all formal tables and budgets. Approximate extra allocation2.24GPU-h is reported separately. Thus the earlier statement that no8192 generation occurred is superseded; no8192 experiment was completed, and no full benchmark started. See decision_report/unexpected_8192_stop.json.
