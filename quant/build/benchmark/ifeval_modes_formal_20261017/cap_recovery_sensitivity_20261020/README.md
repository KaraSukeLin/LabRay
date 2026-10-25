# Recovered-cap sensitivity run

This is a derived, quality-only dataset. It replaces exactly three primary 65K-truncated rows with their saved 100K-cap replay outputs. Unchanged model files are symlinks to the formal run; for the NaviQuant and PrismaAQUA files, every unchanged JSONL row is copied as-is. All three replacements have identical prompt token IDs. The primary run and its score files are unchanged.

The analysis uses the existing `src/analyze_ifeval_modes.py` and `src/ifeval_modes.py` pipeline, with 199,999 paired question-cluster bootstrap resamples and the predeclared 300-comparison family. No model inference ran for this derived dataset. Its timing and allocated-GPU-hour outputs are not execution-cost estimates; use the formal run and retry logs for actual costs.

Recreate the analysis from the stored derived rows with:

```sh
.vllm-venv/bin/python src/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/runs \
  --output build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/analysis \
  --resamples 199999 --family-size 300
```
