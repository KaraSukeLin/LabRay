#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:?Usage: bash scripts/run_representative_pilot.sh NEW_OUTPUT GPU [GPU ...]}
shift
if [ "$#" -eq 0 ]; then
  set -- 0 1 2 3 4
fi
.modelopt-venv/bin/python -u src/run_benchmarks.py \
  --data assets/benchmark/pilot_v1 --output "$output/phase1" --gpus "$@" --long-max-tokens 4096
.modelopt-venv/bin/python -u src/run_benchmarks.py \
  --data assets/benchmark/pilot_if_supplement --output "$output/ifeval_supplement" --gpus "$@"
.modelopt-venv/bin/python src/extend_benchmark_pilot.py merge \
  --phases "$output/phase1" "$output/ifeval_supplement" --output "$output/combined_runs"
.modelopt-venv/bin/python src/benchmark_budget.py --data assets/benchmark/pilot_v2 \
  --runs "$output/combined_runs" --output "$output/analysis" --gpus "$@"
# The 160-member exploratory family counts IFEval once and both MMMU caps.
for cap in 1024 4096; do
  .modelopt-venv/bin/python src/benchmark_analysis.py --runs "$output/combined_runs" \
    --output "$output/paired_$cap" --manifest assets/benchmark/pilot_v2/manifest.json \
    --mmmu-cap "$cap" --family-size 160 --resamples 199999
done
.modelopt-venv/bin/python src/plot_benchmark_pilot.py "$output/analysis"
.modelopt-venv/bin/python src/verify_benchmark_official.py --runs "$output/combined_runs" \
  --include-extensions --output "$output/official_replay"
