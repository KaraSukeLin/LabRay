#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:?Usage: bash scripts/smoke_benchmarks.sh GPU OUTPUT [MODEL ...]}
output=${2:?Missing output directory}
shift 2
if [ "$#" -eq 0 ]; then
  mapfile -t models < <(.vllm-venv/bin/python -c 'import json; print("\n".join(json.load(open("assets/benchmark_config.json"))["models"]))')
else
  models=("$@")
fi
mkdir -p "$output"
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
for model in "${models[@]}"; do
  .modelopt-venv/bin/python -u src/benchmark.py run --model "$model" \
    --output "$output/$model" --limit 2 --max-tokens 128 > "$output/$model.log" 2>&1
done
