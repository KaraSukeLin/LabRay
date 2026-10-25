#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:?Usage: scripts/run_expanded.sh OUTPUT [GPU ...]}
shift
if [ "$#" -eq 0 ]; then set -- 0; fi
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
.vllm-venv/bin/python src/fidelity.py run --all \
  --tokens assets/expanded_v2/tokens.json --logprob-format dense \
  --gpus "$@" --output "$output"
.vllm-venv/bin/python src/analyze.py --run "$output" --output "$output/analysis"
.vllm-venv/bin/python src/plot_intervals.py "$output/analysis"
