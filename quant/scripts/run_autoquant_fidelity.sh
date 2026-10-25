#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:?Usage: bash scripts/run_autoquant_fidelity.sh OUTPUT [GPU]}
gpu=${2:-0}
mkdir "$output"
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
.vllm-venv/bin/python src/extend_fidelity.py --models \
  rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit \
  --gpu "$gpu" --output "$output/new_models"
.vllm-venv/bin/python src/merge_fidelity.py --extensions "$output/new_models" --output "$output/combined"
mapfile -t peers < <(.vllm-venv/bin/python -c 'import json; print("\n".join(json.load(open("assets/fidelity_extension_analysis.json"))["peers"]))')
.vllm-venv/bin/python src/analyze.py --run "$output/combined" --output "$output/analysis" --peers "${peers[@]}"
.vllm-venv/bin/python src/plot_intervals.py "$output/analysis"
