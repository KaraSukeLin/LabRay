#!/usr/bin/env bash
set -euo pipefail

B5_ROOT=/root/projects/decision
B5_EVIDENCE="$B5_ROOT/experiments/orsharc/artifacts/smoke/phase4-preflight-20261023T1355Z"
B5_ADAPTER="$B5_EVIDENCE/startlux_llamacpp_runner.py"
B5_CONFIG="$B5_EVIDENCE/startlux27b-q4_k_m.json"
B5_PREPARED="$B5_ROOT/experiments/orsharc/artifacts/prepared"
B5_RUN="$B5_ROOT/experiments/orsharc/artifacts/runs/phase4/startlux27b-original-single_choice-q4_k_m-0shot-llamacpp"
B5_PYTHON=${B5_PYTHON:-$B5_ROOT/experiments/orsharc/artifacts/env/elephant-repro/bin/python}

"$B5_PYTHON" "$B5_ADAPTER" infer --model-config "$B5_CONFIG" --prepared "$B5_PREPARED" --out "$B5_EVIDENCE/b5-startlux-train2" --partition train_256 --task benchmark --strategy single_choice --shots 0 --limit 2
"$B5_PYTHON" "$B5_ADAPTER" infer --model-config "$B5_CONFIG" --prepared "$B5_PREPARED" --out "$B5_RUN/dev_select" --partition dev_select --task benchmark --strategy single_choice --shots 0
"$B5_PYTHON" "$B5_ADAPTER" infer --model-config "$B5_CONFIG" --prepared "$B5_PREPARED" --out "$B5_RUN/dev_calibrate" --partition dev_calibrate --task benchmark --strategy single_choice --shots 0
"$B5_PYTHON" "$B5_ROOT/experiments/orsharc/run.py" calibrate --prepared "$B5_PREPARED" --select-run "$B5_RUN/dev_select" --calibration-run "$B5_RUN/dev_calibrate" --out "$B5_RUN/calibration.json"
"$B5_PYTHON" "$B5_ADAPTER" infer --model-config "$B5_CONFIG" --prepared "$B5_PREPARED" --out "$B5_RUN/test" --partition test --task benchmark --strategy single_choice --shots 0
"$B5_PYTHON" "$B5_ROOT/experiments/orsharc/run.py" report --prepared "$B5_PREPARED" --run "$B5_RUN/dev_select" --calibration "$B5_RUN/calibration.json" --out "$B5_RUN/reports/dev_select"
"$B5_PYTHON" "$B5_ROOT/experiments/orsharc/run.py" report --prepared "$B5_PREPARED" --run "$B5_RUN/dev_calibrate" --calibration "$B5_RUN/calibration.json" --out "$B5_RUN/reports/dev_calibrate"
"$B5_PYTHON" "$B5_ROOT/experiments/orsharc/run.py" report --prepared "$B5_PREPARED" --run "$B5_RUN/test" --calibration "$B5_RUN/calibration.json" --out "$B5_RUN/reports/test"
