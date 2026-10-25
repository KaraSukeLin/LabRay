#!/usr/bin/env bash
# Phase 2 only. Run after installing the model's separate runtime environment.
set -euo pipefail
B_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
B_MODEL=${1:-elephant}
B_PYTHON=${B_PYTHON:-python}
B_ATTEMPT=${B_ATTEMPT:-$(date -u +%Y%m%dT%H%M%SZ)}
B_OUT="$B_ROOT/artifacts/smoke/$B_MODEL-$B_ATTEMPT"
mkdir -p "$B_OUT"
"$B_PYTHON" "$B_ROOT/run.py" check --prepared "$B_ROOT/artifacts/prepared"
"$B_PYTHON" "$B_ROOT/run.py" infer --model "$B_MODEL" --partition train_256 --limit 8 --out "$B_OUT/original"
"$B_PYTHON" "$B_ROOT/run.py" infer --model "$B_MODEL" --partition train_256 --limit 8 --out "$B_OUT/reload"
if [[ "$B_MODEL" == startlux ]]; then
  "$B_PYTHON" "$B_ROOT/run.py" smoke-verify --original "$B_OUT/original" --reload "$B_OUT/reload" --out "$B_OUT/gate.json"
  exit 0
fi
B_STEPS=2
[[ "$B_MODEL" != kev ]] || B_STEPS=12
"$B_PYTHON" "$B_ROOT/run.py" train --model "$B_MODEL" --size 256 --out "$B_OUT/training" --max-steps "$B_STEPS" --execute
case "$B_MODEL" in
  elephant) B_CHECKPOINT="$B_OUT/training/native/merged" ;;
  kev) B_CHECKPOINT="$B_OUT/training/native" ;;
  qwen) B_CHECKPOINT="$B_OUT/training/merged" ;;
  *) exit 2 ;;
esac
"$B_PYTHON" "$B_ROOT/run.py" infer --model "$B_MODEL" --checkpoint "$B_CHECKPOINT" --partition train_256 --limit 8 --out "$B_OUT/adapted"
"$B_PYTHON" "$B_ROOT/run.py" smoke-verify --original "$B_OUT/original" --reload "$B_OUT/reload" --training "$B_OUT/training" --adapted "$B_OUT/adapted" --out "$B_OUT/gate.json"
