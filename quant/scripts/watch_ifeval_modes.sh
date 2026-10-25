#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: scripts/watch_ifeval_modes.sh RUN_DIR RUN_PID [INTERVAL_SECONDS]}
run_pid=${2:?usage: scripts/watch_ifeval_modes.sh RUN_DIR RUN_PID [INTERVAL_SECONDS]}
interval=${3:-60}
python=.modelopt-venv/bin/python

snapshot() {
  OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 "$python" src/ifeval_modes.py status --output "$run_dir" >/dev/null
  "$python" - "$run_dir/progress.json" >>"$run_dir/progress-history.jsonl" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({k:d[k] for k in ['updated_utc','planned_evaluations','completed_evaluations',
    'completed_paired_questions','gpu_hours','paired_question_ids']} | {'cells':d['cells']}))
PY
}

while kill -0 "$run_pid" 2>/dev/null; do
  snapshot
  sleep "$interval"
done
snapshot
