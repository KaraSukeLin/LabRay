#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: run_ifeval_paired_after_off.sh RUN_DIR OFF_SUPERVISOR_PID}
off_supervisor_pid=${2:?usage: run_ifeval_paired_after_off.sh RUN_DIR OFF_SUPERVISOR_PID}
python=.modelopt-venv/bin/python
log="$run_dir/paired-after-off-supervisor.log"
exec >>"$log" 2>&1

refresh_status() {
  OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 "$python" src/ifeval_modes.py status --output "$run_dir" >/dev/null
}

off_is_complete() {
  "$python" - "$run_dir/progress.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
c=[x for x in d['cells'] if x['mode']=='off']
raise SystemExit(0 if len(c)==15 and all(x['completed']==x['target'] for x in c) else 1)
PY
}

run_stage() {
  local name=$1
  shift
  "$@" >"$run_dir/$name.log" 2>&1 &
  local run_pid=$!
  OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 bash scripts/watch_ifeval_modes.sh "$run_dir" "$run_pid" 60 \
    >"$run_dir/$name-watch.log" 2>&1 &
  local watch_pid=$!
  local code=0
  wait "$run_pid" || code=$?
  wait "$watch_pid" || true
  return "$code"
}

echo "Waiting for Non-thinking supervisor PID $off_supervisor_pid to finish."
while kill -0 "$off_supervisor_pid" 2>/dev/null; do sleep 30; done
refresh_status

if ! off_is_complete; then
  echo "Non-thinking matrix is incomplete; retrying pending records once."
  run_stage off-resume env OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    "$python" -u src/ifeval_modes.py run --full --resume --output "$run_dir" --gpus 0 1 2 3 4 \
    --task-modes off --task-seeds 20261016 20261017 20261018 || true
  refresh_status
fi

if ! off_is_complete; then
  echo "Not starting paired Thinking continuation: Non-thinking records remain incomplete."
  exit 0
fi

read -r seconds gpu_hours < <("$python" - "$run_dir/progress.json" <<'PY'
import json, math, sys
from datetime import datetime, timezone
d=json.load(open(sys.argv[1]))
cutoff=datetime.fromisoformat('2026-10-18T22:00:00+00:00')
remaining=max(0, math.floor((cutoff-datetime.now(timezone.utc)).total_seconds())-120)
gpu_left=max(0., 150.-float(d['gpu_hours'])-1.)
print(min(remaining, math.floor(gpu_left*3600/5)), round(float(d['gpu_hours']), 2))
PY
)

if (( seconds < 120 )); then
  echo "Not starting paired Thinking continuation: only ${seconds}s remain in budget window."
  exit 0
fi

"$python" - "$run_dir" "$seconds" "$gpu_hours" <<'PY'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
p=pathlib.Path(sys.argv[1]); e=json.loads((p/'experiment.json').read_text())
(p/'paired-after-off-plan.json').write_text(json.dumps({
  'created_utc':datetime.now(timezone.utc).isoformat(),
  'models':e['models'], 'modes':['on'], 'seeds':[20261017,20261018],
  'question_count':len(e['question_ids']),
  'question_ids_sha256':hashlib.sha256(json.dumps(e['question_ids'],separators=(',',':')).encode()).hexdigest(),
  'max_run_seconds':int(sys.argv[2]), 'gpu_hours_before_start':float(sys.argv[3]),
  'sampling':e['config']['modes']['on'], 'mtp':e['config']['mtp'],
  'kv_cache_dtype':e['config']['kv_cache_dtype'],
}, indent=2)+'\n')
PY

echo "Running paired Thinking seeds 20261017/20261018 on the full fixed question order for up to ${seconds}s; prior GPU-hours=${gpu_hours}."
run_stage paired-on-extension timeout --signal=INT --kill-after=120 "${seconds}s" env \
  OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  "$python" -u src/ifeval_modes.py run --full --resume --output "$run_dir" --gpus 0 1 2 3 4 \
  --task-modes on --task-seeds 20261017 20261018 || true
refresh_status
echo "Paired Thinking continuation ended at $(date -u +%FT%TZ)."
