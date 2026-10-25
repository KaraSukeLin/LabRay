#!/usr/bin/env bash
set -euo pipefail
cd /root/projects/decision
B_SMOKE=experiments/orsharc/artifacts/smoke
B_PHASE3="$B_SMOKE/phase3-20261023"
python3 experiments/orsharc/run.py smoke-verify \
  --original "$B_SMOKE/elephant-20261023T065538Z/original" \
  --reload "$B_SMOKE/elephant-20261023T065538Z/reload" \
  --training "$B_PHASE3/production-training" \
  --adapted "$B_PHASE3/cli-inference" \
  --admission-evidence "$B_PHASE3/evidence-index.json" \
  --out "$B_PHASE3/admission.json"
