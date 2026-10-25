#!/usr/bin/env bash
# Phase 2 environment and weight preparation; no training or inference.
set -euo pipefail
B_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
B_VARIANT=${1:-elephant}
case "$B_VARIANT" in
  elephant|startlux|startlux-vllm|kev|qwen) ;;
  *) echo 'Use elephant, startlux, startlux-vllm, kev, or qwen' >&2; exit 2 ;;
esac
B_MODEL=${B_VARIANT%-vllm}
B_ENV=${B_ENV:-"$B_ROOT/artifacts/env/$B_VARIANT"}
# Elephant has a measured H100 runtime. Replay it without dependency resolution:
# torch 2.8 metadata pins Triton 3.4, but FLA rejects that version on Hopper.
# Keep this explicit, verified override until the whole runtime is re-gated.
if [[ "$B_VARIANT" == elephant ]]; then
  if [[ ! -f "$B_ROOT/artifacts/upstream/onejev.receipt.json" ]]; then
    python3 "$B_ROOT/run.py" bootstrap --sources onejev
  fi
  if [[ ! -x "$B_ENV/bin/python" ]]; then
    uv venv --python 3.12.14 --seed "$B_ENV"
  fi
  uv pip install --python "$B_ENV/bin/python" --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    -r "$B_ROOT/elephant-runtime.lock"
  "$B_ENV/bin/python" "$B_ROOT/run.py" environment-verify --out "$B_ENV/runtime.json"
  uv pip freeze --python "$B_ENV/bin/python" > "$B_ENV/resolved.freeze.txt"
  if [[ "${B_ENV_ONLY:-0}" == 1 ]]; then exit 0; fi
  "$B_ENV/bin/hf" download baselabs/elephant-1 \
    --revision b3f8a93462fb1051a9de031e94a7ff380e254072 --local-dir /root/model/orsharc-elephant
  exit 0
fi
python3 "$B_ROOT/run.py" bootstrap --sources onejev kev startlux unsloth
uv venv --python 3.12 --seed "$B_ENV"
B_PYTHON="$B_ENV/bin/python"
uv pip install --python "$B_PYTHON" huggingface_hub hf_xet
read -r B_REPO B_REVISION < <(python3 - "$B_ROOT/configs/sources.json" "$B_MODEL" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))['models'][sys.argv[2]]
print(m['repo'],m['revision'])
PY
)
B_MODEL_DIR="/root/model/orsharc-$B_MODEL"
"$B_ENV/bin/hf" download "$B_REPO" --revision "$B_REVISION" --local-dir "$B_MODEL_DIR"
if [[ "$B_VARIANT" == startlux-vllm ]]; then
  uv pip install --python "$B_PYTHON" -e "$B_ROOT/artifacts/upstream/startlux[vllm]"
else
  uv pip install --python "$B_PYTHON" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  case "$B_VARIANT" in
    elephant)
      uv pip install --python "$B_PYTHON" -c "$B_MODEL_DIR/constraints.txt" -e "$B_ROOT/artifacts/upstream/onejev[train]" flash-linear-attention
      uv pip install --python "$B_PYTHON" -c "$B_MODEL_DIR/constraints.txt" --no-build-isolation causal-conv1d
      ;;
    startlux)
      uv pip install --python "$B_PYTHON" -e "$B_ROOT/artifacts/upstream/startlux[transformers]" flash-linear-attention
      uv pip install --python "$B_PYTHON" --no-build-isolation causal-conv1d
      ;;
    kev)
      uv pip install --python "$B_PYTHON" -e "$B_ROOT/artifacts/upstream/kev[serve]"
      ;;
    qwen)
      uv pip install --python "$B_PYTHON" -e "$B_ROOT/artifacts/upstream/unsloth" trl datasets bitsandbytes
      ;;
  esac
fi
uv pip freeze --python "$B_PYTHON" > "$B_ROOT/artifacts/env/$B_VARIANT.freeze.txt"
printf 'Environment: %s\nModel: %s at %s\n' "$B_ENV" "$B_REPO" "$B_REVISION"
printf 'Phase 2 next: B_PYTHON=%q bash %q %q\n' "$B_PYTHON" "$B_ROOT/phase2_smoke.sh" "$B_MODEL"
