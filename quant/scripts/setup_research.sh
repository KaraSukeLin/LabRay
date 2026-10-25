#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.vllm-venv/bin/python src/prepare_research_tools.py
if [ ! -d .modelopt-venv ]; then
  uv venv --python .vllm-venv/bin/python --system-site-packages .modelopt-venv
fi
.vllm-venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig
Path('.modelopt-venv/lib/python3.12/site-packages/fidelity_base.pth').write_text(
    'import site; site.addsitedir('+repr(sysconfig.get_paths()['purelib'])+')\n')
PY
uv pip install --python .modelopt-venv/bin/python --no-deps ./build/autoquant/tools/Model-Optimizer
uv pip install --python .modelopt-venv/bin/python --no-deps -r requirements-research.txt
.modelopt-venv/bin/python - <<'PY'
import nltk
nltk.download('punkt_tab',download_dir='assets/benchmark/nltk_data',raise_on_error=True)
PY
