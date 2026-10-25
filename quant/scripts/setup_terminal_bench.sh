#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, subprocess
from pathlib import Path
c=json.loads(Path('assets/terminal_bench_config.json').read_text())
for key,name in [('harbor','harbor'),('dataset','terminal-bench-2-1')]:
    p=Path('build/terminal_bench/sources')/name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone',c[key]['repository'],str(p)],check=True)
        subprocess.run(['git','-C',str(p),'checkout','--detach',c[key]['revision']],check=True)
    assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==c[key]['revision']
    assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=no'],text=True)
PY
UV_PROJECT_ENVIRONMENT="$PWD/.harbor-venv" uv sync --frozen --no-dev --no-default-groups \
  --project build/terminal_bench/sources/harbor --python "$PWD/.vllm-venv/bin/python"
uv pip freeze --python .harbor-venv/bin/python > build/terminal_bench/harbor.freeze.txt
.modelopt-venv/bin/python src/terminal_bench.py prepare
.modelopt-venv/bin/python src/terminal_bench.py resolve-images
