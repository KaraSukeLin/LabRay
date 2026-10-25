# Fidelity experiment

Keep code in `src/`, fixed text data in `assets/`, targeted tests in `tests/`, and experiment artifacts in `build/`.
Model weights remain in `/root/model/`.

Run targeted CPU checks with:

```sh
.vllm-venv/bin/python -m pytest -q tests/test_fidelity.py
```

Run the user-requested GPU experiment with:

```sh
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 \
  .vllm-venv/bin/python src/fidelity.py run --all --output build/comparison
```

Run only tests relevant to the change. Do not run a full test suite.

Expanded evaluation uses the fixed dataset in `assets/expanded_v2/` and
`assets/expanded_config.json`. Run it with:

```sh
scripts/run_expanded.sh build/new_run 0 1 2 3 4
```

Use one GPU number for a sequential run. The script runs the independent BF16
sanity gate before candidates, then analysis and plots. Keep MTP off and KV cache
BF16. `src/prepare_dataset.py` rebuilds the public-source dataset;
`src/analyze.py` uses SciPy to resample whole documents within domain/context
strata. Run the relevant CPU checks with:

```sh
.vllm-venv/bin/python -m pytest -q tests/test_fidelity.py tests/test_analysis.py
```

Preserve first-round artifacts in `build/final/` and expanded artifacts in
`build/expanded/`. Do not mix preliminary or interrupted runs into final tables.

Automatic-quantization extensions live in `build/autoquant/`. Reuse the unchanged
`src/fidelity.py` and the original full reference with `src/extend_fidelity.py`;
run a fresh full-data BF16 gate before new candidates. The formal nine-model list
is `assets/benchmark_config.json`; it excludes the original RadixArk NVFP4.
`assets/fidelity_extension_analysis.json` fixes six Fast peers and 18 primary
comparisons. Run the prepared checkpoints with:

```sh
bash scripts/run_autoquant_fidelity.sh build/new_autoquant_run 0
```

Use `bash scripts/setup_research.sh` for the isolated ModelOpt/benchmark overlay.
Keep the validated `.vllm-venv` unchanged. The overlay must process the base
environment's `.pth` files with `site.addsitedir`, so spawned workers find CUTLASS.
Benchmark smoke tests use only two questions per task and 128 output tokens:

```sh
bash scripts/smoke_benchmarks.sh 0 build/new_benchmark_smoke
```

Do not start a full IFEval/MMMU-Pro run without a new user instruction.

The 2026-10-17 official-source recheck is `build/benchmark/OFFICIAL_REVIEW.md`.
Formal inputs use `assets/benchmark/prepared_v2/`: canonical Google repository
IFEval input (the Hub copy differs on key 2785) and official MMMU-Pro. Preserve
old prepared inputs and smoke artifacts. Default sampling follows Qwen's
non-thinking recommendation, and image limits match its published processor.
`src/benchmark.py run --mtp` enables native MTP; omission means Off.
For both modes and all model/mode effects, analysis supports a predeclared
300-comparison family. MTP startup and positive acceptance counts do not prove
token equality or unchanged capability; preserve the observed differences.
