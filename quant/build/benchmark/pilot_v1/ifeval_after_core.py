"""Wait for the core pilot, then run only its IFEval supplement."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "build/benchmark/pilot_v1"
CORE_LAUNCH = RUN / "runs/launch.json"
OUT = RUN / "ifeval_supplement_runs"

print("Waiting for all nine core jobs to finish.", flush=True)
while not CORE_LAUNCH.exists():
    time.sleep(5)

core = json.loads(CORE_LAUNCH.read_text())
assert len(core["jobs"]) == 9 and all(job["exit_code"] == 0 for job in core["jobs"]), core
assert not OUT.exists(), f"Refusing to overwrite {OUT}"

command = [
    sys.executable, "-u", "src/run_benchmarks.py",
    "--data", "assets/benchmark/pilot_if_supplement",
    "--output", str(OUT), "--gpus", "0", "1", "2", "3", "4",
]
print("Starting IFEval supplement on five GPUs.", flush=True)
subprocess.run(command, cwd=ROOT, check=True)

supplement = json.loads((OUT / "launch.json").read_text())
assert len(supplement["jobs"]) == 9 and all(job["exit_code"] == 0 for job in supplement["jobs"]), supplement
(RUN / "ifeval_supplement_complete.json").write_text(json.dumps({
    "models": 9,
    "cases_per_model": 40,
    "total_model_question_cases": 360,
    "all_exit_codes_zero": True,
    "eight_k_stage_started": False,
    "final_report_started": False,
    "pause_requested_by_user": True,
}, indent=2) + "\n")
print("IFEval supplement complete; stopped before analysis and final report.", flush=True)
