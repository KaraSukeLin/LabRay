#!/usr/bin/env python3
"""Resume final analysis and Fidelity after the seed-20261021 retries finish."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
PARENT = Path(__file__).resolve().parent
SEED2 = ROOT / "build/terminal_bench/low_mtp3/formal_20261021_seed_20261021"
SEED3 = ROOT / "build/terminal_bench/low_mtp3/formal_20261021_seed_20261022"
RETRY_DONE = PARENT / "seed_20261021_retry_resume_complete.json"
RETRY_FAIL = PARENT / "seed_20261021_retry_resume_failure.json"
ORCHESTRATOR = ROOT / "src/terminal_formal_multiseed_orchestrator.py"


def matching_processes(needle: str, script: str | None = None) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        except OSError:
            continue
        if needle in cmd and (script is None or script in cmd):
            found.append(int(proc.name))
    return found


def wait_for_retry_resume() -> None:
    while not RETRY_DONE.exists():
        if RETRY_FAIL.exists():
            raise RuntimeError(f"Seed retry helper reported failure: {RETRY_FAIL}")
        if not matching_processes("seed_20261021_retry_resume.py"):
            raise RuntimeError("Seed retry helper exited without a completion record")
        print("waiting for seed 20261021 retry workers", flush=True)
        time.sleep(30)


def wait_for_idle_controllers() -> None:
    while True:
        seed3 = matching_processes(str(SEED3), "terminal_formal_scheduler.py")
        parent = matching_processes("terminal_formal_multiseed_orchestrator.py")
        retry = matching_processes("seed_20261021_retry_resume.py")
        if not seed3 and not parent and not retry:
            return
        print(f"waiting for controller cleanup: seed3={seed3} parent={parent} retry={retry}",
              flush=True)
        time.sleep(15)


def main() -> int:
    print("post-retry finalizer waiting for the retry queue", flush=True)
    wait_for_retry_resume()
    wait_for_idle_controllers()

    seed2_analysis = SEED2 / "analysis"
    if seed2_analysis.exists():
        backup = SEED2 / f"analysis_before_retry_{time.time_ns()}"
        seed2_analysis.rename(backup)
        print(f"preserved stale seed analysis at {backup}", flush=True)

    aggregate = PARENT / "analysis"
    if aggregate.exists():
        backup = PARENT / f"analysis_before_retry_resume_{time.time_ns()}"
        aggregate.rename(backup)
        print(f"preserved prior aggregate at {backup}", flush=True)

    log = PARENT / "post_retry_orchestrator.log"
    argv = [str(ROOT / ".harbor-venv/bin/python"), "-u", str(ORCHESTRATOR)]
    with log.open("a") as stream:
        code = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"Recovered orchestrator exited {code}; see {log}")

    analysis = json.loads((aggregate / "analysis.json").read_text())
    fidelity = json.loads((PARENT / "agentic_fidelity/agentic_summary.json").read_text())
    if analysis.get("observed_slots") != 801 or not analysis.get("all_matrix_slots_terminal"):
        raise RuntimeError("Final analysis does not prove all 801 slots are terminal")
    final_state = json.loads((PARENT / "FINAL_STATE.json").read_text())
    save = PARENT / "POST_RETRY_FINALIZATION.json"
    save.write_text(json.dumps({
        "state": "COMPLETE", "updated_at": time.time(),
        "analysis": str(aggregate), "observed_slots": analysis["observed_slots"],
        "all_matrix_slots_terminal": analysis["all_matrix_slots_terminal"],
        "agentic_fidelity": str(PARENT / "agentic_fidelity"),
        "fidelity_result_count": len(fidelity.get("results", [])),
        "final_state": final_state,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"final analysis and Fidelity complete: {aggregate}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        (PARENT / "POST_RETRY_FINALIZATION_FAILURE.json").write_text(json.dumps({
            "time": time.time(), "error": f"{type(exc).__name__}: {exc}",
        }, indent=2) + "\n")
        raise
