#!/usr/bin/env python3
"""Resume the queued clean pre-agent infrastructure retries for seed 20261021.

Wait for the seed 20261022 pool and parent orchestrator to exit, then use the
existing queue and five GPU workers. No new matrix cells or retries are added.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
PARENT = Path(__file__).resolve().parent
PROTOCOL = json.loads((PARENT / "PROTOCOL.json").read_text())
SEED2 = ROOT / PROTOCOL["seed_run_directories"]["20261021"]
SEED3 = ROOT / PROTOCOL["seed_run_directories"]["20261022"]
SEED2_CONFIG = ROOT / PROTOCOL["seed_config_files"]["20261021"]
CONFIG = ROOT / "assets/terminal_bench_low_mtp3.json"
SCHEDULER_PATH = ROOT / "src/terminal_formal_scheduler.py"
sys.path.insert(0, str(ROOT / "src"))
import terminal_formal_scheduler as scheduler  # noqa: E402


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def matching_processes(run_dir: Path) -> list[tuple[int, str]]:
    matches = []
    needle = str(run_dir.resolve())
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        except OSError:
            continue
        if "terminal_formal_scheduler.py" in cmd and needle in cmd:
            matches.append((int(proc.name), cmd.strip()))
    return matches


def open_counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT status, COUNT(*) FROM trials GROUP BY status").fetchall()
    con.close()
    counts = dict(rows)
    return {"PENDING": counts.get("PENDING", 0), "RUNNING": counts.get("RUNNING", 0),
            "all_rows": sum(counts.values())}


def wait_for_processes() -> None:
    while True:
        live3 = matching_processes(SEED3)
        live_parent = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = proc.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
            except OSError:
                continue
            if "terminal_formal_multiseed_orchestrator.py" in cmd:
                live_parent.append(int(proc.name))
        if not live3 and not live_parent:
            return
        print(f"waiting for seed3/orchestrator cleanup: seed3={len(live3)} parent={len(live_parent)}",
              flush=True)
        time.sleep(30)


def wait_for_idle_gpus() -> None:
    deadline = time.monotonic() + 1200
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
        used = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if len(used) == 5 and all(value < 500 for value in used):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs did not become idle within 20 minutes: {used} MiB")
        print(f"waiting for five idle GPUs; memory used MiB={used}", flush=True)
        time.sleep(30)


def stop_workers(children: dict[int, subprocess.Popen]) -> None:
    for child in children.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for child in children.values():
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    for child in children.values():
        if child.poll() is None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main() -> int:
    if PROTOCOL["matrix"]["seeds"] != [20261020, 20261021, 20261022]:
        raise RuntimeError("Frozen seed protocol does not match the authorized three-seed design")

    print("waiting for the active seed 20261022 workers and orchestrator to finish", flush=True)
    wait_for_processes()
    seed3_counts = open_counts(SEED3 / "matrix.sqlite")
    if seed3_counts["PENDING"] or seed3_counts["RUNNING"]:
        raise RuntimeError(f"Seed 20261022 queue is not drained: {seed3_counts}")
    wait_for_idle_gpus()

    plan = json.loads((SEED2 / "PLAN.json").read_text())
    if plan.get("seed") != 20261021 or plan.get("planned_slots") != 267:
        raise RuntimeError("Seed 20261021 plan differs from the frozen protocol")
    initial = open_counts(SEED2 / "matrix.sqlite")
    if initial["PENDING"] != 11 or initial["RUNNING"] != 0:
        raise RuntimeError(f"Unexpected seed 20261021 retry queue state: {initial}")
    state = json.loads((SEED2 / "STATE.json").read_text())
    if not state.get("environment_validation_complete"):
        raise RuntimeError("Seed 20261021 environment validation is not complete")
    if plan["config"].get("models") != json.loads(SEED2_CONFIG.read_text()).get("models"):
        raise RuntimeError("Frozen seed config models differ from the saved plan")

    previous_config = CONFIG.read_bytes()
    save(PARENT / "seed_20261021_retry_resume_started.json", {
        "time": time.time(), "queue_before": initial, "seed3_queue": seed3_counts,
        "config_before_sha256": __import__("hashlib").sha256(previous_config).hexdigest(),
        "queued_retry_attempts": 11,
    })
    shutil.copy2(SEED2_CONFIG, CONFIG)
    active_logs = {}
    children = {}
    launch_records = []
    failures = []
    try:
        if json.loads(CONFIG.read_text()).get("seeds") != [20261021]:
            raise RuntimeError("Seed 20261021 frozen config was not installed")
        for gpu in range(5):
            active_logs[gpu] = (SEED2 / f"retry_seed_20261021_gpu{gpu}.log").open("a")
        for gpu in range(5):
            child, record = scheduler.launch_worker(gpu, SEED2, active_logs)
            children[gpu] = child
            launch_records.append(record)
            save(PARENT / "seed_20261021_retry_workers.json", {
                "time": time.time(), "workers": [
                    {"gpu": index, "pid": proc.pid} for index, proc in children.items()],
                "launch_records": launch_records,
            })
            print(f"launched seed 20261021 retry worker gpu={gpu} pid={child.pid}", flush=True)

        last_report = 0.0
        while True:
            counts = open_counts(SEED2 / "matrix.sqlite")
            now = time.time()
            if now - last_report >= 60:
                save(PARENT / "seed_20261021_retry_progress.json", {
                    "time": now, "queue": counts,
                    "workers": {str(gpu): {"pid": child.pid, "returncode": child.poll()}
                                for gpu, child in children.items()},
                })
                print(f"seed 20261021 retry progress: {counts}", flush=True)
                last_report = now
            dead = {gpu: child.returncode for gpu, child in children.items()
                    if child.poll() is not None and child.returncode != 0}
            if dead:
                failures.append({"time": now, "worker_exit_codes": dead, "queue": counts})
                raise RuntimeError(f"Retry worker exited unexpectedly: {dead}; queue={counts}")
            if counts["PENDING"] == 0 and counts["RUNNING"] == 0:
                exit_codes = {gpu: child.wait(timeout=120) for gpu, child in children.items()}
                if any(code != 0 for code in exit_codes.values()):
                    raise RuntimeError(f"Retry workers did not exit cleanly: {exit_codes}")
                break
            time.sleep(10)

        validation = json.loads((SEED2 / "environment_validation.json").read_text())
        queue = scheduler.Queue(SEED2 / "matrix.sqlite", plan["queue_identity"])
        scheduler.finalize(SEED2, plan["config"], plan["tasks"], validation, queue)
        final = json.loads((SEED2 / "FINAL_STATE.json").read_text())
        save(PARENT / "seed_20261021_retry_resume_complete.json", {
            "time": time.time(), "queue_after": open_counts(SEED2 / "matrix.sqlite"),
            "final_state": final, "worker_exit_codes": {
                str(gpu): child.returncode for gpu, child in children.items()},
        })
        print(f"seed 20261021 retry attempts complete: {final['state']}", flush=True)
        return 0
    finally:
        if any(child.poll() is None for child in children.values()):
            stop_workers(children)
        for stream in active_logs.values():
            stream.close()
        CONFIG.write_bytes(previous_config)
        if failures:
            save(PARENT / "seed_20261021_retry_resume_failure.json", {
                "time": time.time(), "failures": failures,
                "queue_after": open_counts(SEED2 / "matrix.sqlite"),
            })


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("retry resume interrupted; worker groups stopped and config restored", flush=True)
        raise
