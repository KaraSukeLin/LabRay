#!/usr/bin/env python3
"""Dynamically run the already-queued seed-20261021 infrastructure retries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PARENT = Path(__file__).resolve().parent
SEED1 = ROOT / "build/terminal_bench/low_mtp3/formal_20261021"
SEED2 = ROOT / "build/terminal_bench/low_mtp3/formal_20261021_seed_20261021"
SEED3 = ROOT / "build/terminal_bench/low_mtp3/formal_20261021_seed_20261022"
SEED2_CONFIG = ROOT / "build/terminal_bench/low_mtp3/formal_three_seed_20261022/configs/config_seed_20261021.json"
HARBOR_PY = ROOT / ".harbor-venv/bin/python"
sys.path.insert(0, str(ROOT / "src"))
import terminal_formal_scheduler as scheduler


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def counts(run: Path) -> dict[str, int]:
    con = sqlite3.connect(run / "matrix.sqlite")
    rows = con.execute("SELECT status, COUNT(*) FROM trials GROUP BY status").fetchall()
    con.close()
    out = dict(rows)
    out["PENDING"] = out.get("PENDING", 0)
    out["RUNNING"] = out.get("RUNNING", 0)
    out["all_rows"] = sum(value for key, value in out.items() if key not in {"all_rows"})
    return out


def latest_rows(run: Path) -> list[dict]:
    con = sqlite3.connect(run / "matrix.sqlite")
    con.row_factory = sqlite3.Row
    latest = {}
    for raw in con.execute("SELECT task, model, attempt, started, status, gpu FROM trials"):
        row = dict(raw)
        key = (row["task"], row["model"])
        prior = latest.get(key)
        if prior is None or (row["attempt"], row["started"] or 0) > (prior["attempt"], prior["started"] or 0):
            latest[key] = row
    con.close()
    return list(latest.values())


def gpu_memory() -> dict[int, int]:
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    return {int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
            for line in proc.stdout.splitlines() if line.strip()}


def port_busy(gpu: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        return sock.connect_ex(("127.0.0.1", 18100 + gpu)) == 0
    finally:
        sock.close()


def seed3_pending() -> int:
    return counts(SEED3)["PENDING"]


def available_gpu(gpu: int) -> bool:
    if seed3_pending():
        return False
    if any(row["status"] == "RUNNING" and row.get("gpu") == gpu for row in latest_rows(SEED3)):
        return False
    for run in (SEED1, SEED2):
        if any(row["status"] == "RUNNING" and row.get("gpu") == gpu for row in latest_rows(run)):
            return False
    try:
        if gpu_memory().get(gpu, 999999) >= 500 or port_busy(gpu):
            return False
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return True


def worker_main(gpu: int) -> int:
    os.environ["OPENAI_API_KEY"] = "local-only"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    scheduler.CONFIG = SEED2_CONFIG
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    scheduler.worker(gpu, str(SEED2))
    return 0


def launch_worker(gpu: int, logs: dict[int, object]) -> tuple[subprocess.Popen, dict]:
    env = dict(os.environ, OPENAI_API_KEY="local-only", LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONDONTWRITEBYTECODE="1")
    argv = [str(HARBOR_PY), str(Path(__file__).resolve()), "--worker", str(gpu)]
    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=logs[gpu], stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except OSError:
        ticks = None
    return child, {"gpu": gpu, "pid": child.pid, "start_ticks": ticks, "command": argv, "started_at": time.time()}


def matching_processes(needle: str) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        except OSError:
            continue
        if needle in cmd:
            found.append(int(proc.name))
    return found


def wait_for_parent_and_seed3() -> None:
    last_report = 0.0
    while True:
        scheduler_processes = []
        parent_processes = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = proc.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
            except OSError:
                continue
            if "terminal_formal_scheduler.py" in cmd and str(SEED3) in cmd:
                scheduler_processes.append(int(proc.name))
            if "terminal_formal_multiseed_orchestrator.py" in cmd:
                parent_processes.append(int(proc.name))
        if not scheduler_processes and not parent_processes:
            return
        now = time.time()
        if now - last_report >= 60:
            print(f"retry queue terminal; waiting to finalize seed2 after seed3/parent cleanup: "
                  f"seed3={scheduler_processes} parent={parent_processes}", flush=True)
            last_report = now
        time.sleep(10)


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


def controller_main() -> int:
    protocol = json.loads((PARENT / "PROTOCOL.json").read_text())
    plan = json.loads((SEED2 / "PLAN.json").read_text())
    state = json.loads((SEED2 / "STATE.json").read_text())
    if protocol["matrix"]["seeds"] != [20261020, 20261021, 20261022]:
        raise RuntimeError("Frozen three-seed protocol mismatch")
    if plan.get("seed") != 20261021 or plan.get("planned_slots") != 267:
        raise RuntimeError("Seed 20261021 plan mismatch")
    if not state.get("environment_validation_complete"):
        raise RuntimeError("Seed 20261021 environment validation is not complete")
    if plan["config"].get("models") != json.loads(SEED2_CONFIG.read_text()).get("models"):
        raise RuntimeError("Seed 20261021 frozen config model list differs from its saved plan")
    before = counts(SEED2)
    if before["PENDING"] != 11 or before["RUNNING"] != 0:
        raise RuntimeError(f"Unexpected retry queue before dynamic dispatch: {before}")
    seed3 = counts(SEED3)
    if seed3["PENDING"]:
        raise RuntimeError(f"Seed 20261022 still has queued tasks; do not take its workers: {seed3}")

    logs: dict[int, object] = {}
    children: dict[int, subprocess.Popen] = {}
    records = []
    started = time.time()
    start_artifact = PARENT / "seed_20261021_retry_resume_started.json"
    save(start_artifact, {"time": started, "queue_before": before, "seed3_queue": seed3,
        "dispatch_policy": "dynamic free-GPU dispatch; seed2 frozen config loaded per worker",
        "retry_attempts": 11, "config_path": str(SEED2_CONFIG)})
    failures = []
    last_report = 0.0
    idle_deadline = time.monotonic() + 1200
    try:
        while True:
            if seed3_pending():
                raise RuntimeError("Seed 20261022 gained pending work; pausing cross-seed GPU dispatch")
            for gpu in range(5):
                if gpu in children or not available_gpu(gpu):
                    continue
                if gpu not in logs:
                    logs[gpu] = (SEED2 / f"retry_seed_20261021_gpu{gpu}.log").open("a")
                child, record = launch_worker(gpu, logs)
                children[gpu] = child
                records.append(record)
                save(PARENT / "seed_20261021_retry_workers.json", {
                    "time": time.time(), "workers": records,
                    "seed2_queue": counts(SEED2), "seed3_running_gpus": sorted(
                        row["gpu"] for row in latest_rows(SEED3) if row["status"] == "RUNNING"),
                })
                print(f"launched retry worker gpu={gpu} pid={child.pid}", flush=True)
            queue = counts(SEED2)
            now = time.time()
            if now - last_report >= 60:
                save(PARENT / "seed_20261021_retry_progress.json", {
                    "time": now, "queue": queue,
                    "launched_gpus": sorted(children),
                    "workers": {str(gpu): {"pid": child.pid, "returncode": child.poll()}
                                for gpu, child in children.items()},
                    "seed3_running_gpus": sorted(
                        row["gpu"] for row in latest_rows(SEED3) if row["status"] == "RUNNING"),
                    "gpu_memory_mib": gpu_memory(),
                })
                print(f"seed 20261021 dynamic retry progress: {queue}; GPUs={sorted(children)}", flush=True)
                last_report = now
            dead = {gpu: child.returncode for gpu, child in children.items()
                    if child.poll() is not None and child.returncode != 0}
            if dead and (queue["PENDING"] or queue["RUNNING"]):
                failures.append({"time": now, "worker_exit_codes": dead, "queue": queue})
                raise RuntimeError(f"Retry worker exited unexpectedly: {dead}; queue={queue}")
            if queue["PENDING"] == 0 and queue["RUNNING"] == 0:
                for child in children.values():
                    if child.poll() is None:
                        child.wait(timeout=120)
                exit_codes = {gpu: child.returncode for gpu, child in children.items()}
                if any(code != 0 for code in exit_codes.values()):
                    raise RuntimeError(f"Retry workers did not exit cleanly: {exit_codes}")
                break
            if not children and time.monotonic() >= idle_deadline:
                raise RuntimeError("No safe GPU became available for retries within 20 minutes")
            time.sleep(10)

        wait_for_parent_and_seed3()
        validation = json.loads((SEED2 / "environment_validation.json").read_text())
        queue_obj = scheduler.Queue(SEED2 / "matrix.sqlite", plan["queue_identity"])
        scheduler.finalize(SEED2, plan["config"], plan["tasks"], validation, queue_obj)
        final_state = json.loads((SEED2 / "FINAL_STATE.json").read_text())
        final_counts = counts(SEED2)
        if final_counts["PENDING"] or final_counts["RUNNING"]:
            raise RuntimeError(f"Seed2 finalization left open trial attempts: {final_counts}")
        save(PARENT / "seed_20261021_retry_resume_complete.json", {
            "time": time.time(), "queue_after": final_counts, "final_state": final_state,
            "worker_exit_codes": {str(gpu): child.returncode for gpu, child in children.items()},
            "workers": records,
        })
        print(f"seed 20261021 retry attempts terminal: {final_state['state']}", flush=True)
        return 0
    finally:
        if any(child.poll() is None for child in children.values()):
            stop_workers(children)
        for stream in logs.values():
            stream.close()
        if failures:
            save(PARENT / "seed_20261021_retry_resume_failure.json", {
                "time": time.time(), "failures": failures, "queue_after": counts(SEED2),
                "workers": records,
            })


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return worker_main(int(sys.argv[2]))
    return controller_main()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("retry resume interrupted; child worker groups are being cleaned", flush=True)
        raise
