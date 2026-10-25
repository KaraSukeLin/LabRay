import datetime, json, os, pathlib, signal, time
root = pathlib.Path('/root/projects/quant')
run_dir = root / 'build/benchmark/ifeval_modes_formal_20261017'
runner_pid = 49163
deadline = datetime.datetime.fromisoformat('2026-10-18T22:43:08.469383+00:00')
status_path = run_dir / 'deadline-guard-30h.json'
def runner_active():
    try: cmd = pathlib.Path(f'/proc/{runner_pid}/cmdline').read_bytes()
    except (OSError, PermissionError): return False
    return b'src/ifeval_modes.py\x00run' in cmd and b'ifeval_modes_formal_20261017' in cmd
if not runner_active(): raise SystemExit(f'Expected active runner PID {runner_pid}')
print(json.dumps({'state':'armed','runner_pid':runner_pid,'deadline_utc':deadline.isoformat()}), flush=True)
while True:
    remaining = (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining <= 0: break
    if not runner_active():
        print(json.dumps({'state':'runner_exited_before_deadline','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}), flush=True)
        raise SystemExit(0)
    time.sleep(min(60, remaining))
active = runner_active()
if active: os.kill(runner_pid, signal.SIGINT)
result = {'state':'deadline_reached_interrupt_sent' if active else 'deadline_reached_no_active_runner','runner_pid':runner_pid,'deadline_utc':deadline.isoformat(),'written_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
status_path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
