"""Multi-task exploratory PRoot transport. Shared kernel; not a sandbox.

Retained Popen/receipt and PID-start-time ownership follow the earlier
proot_terminus probe. No upstream code is patched or registered globally.
"""
import asyncio
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from harbor.environments.base import BaseEnvironment, ExecResult
from terminal_proot_images import unpack, image_info

BASE = Path(__file__).resolve().parents[1] / 'build/terminal_bench/proot_pilot'
PROOT = BASE.parent / 'diagnostics/proot_version_ab/downloads/proot'
INSTANCES = []


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


class PRootEnvironment(BaseEnvironment):
    @staticmethod
    def type(): return 'proot-multitask'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = self.trial_paths.trial_dir / 'proot'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.root = self.directory / 'rootfs'
        self.handles = []
        self.identities = {}
        self.short_tmp = None
        self.started = False
        INSTANCES.append(self)
        assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0

    def _validate_definition(self):
        self.image = image_info(self.environment_name)
        assert self.task_env_config.docker_image == self.image['pinned']['tag']
        user = self.image['config']['config'].get('User')
        if user not in (None, '', 'root', '0'):
            raise ValueError(f'Unvalidated image USER: {user}')
        assert hashlib.sha256(PROOT.read_bytes()).hexdigest() == 'db185300a9daabbce86aca7632d8c7fdc4249f6f153171954013b70c5f491ec0'

    async def start(self, force_build=False):
        began = time.monotonic()
        self.root.mkdir()
        omissions = await asyncio.to_thread(unpack, self.root, self.image)
        info = self.image['config']['config']
        assert not info.get('User') or info['User'] in ('root', '0')
        self.cwd = info.get('WorkingDir') or '/'
        self.image_env = dict(v.split('=', 1) for v in info.get('Env', []))
        self.image_env.update(HOME='/root', LANG='C.UTF-8', TERM='xterm-256color', SHELL='/bin/bash')
        for path in ('__harbor', 'logs/agent', 'logs/verifier', 'logs/artifacts', 'dev/pts', 'dev/shm', 'etc', 'app'):
            (self.root / path).mkdir(parents=True, exist_ok=True)
        (self.root / 'dev/shm').chmod(0o1777)
        (self.root / 'usr/local/lib').mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / 'libcurrent_exe.so', self.root / 'usr/local/lib/libpilot-current-exe.so')
        (BASE / 's').mkdir(exist_ok=True)
        self.short_tmp = Path(tempfile.mkdtemp(dir=BASE / 's'))
        self.short_tmp.chmod(0o1777)
        (self.directory / 'short-tmp').symlink_to(self.short_tmp)
        self.host_env = {**self.image_env, 'PROOT_TMP_DIR': str(self.directory)}
        # Own copy, not a bind exposing host /etc. No host home/model/proc/sys.
        shutil.copyfile('/etc/resolv.conf', self.root / 'etc/resolv.conf')
        self.started = True
        write(self.directory / 'start.json', {'image': self.image['pinned'],
              'cwd': self.cwd, 'image_env': self.image_env, 'device_nodes_omitted': omissions,
              'extraction_sec': time.monotonic()-began, 'runtime': str(PROOT), 'runtime_flags': ['-0'], 'resource_enforcement': 'none',
              'differences': ['PRoot shared kernel/network/PID/devices; no per-trial cgroup limits',
                'Own short /tmp bind, explicit terminal devices, copied DNS configuration',
                'Own /dev/shm directory on the workspace filesystem, not Docker tmpfs or host shared memory',
                'Trusted Verifier only: own-executable readlink compatibility via AT_EXECFN/LD_PRELOAD; no host /proc bind',
                'Image CMD is not run as PID 1; retained tracer supervisor owns exec descendants']})

    def guest_path(self, path):
        """Resolve guest symlinks in the guest root for file transfers."""
        parts = list(PurePosixPath(path if path.startswith('/') else self.cwd + '/' + path).parts[1:])
        resolved = []; links = 0
        while parts:
            part = parts.pop(0)
            if part in ('', '.'): continue
            if part == '..':
                if resolved: resolved.pop()
                continue
            candidate_parts = [*resolved, part]
            candidate = (self.short_tmp.joinpath(*candidate_parts[1:]) if candidate_parts[0] == 'tmp'
                         else self.root.joinpath(*candidate_parts))
            if candidate.is_symlink():
                links += 1
                if links > 40: raise RuntimeError('Guest symlink loop')
                target = PurePosixPath(os.readlink(candidate))
                if target.is_absolute(): resolved = []; target = PurePosixPath(*target.parts[1:])
                parts = list(target.parts) + parts
            else: resolved.append(part)
        return self.short_tmp.joinpath(*resolved[1:]) if resolved and resolved[0] == 'tmp' else self.root.joinpath(*resolved)

    async def upload_file(self, source_path, target_path):
        dest = self.guest_path(str(target_path)); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        self.transfer('upload_file', source_path, target_path)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, self.guest_path(str(target_dir)), symlinks=True, dirs_exist_ok=True)
        self.transfer('upload_dir', source_dir, target_dir)

    async def download_file(self, source_path, target_path):
        dest = Path(target_path); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.guest_path(str(source_path)), dest)
        self.transfer('download_file', source_path, target_path)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(self.guest_path(str(source_dir)), target_dir, symlinks=True, dirs_exist_ok=True)
        self.transfer('download_dir', source_dir, target_dir)

    def transfer(self, op, source, target):
        with (self.directory / 'transfers.jsonl').open('a') as out:
            out.write(json.dumps({'operation': op, 'source': str(source), 'target': str(target), 'time': time.time()}) + '\n')

    def processes(self):
        found = []
        tracers = {str(h['process'].pid) for h in self.handles}
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try:
                status = dict(s.split(':', 1) for s in (entry / 'status').read_text().splitlines() if ':' in s)
                ticks = (entry / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                pid = int(entry.name)
                env = (entry / 'environ').read_bytes().split(b'\0')
                try: exe = os.readlink(entry / 'exe')
                except OSError: exe = '[unavailable]'
                if not (self.identities.get(pid) == ticks or status['TracerPid'].strip() in tracers
                        or exe.startswith(str(self.root) + '/') or f'PROOT_TMP_DIR={self.directory}'.encode() in env): continue
                self.identities[pid] = ticks
                found.append({'pid': pid, 'start_ticks': ticks, 'exe': exe,
                              **{k: status[k].strip() for k in ('Name','State','PPid','TracerPid')}})
            except (OSError, ProcessLookupError): pass
        return found

    def save(self, h):
        data = h['meta']; data['proot_return_code'] = h['process'].poll()
        data['command_return_code'] = int(h['receipt'].read_text()) if h['receipt'].exists() else None
        data['elapsed_sec'] = time.monotonic() - h['started']
        write(h['prefix'].with_suffix('.json'), data)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        assert self.started
        assert (user if user is not None else self.default_user) in (None, 0, '0', 'root')
        index = len(self.handles); name = f'exec-{index:04d}'
        prefix = self.directory / name; receipt = self.root / '__harbor' / (name + '.rc')
        merged = self._merge_env(env) or {}
        verifier_compat = command.startswith('(/tests/test.sh) > /logs/verifier/')
        if verifier_compat:
            merged = {**merged, 'LD_PRELOAD': '/usr/local/lib/libpilot-current-exe.so'}
        inner = ['/usr/bin/env', '-u', 'PROOT_TMP_DIR', *[f'{k}={v}' for k,v in merged.items()], '/bin/bash', '-c', command]
        wrapper = shlex.join(inner) + f'; rc=$?; printf "%s\\n" "$rc" > /__harbor/{receipt.name}.part; mv /__harbor/{receipt.name}.part /__harbor/{receipt.name}; exit "$rc"'
        argv = [str(PROOT), '-0', '-r', str(self.root), '-w', cwd or self.cwd,
                '-b', str(self.short_tmp) + ':/tmp']
        for device in ('null','pts','ptmx','tty','urandom','random'):
            argv += ['-b', f'/dev/{device}:/dev/{device}']
        argv += ['/bin/bash','-c',wrapper]
        with prefix.with_suffix('.stdout').open('wb') as out, prefix.with_suffix('.stderr').open('wb') as err:
            process = subprocess.Popen(argv, env=self.host_env, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, start_new_session=True)
        h = {'process': process, 'prefix': prefix, 'receipt': receipt, 'started': time.monotonic(),
             'meta': {'command': command, 'argv': argv, 'cwd': cwd or self.cwd, 'env': merged, 'pid': process.pid,
                      'timeout_sec': timeout_sec, 'effective_timeout_sec': timeout_sec if timeout_sec is not None else 900,
                      'user': user, 'trusted_verifier_own_exe_shim': verifier_compat, 'started_at': time.time()}}
        self.handles.append(h); self.save(h)
        try:
            async with asyncio.timeout(timeout_sec if timeout_sec is not None else 900):
                while not receipt.exists():
                    if process.poll() is not None: raise RuntimeError(f'PRoot exited before receipt: {prefix}')
                    await asyncio.sleep(0.01)
            h['meta']['receipt_sec'] = time.monotonic()-h['started']
            self.processes(); self.save(h)
            return ExecResult(return_code=int(receipt.read_text()), stdout=prefix.with_suffix('.stdout').read_text(errors='replace'),
                              stderr=prefix.with_suffix('.stderr').read_text(errors='replace'))
        except (TimeoutError, asyncio.CancelledError):
            h['meta']['cancelled_or_timed_out'] = True
            await self.kill_exec(h)
            self.save(h)
            raise

    @staticmethod
    def send(process, sig):
        try:
            ticks = (Path('/proc') / str(process['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            if ticks == process['start_ticks']: os.kill(process['pid'], sig)
        except (ProcessLookupError, FileNotFoundError): pass

    def reap(self):
        for h in self.handles: h['process'].poll()
        for pid in list(self.identities):
            try: os.waitpid(pid, os.WNOHANG)
            except ChildProcessError: pass

    async def kill_exec(self, h):
        owned = [p for p in self.processes() if p['TracerPid'] == str(h['process'].pid) or p['pid'] == h['process'].pid]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for process in owned: self.send(process, sig)
            await asyncio.sleep(0.15); self.reap()
        if h['process'].poll() is None:
            h['process'].kill(); await asyncio.to_thread(h['process'].wait, 2)

    async def stop(self, delete=True):
        # Preserve rootfs and logs regardless of delete: exploratory evidence.
        before = self.processes()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for process in self.processes(): self.send(process, sig)
            await asyncio.sleep(0.15); self.reap()
        for h in self.handles:
            if h['process'].poll() is None: await self.kill_exec(h)
            self.save(h)
        self.reap(); after = self.processes()
        write(self.directory / 'cleanup.json', {'before': before, 'after': after, 'delete_requested': delete,
             'rootfs_retained_for_evidence': True, 'child_cpu': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime})
        self.started = False
        if after: raise RuntimeError(f'Owned processes remain: {after}')
