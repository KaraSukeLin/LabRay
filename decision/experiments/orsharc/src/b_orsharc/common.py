"""Small file and provenance helpers. No accelerator imports."""

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def owner(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another writer owns {directory}") from exc
        yield


def environment():
    versions = {}
    for name in (
        "torch",
        "triton",
        "flash-linear-attention",
        "fla-core",
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "vllm",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "libraries": versions}


def code_hash():
    return signature({p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def checkpoint_hash(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"checkpoint must be a local, pinned snapshot: {path}")
    files = {
        str(p.relative_to(path)): digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and ".cache" not in p.relative_to(path).parts
    }
    if not files:
        raise ValueError(f"empty checkpoint: {path}")
    return {"path": str(path.resolve()), "sha256": signature(files), "files": files}


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
