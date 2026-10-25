"""Fetch pinned source only. Never install packages or download weights implicitly."""

import subprocess
import urllib.request

from .common import digest, git_revision, read_json, signature, write_json
from .runner import ROOT


def bootstrap(names):
    lock = read_json(ROOT / "configs/sources.json")
    results = {}
    for name in names:
        meta = lock["onejev"] if name == "onejev" else lock["code"][name]
        path = ROOT / "artifacts/upstream" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--no-checkout", "https://github.com/" + meta["repo"] + ".git", str(path)], check=True
            )
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", meta["sha"]], check=True)
        if git_revision(path) != meta["sha"]:
            raise ValueError(f"wrong upstream revision: {path}; preserve it and choose a new source directory")
        receipt_path = path.parent / f"{name}.receipt.json"
        if receipt_path.exists():
            old = read_json(receipt_path)
            actual = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*.py")) if ".git" not in p.parts}
            if signature(actual) != old["python_sha256"]:
                raise ValueError(f"source changed: {path}; do not overwrite an audited runtime")
            results[name] = old
            continue
        if name == "onejev":
            patch = path.parent / "elephant-serve.patch"
            rev = lock["models"]["elephant"]["revision"]
            with urllib.request.urlopen(
                f"https://huggingface.co/baselabs/elephant-1/resolve/{rev}/serve.patch", timeout=120
            ) as response:
                patch.write_bytes(response.read())
            if digest(patch) != lock["elephant_patch_sha256"]:
                raise ValueError("Elephant official serving patch hash differs")
            subprocess.run(["git", "-C", str(path), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(path), "apply", str(patch)], check=True)
        files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*.py")) if ".git" not in p.parts}
        result = {
            "repo": meta["repo"],
            "revision": meta["sha"],
            "python_sha256": signature(files),
            "patch_sha256": lock["elephant_patch_sha256"] if name == "onejev" else None,
        }
        write_json(receipt_path, result)
        results[name] = result
    return results
