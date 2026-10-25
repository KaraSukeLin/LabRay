from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
from threading import Lock
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from terminal_proot_images import cache_blob


def test_shared_oci_blob_is_downloaded_once_under_parallel_image_preparation(tmp_path):
    content = b'pinned OCI layer bytes\n'
    digest = 'sha256:'+hashlib.sha256(content).hexdigest()
    target = tmp_path/'downloads'/digest.replace(':', '-')
    locks = tmp_path/'locks'
    calls = 0
    calls_lock = Lock()

    def download(part):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(.02)
        part.write_bytes(content)

    def fetch(_):
        return cache_blob(target, digest, download, locks)

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(fetch, range(8)))

    assert calls == 1
    assert all(path == target for path in paths)
    assert target.read_bytes() == content
