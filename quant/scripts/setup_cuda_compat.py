#!/usr/bin/env python3
"""Extract NVIDIA 575 CUDA compatibility libraries locally for this H100 host."""
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import zstandard

root = Path(__file__).resolve().parents[1]
url = ('https://conda.anaconda.org/conda-forge/linux-64/'
       'cuda-compat-impl-575.57.08-h9ab20c4_0.conda')
archive = urllib.request.urlopen(url, timeout=60).read()
with zipfile.ZipFile(io.BytesIO(archive)) as package:
    name = next(n for n in package.namelist() if n.startswith('pkg-'))
    data = zstandard.ZstdDecompressor().decompress(package.read(name))
    with tarfile.open(fileobj=io.BytesIO(data)) as files:
        files.extractall(root / 'build/cuda-compat', filter='data')
print('Archive SHA-256:', hashlib.sha256(archive).hexdigest())
print('Library directory:', root / 'build/cuda-compat/cuda-compat')
