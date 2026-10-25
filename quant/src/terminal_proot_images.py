"""Pinned official OCI images for the multi-task PRoot backend."""
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
from terminal_bench import ROOT, read, save, sha

CACHE = ROOT/'build/terminal_bench/low_mtp3/images'
DOWNLOADS = CACHE/'blobs'
ACCEPT = ', '.join(['application/vnd.oci.image.index.v1+json',
    'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
    'application/vnd.docker.distribution.manifest.v2+json'])


def cache_blob(path, digest, download, lock_dir=None):
    """Serialize shared OCI-layer writes across parallel image preparation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(lock_dir) if lock_dir else CACHE/'locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir/(digest.replace(':', '-')+'.lock')
    with lock_path.open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            part = path.with_suffix('.part')
            download(part)
            part.replace(path)
        assert 'sha256:'+sha(path) == digest
    return path


def image_info(task):
    info = read(CACHE/(task+'.json'))
    assert info['pinned'] == read(ROOT/'assets/terminal_bench_images.json')[task]
    return info


def prepare_image(task):
    CACHE.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(exist_ok=True)
    if (CACHE/(task+'.json')).exists():
        return image_info(task)
    pinned = read(ROOT/'assets/terminal_bench_images.json')[task]
    repository, digest = pinned['digest'].split('@')
    with urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repository+':pull', timeout=30) as response:
        token = json.load(response)['token']
    def fetch(digest, kind):
        path = DOWNLOADS/digest.replace(':', '-')
        old = ROOT/'build/terminal_bench/proot_pilot/downloads'/path.name
        req = urllib.request.Request(f'https://registry-1.docker.io/v2/{repository}/{kind}/{digest}',
            headers={'Authorization':'Bearer '+token, 'Accept':ACCEPT})
        def download(part):
            part.unlink(missing_ok=True)
            if old.exists():
                part.hardlink_to(old)
                return
            with urllib.request.urlopen(req, timeout=120) as response, part.open('wb') as out:
                shutil.copyfileobj(response,out)
        return cache_blob(path, digest, download)
    manifest_path = fetch(digest,'manifests')
    manifest = read(manifest_path)
    if 'manifests' in manifest:
        chosen = next(m for m in manifest['manifests'] if m.get('platform',{}).get('architecture')=='amd64' and m['platform']['os']=='linux')
        manifest_path = fetch(chosen['digest'],'manifests')
        manifest = read(manifest_path)
    config_path = fetch(manifest['config']['digest'],'blobs')
    config = read(config_path)
    assert config['architecture']=='amd64' and config['os']=='linux'
    layers = [dict(layer,path=str(fetch(layer['digest'],'blobs'))) for layer in manifest['layers']]
    info = dict(pinned=pinned,manifest=str(manifest_path),config_file=str(config_path),config=config,layers=layers,time=time.time())
    save(CACHE/(task+'.json'),info)
    return info


def unpack(root, info):
    """Apply whiteouts before each layer, preserving regular modes and symlinks.

    These pinned official images are trusted; this is not an untrusted OCI unpacker.
    Device nodes are recorded and omitted: this platform does not allow mknod.
    Required /dev entries are later explicit binds, not fabricated devices.
    """
    omissions = []
    for layer in info['layers']:
        path = Path(layer['path']); assert 'sha256:' + sha(path) == layer['digest']
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            for member in members:
                rel = Path(member.name)
                assert not rel.is_absolute() and '..' not in rel.parts
                if rel.name == '.wh..wh..opq':
                    directory = root / rel.parent
                    if directory.exists():
                        for child in directory.iterdir():
                            if child.is_dir() and not child.is_symlink(): shutil.rmtree(child)
                            else: child.unlink()
                elif rel.name.startswith('.wh.'):
                    victim = root / rel.parent / rel.name[4:]
                    if victim.is_dir() and not victim.is_symlink(): shutil.rmtree(victim)
                    else: victim.unlink(missing_ok=True)
            selected = []
            for member in members:
                if Path(member.name).name.startswith('.wh.'): continue
                if member.isdev(): omissions.append(member.name); continue
                selected.append(member)
            for member in selected:
                if Path(member.name) == Path('.'):
                    assert member.isdir(), 'OCI root entry must be a directory'
                    archive.extract(member, root, filter='fully_trusted')
                    continue
                parent = (root / member.name).parent.resolve()
                assert parent.is_relative_to(root.resolve()), ('external archive parent', member.name)
                archive.extract(member, root, filter='fully_trusted')
    return omissions

