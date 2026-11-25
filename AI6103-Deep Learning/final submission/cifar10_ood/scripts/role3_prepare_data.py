"""Prepare Task E data without competing with an existing prepare_data.py.

Run via role3_prepare_data.sh on TC2. Manual archives download concurrently;
torchvision preparation waits for the older process in this exact project.
This script never kills that process. A successful exit requires all datasets
to load with the expected counts, and every file-backed image to decode.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile


IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
EXPECTED = {'CIFAR10_train': 50000, 'CIFAR10_test': 10000, 'CIFAR100': 10000,
            'SVHN': 26032, 'MNIST': 10000, 'DTD': 1880,
            'TIN': 10000, 'PLACES365': 36500}


def say(message):
    print(time.strftime('%F %T'), message, flush=True)


def image_paths(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT)


def verify_images(paths, expected):
    from PIL import Image
    if len(paths) != expected:
        raise ValueError(f'Expected {expected} images, found {len(paths)}')
    for path in paths:
        with Image.open(path) as image:
            image.convert('RGB').load()


def target_path(root, name):
    """Reject archive paths outside the intended extraction directory."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe archive member: {name!r}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Archive target escapes extraction directory: {name!r}')
    return target


def download(project, url, archive, name):
    log = project / 'logs' / f'role3_download_{name}.log'
    say(f'{name}: downloading/resuming {archive.name}; log={log}')
    with log.open('ab') as output:
        subprocess.run(['wget', '--timeout=45', '--tries=3', '--waitretry=5',
                        '--progress=dot:giga', '-c', '-O', str(archive), url],
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def prepare_manual(project, name):
    data = project / 'data'
    if name == 'TIN':
        folder = data / 'tiny-imagenet-200' / 'test' / 'images'
        archive = project / 'tiny-imagenet-200.zip'
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    else:
        folder = data / 'places365'
        archive = project / 'val_256.tar'
        url = 'http://data.csail.mit.edu/places/places365/val_256.tar'
    try:
        verify_images(image_paths(folder), EXPECTED[name])
        say(f'{name}: existing images verified')
        return
    except (OSError, ValueError) as exc:
        say(f'{name}: preparation needed ({exc})')
    download(project, url, archive, name)
    if name == 'TIN':
        with zipfile.ZipFile(archive) as source:
            corrupt = source.testzip()
            if corrupt:
                raise ValueError(f'TIN ZIP checksum failure: {corrupt}')
            members = [m for m in source.infolist()
                       if m.filename.startswith('tiny-imagenet-200/test/images/')
                       and not m.is_dir()]
            if len(members) != EXPECTED[name]:
                raise ValueError(f'TIN archive contains {len(members)} test images')
            for member in members:
                target = target_path(data, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as source:
            for member in source:
                target = target_path(folder, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError(f'Unsupported Places archive member: {member.name!r}')
    verify_images(image_paths(folder), EXPECTED[name])
    say(f'{name}: {EXPECTED[name]} images extracted and decoded')


def older_download_pids(project):
    """Only this user's Python processes running this project's old entry point."""
    expected = (project / 'prepare_data.py').resolve()
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if not argv or not Path(argv[0]).name.startswith('python'):
                continue
            for arg in argv[1:]:
                if Path(arg).name != 'prepare_data.py':
                    continue
                path = Path(arg)
                if not path.is_absolute():
                    path = (proc / 'cwd').resolve() / path
                if path.resolve() == expected:
                    found.append(int(proc.name))
                    break
        except (OSError, ProcessLookupError):
            continue
    return sorted(found)


def wait_for_existing(project, max_seconds):
    start = time.monotonic()
    next_log = 0
    while True:
        pids = older_download_pids(project)
        if not pids:
            return
        elapsed = time.monotonic() - start
        if elapsed >= max_seconds:
            raise TimeoutError(f'Existing prepare_data.py still active: {pids}; left running')
        if elapsed >= next_log:
            say(f'Waiting for existing prepare_data.py PIDs {pids}; no competing writes')
            next_log = elapsed + 300
        time.sleep(15)


def torchvision_datasets(data, download):
    from torchvision import datasets
    return [
        ('CIFAR10_train', lambda: datasets.CIFAR10(str(data), train=True, download=download)),
        ('CIFAR10_test', lambda: datasets.CIFAR10(str(data), train=False, download=download)),
        ('CIFAR100', lambda: datasets.CIFAR100(str(data), train=False, download=download)),
        ('SVHN', lambda: datasets.SVHN(str(data / 'svhn'), split='test', download=download)),
        ('MNIST', lambda: datasets.MNIST(str(data), train=False, download=download)),
        ('DTD', lambda: datasets.DTD(str(data), split='test', partition=1, download=download)),
    ]


def prepare_torchvision(project, max_wait):
    wait_for_existing(project, max_wait)
    # A child keeps dataset allocations/imports out of the concurrent workers.
    subprocess.run([sys.executable, '-u', __file__, '--project', str(project),
                    '--torchvision-worker'], check=True)


def torchvision_worker(project):
    socket.setdefaulttimeout(60)
    for name, factory in torchvision_datasets(project / 'data', True):
        for attempt in range(1, 4):
            try:
                dataset = factory()
                if len(dataset) != EXPECTED[name]:
                    raise ValueError(f'{name}: expected {EXPECTED[name]}, found {len(dataset)}')
                del dataset
                gc.collect()
                say(f'{name}: download/integrity check passed')
                break
            except Exception as exc:
                say(f'{name}: attempt {attempt}/3 failed: {exc!r}')
                if attempt == 3:
                    raise
                time.sleep(15)


def validate_all(project):
    counts = {}
    for name, factory in torchvision_datasets(project / 'data', False):
        dataset = factory()
        counts[name] = len(dataset)
        if len(dataset) != EXPECTED[name]:
            raise ValueError(f'{name}: expected {EXPECTED[name]}, found {len(dataset)}')
        # DTD uses image files; the others load their complete arrays in memory.
        if name == 'DTD':
            for index in range(len(dataset)):
                image, _ = dataset[index]
                image.load()
        del dataset
        gc.collect()
    sys.path.insert(0, str(project))
    from hcm.data import ImageDirDataset
    for name, folder in [('TIN', project / 'data/tiny-imagenet-200/test/images'),
                         ('PLACES365', project / 'data/places365')]:
        dataset = ImageDirDataset(str(folder))
        verify_images([Path(p) for p in dataset.files], EXPECTED[name])
        counts[name] = len(dataset)
    report = {'status': 'ready', 'checked_at': time.strftime('%FT%T%z'),
              'project': str(project), 'counts': counts,
              'validation': 'download=False; exact counts; DTD/TIN/Places image decoding'}
    target = project / 'logs/role3_data_validation.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temp.replace(target)
    say('ROLE3_DATA_READY')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--old-download-wait', type=int, default=21600)
    parser.add_argument('--torchvision-worker', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    (project / 'logs').mkdir(exist_ok=True)
    (project / 'data').mkdir(exist_ok=True)
    if args.torchvision_worker:
        torchvision_worker(project)
        return
    import fcntl
    with (project / 'logs/role3_data.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (project / 'logs/role3_data_validation.json').write_text(
            json.dumps({'status': 'preparing', 'started_at': time.strftime('%FT%T%z'),
                        'project': str(project)}, indent=2) + '\n', encoding='utf-8')
        # Separate archives and torchvision files can safely progress in parallel.
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(prepare_manual, project, name) for name in ('TIN', 'PLACES365')]
            futures.append(executor.submit(prepare_torchvision, project, args.old_download_wait))
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(repr(exc))
                    say(f'Data preparation failed: {exc!r}')
            if errors:
                raise RuntimeError('; '.join(errors))
        validate_all(project)


if __name__ == '__main__':
    main()
