"""Atomic artifact writes and content-based provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda f: f.write(payload))


def implementation_hash():
    root = Path(__file__).resolve().parent
    names = ("artifacts.py", "config.py", "dataset.py", "train.py", "protocol.py", "run_all.py",
             "gpu_guard.py", "tc2_guard.py", "execution_control.py", "gpu_smoke.py", "linux_runtime.py", "overnight.py")
    return fingerprint({name: file_hash(root / name) for name in names})
