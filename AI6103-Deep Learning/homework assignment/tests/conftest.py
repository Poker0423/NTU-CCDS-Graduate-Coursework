"""TEST-ONLY synthetic fixtures; temporary directories never feed homework artifacts."""
import math
from pathlib import Path
import sys
# The CPU overlay shares read-only global packages. Keep assertion rewrites in memory;
# old pytest/atomicwrites otherwise retries denied scipy-site-package cache writes.
sys.dont_write_bytecode = True
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.config import section3
from scripts.train import METRICS, environment

torch.set_num_threads(2)


@pytest.fixture
def context():
    return {"dataset_fingerprint": "TEST_ONLY", "stats_fingerprint": "TEST_ONLY", "implementation_fingerprint": "TEST_ONLY"}


@pytest.fixture
def configs(context):
    return section3(context)


def fixture_record(cfg, loss=2.0, val=3.0, last_epoch=None):
    n = cfg.epochs if last_epoch is None else last_epoch
    h = {key: [loss if "loss" in key else 0.1] * n for key in METRICS}
    h["val_loss"] = [val] * n
    h["lr"] = [cfg.lr * (1 + math.cos(math.pi * i / cfg.epochs)) / 2 if cfg.scheduler == "cosine" else cfg.lr for i in range(n)]
    h["next_lr"] = [cfg.lr * (1 + math.cos(math.pi * (i + 1) / cfg.epochs)) / 2 if cfg.scheduler == "cosine" else cfg.lr for i in range(n)]
    h["epoch_seconds"] = [0.01] * n
    return {"schema_version": 1, "kind": "OFFICIAL_EXPERIMENT", "status": "COMPLETE" if n == cfg.epochs else "PARTIAL",
            "config": cfg.to_dict(), "fingerprint": cfg.fingerprint, "history": h, "last_epoch": n,
            "environment": environment(torch.device("cpu")), "started_at": "TEST_ONLY",
            "ended_at": "TEST_ONLY" if n == cfg.epochs else None, "resume_timestamps": []}
