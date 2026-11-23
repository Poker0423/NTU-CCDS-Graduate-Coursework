"""Sample-weighted metrics and epoch-boundary resumable training; no test loader API."""
import copy
import os
import platform
import random
from contextlib import contextmanager
from pathlib import Path
import time
try:
    from . import gpu_guard, execution_control
except ImportError:
    import gpu_guard, execution_control
gpu_guard.bootstrap()
import numpy as np
import torch
from torch import nn
import torchvision
import PIL
from torchvision.models import efficientnet_v2_s
try:
    from .artifacts import atomic_write, file_hash, fingerprint, read_json, utc_now, write_json
except ImportError:
    from artifacts import atomic_write, file_hash, fingerprint, read_json, utc_now, write_json

METRICS = ("train_loss", "train_acc", "val_loss", "val_acc")


def device_for_training(allow_cpu=False):
    if gpu_guard.bootstrap() == "cpu":
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("GPU_NOT_AVAILABLE: explicit CPU QA profile")
    if gpu_guard.on_server() or os.environ.get("AI6103_PROFILE") in ("gpu6", "tc2"):
        if allow_cpu:
            raise gpu_guard.GPUAccessError("Remote official training forbids CPU fallback")
        gpu_guard.verify_cuda(torch, require_idle=True)
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        gpu_guard.verify_cuda(torch, require_idle=True)
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("GPU_NOT_AVAILABLE: official training refused; --allow-cpu is an explicit future override")


def environment(device):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "pillow": PIL.__version__, "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda, "device": str(device),
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
            "determinism_limitations": "Same environment/device and epoch boundaries only; cross-platform identity is not guaranteed."}
    if device.type == "cuda":
        info.update(gpu_guard.identity())
    return info


def seed_all(seed):
    gpu_guard.bootstrap()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_guard.verify_cuda(torch)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed(seed)  # logical device 0 only, after verified single-device mask
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_model(num_classes=120, device="cpu"):
    if torch.device(device).type == "cuda":
        gpu_guard.verify_cuda(torch)
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)


def build_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                           weight_decay=config.weight_decay)


def build_scheduler(optimizer, config):
    return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)
            if config.scheduler == "cosine" else None)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        raise ValueError("Mixup alpha must be positive")
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=False, alpha=0.2):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        execution_control.check_stop()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_mixup:
            images, ya, yb, lam = mixup_data(images, labels, alpha)
            logits = model(images)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            preds = logits.argmax(1)
            correct += (lam * (preds == ya).float() + (1 - lam) * (preds == yb).float()).sum().item()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite optimization loss; experiment remains PARTIAL")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    if not total:
        raise ValueError("Empty training loader")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device):
    if getattr(loader.dataset, "split", None) == "test":
        raise ValueError("Test loader is forbidden in validation")
    return _evaluate(model, loader, criterion, device)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        execution_control.check_stop()
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    if not total:
        raise ValueError("Empty evaluation loader")
    return total_loss / total, correct / total


def capture_rng(loaders=()):
    gpu_guard.bootstrap()
    if torch.cuda.is_available():
        gpu_guard.verify_cuda(torch)
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "cuda": [torch.cuda.get_rng_state(0)] if torch.cuda.is_available() else [],
             "loaders": [loader.generator.get_state() if loader.generator else None for loader in loaders]}
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, loaders=()):
    gpu_guard.bootstrap()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        gpu_guard.verify_cuda(torch)
        if len(state["cuda"]) != 1:
            raise gpu_guard.GPUAccessError("Checkpoint must contain only the authorized CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"][0], 0)
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if len(loaders) != len(state["loaders"]):
        raise ValueError("Loader RNG mismatch")
    for loader, value in zip(loaders, state["loaders"]):
        if value is not None:
            loader.generator.set_state(value)


def save_checkpoint(path, model, optimizer, scheduler, record, loaders=()):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler else None,
             "record": copy.deepcopy(record), "rng": capture_rng(loaders)}
    atomic_write(path, lambda f: torch.save(state, f))


def load_checkpoint(path, config, model=None, optimizer=None, scheduler=None, loaders=(), restore_random=True):
    # Local checkpoints contain Python/NumPy RNG state. Never load an untrusted .pt file.
    state = torch.load(path, map_location="cpu", weights_only=False)
    record = state["record"]
    if record["config"] != config.to_dict() or record["fingerprint"] != config.fingerprint:
        raise ValueError("Incompatible checkpoint configuration")
    if model is not None:
        model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("Missing scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    elif state["scheduler"] is not None and optimizer is not None:
        raise ValueError("Unexpected scheduler state")
    if restore_random:
        restore_rng(state["rng"], loaders)
    return state


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".run.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run locked: {path}. Verify the original process is stopped before removing a stale lock.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} time={utc_now()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def train_experiment(config, train_loader, val_loader, output_root, device, resume=False,
                     dataset_root=None):
    """Every call initializes a fresh model unless explicitly restoring a compatible run."""
    config.validate()
    if torch.device(device).type == "cuda":
        gpu_guard.verify_cuda(torch, require_idle=True)
    if getattr(train_loader.dataset, "split", None) == "test" or getattr(val_loader.dataset, "split", None) == "test":
        raise ValueError("Test data forbidden in training")
    output_root = Path(output_root)
    if (output_root / "frozen_manifest.json").exists():
        raise ValueError("Experiments are frozen; use a new output directory for new research")
    directory = output_root / config.experiment_id
    record_path = directory / "record.json"
    with run_lock(directory):
        if any(directory.glob("checkpoint_*.pt")) and not record_path.exists():
            raise ValueError("Orphan checkpoints preserved; inspect and use a new output directory")
        seed_all(config.seed)
        model = build_model(config.num_classes, device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        loaders = (train_loader, val_loader)
        if record_path.exists():
            if not resume:
                raise FileExistsError("Run exists; explicit --resume required")
            record = read_json(record_path)
            if record["config"] != config.to_dict() or record["fingerprint"] != config.fingerprint:
                raise ValueError("Incompatible existing record")
            if record["environment"] != environment(device):
                raise ValueError("Resume environment changed; preserve run and start a separate experiment")
            checkpoint = directory / record["checkpoint"]
            if file_hash(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            state = load_checkpoint(checkpoint, config, model, optimizer, scheduler, loaders)
            if state["record"]["last_epoch"] != record["last_epoch"] or state["record"]["history"] != record["history"]:
                raise ValueError("Checkpoint/record epoch mismatch")
            if record["status"] == "COMPLETE":
                return record
            record["resume_timestamps"].append(utc_now())
        else:
            record = {"schema_version": 1, "kind": "OFFICIAL_EXPERIMENT", "status": "PARTIAL",
                      "config": config.to_dict(), "fingerprint": config.fingerprint,
                      "environment": environment(device), "dataset_root": str(Path(dataset_root).resolve()) if dataset_root else None,
                      "output_root": str(output_root.resolve()), "started_at": utc_now(),
                      "ended_at": None, "resume_timestamps": [], "last_epoch": 0,
                      "history": {key: [] for key in (*METRICS, "lr", "next_lr", "epoch_seconds")},
                      "metric_semantics": "Mixup optimization CE and soft-label-weighted train accuracy" if config.mixup else "Augmented train CE and accuracy",
                      "validation_semantics": "Clean validation cross-entropy and top-1 accuracy"}

        def commit():
            # Two generations preserve the last committed checkpoint during the next atomic write.
            record["checkpoint"] = f"checkpoint_{record['last_epoch'] % 2}.pt"
            cp = directory / record["checkpoint"]
            save_checkpoint(cp, model, optimizer, scheduler, record, loaders)
            record["checkpoint_sha256"] = file_hash(cp)
            write_json(record_path, record)

        if record["last_epoch"] == 0 and not record_path.exists():
            commit()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(record["last_epoch"] + 1, config.epochs + 1):
            execution_control.check_stop()
            if torch.device(device).type == "cuda":
                record["last_resource_snapshot"] = gpu_guard.check_running_resources(torch)
            import shutil
            if shutil.disk_usage(output_root).free < 1024**3:
                raise RuntimeError("Insufficient disk headroom for atomic checkpoint rotation")
            start = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, config.mixup, config.mixup_alpha)
            vl, va = evaluate_validation(model, val_loader, criterion, device)
            if not all(np.isfinite(v) for v in (tl, ta, vl, va)):
                raise ValueError("Non-finite metrics; last committed epoch preserved")
            if scheduler is not None:
                scheduler.step()  # epoch e uses eta_(e-1); after epoch 300 eta_300=0
            for key, value in zip(METRICS, (tl, ta, vl, va)):
                record["history"][key].append(float(value))
            record["history"]["lr"].append(lr)
            record["history"]["next_lr"].append(optimizer.param_groups[0]["lr"])
            record["history"]["epoch_seconds"].append(time.monotonic() - start)
            record["last_epoch"] = epoch
            if epoch == config.epochs:
                record["status"] = "COMPLETE"
                record["ended_at"] = utc_now()
            commit()
            print(f"[{config.experiment_id}] {epoch}/{config.epochs} train_loss={tl:.6f} val_loss={vl:.6f}", flush=True)
        return record
