"""Verified dependencies, deterministic selection and frozen one-time final evaluation."""
import math
import os
from pathlib import Path
try:
    from .gpu_guard import bootstrap
except ImportError:
    from gpu_guard import bootstrap
bootstrap()
import torch
try:
    from . import config as registry
    from .artifacts import file_hash, fingerprint, implementation_hash, read_json, utc_now, write_json
    from .dataset import inspect_dataset, get_final_test_loader
    from .train import METRICS, build_model, load_checkpoint, _evaluate, environment, run_lock
except ImportError:
    import config as registry
    from artifacts import file_hash, fingerprint, implementation_hash, read_json, utc_now, write_json
    from dataset import inspect_dataset, get_final_test_loader
    from train import METRICS, build_model, load_checkpoint, _evaluate, environment, run_lock


def training_implementation_fingerprint():
    """Preserve an explicitly audited training lineage across this analysis-only fix.

    The raw implementation_hash remains truthful. Every hashed file must match the
    audited baseline except the two exact selection/verification replacements. Future
    changes fail closed; this is not permission to ignore implementation fingerprints.
    """
    current = implementation_hash()
    path = registry.ROOT / "deployment/tc2/section3_selection_lineage.json"
    if not path.exists():
        return current
    lineage = read_json(path)
    baseline = lineage["baseline_files"]
    replacements = lineage["analysis_only_replacements"]
    if (lineage.get("schema_version") != 1 or lineage.get("kind") != "AUDITED_SELECTION_ONLY_PATCH"
            or set(replacements) != {"protocol.py", "run_all.py"}
            or not set(replacements) <= set(baseline)
            or any(Path(name).name != name for name in baseline)
            or fingerprint(baseline) != lineage["training_implementation_fingerprint"]):
        raise ValueError("Invalid selection-only implementation lineage")
    actual = {name: file_hash(registry.ROOT / "scripts" / name) for name in baseline}
    if fingerprint(actual) != current or actual != {**baseline, **replacements}:
        raise ValueError("Implementation differs from the audited selection-only patch; preserve completed runs")
    return lineage["training_implementation_fingerprint"]


def load_context(data_root, output_root):
    integrity = inspect_dataset(data_root)
    stats = read_json(Path(output_root) / "normalization.json")
    if stats["dataset_fingerprint"] != integrity["dataset_fingerprint"] or stats["status"] != "VERIFIED":
        raise ValueError("Normalization provenance does not match this dataset")
    if integrity["cross_split_duplicate_contents"] and stats.get("duplicate_policy") != "PRESERVE_SUPPLIED_SPLITS":
        raise ValueError("Duplicate-image provenance must be reviewed before training")
    if len(stats["mean"]) != 3 or len(stats["std"]) != 3 or not all(math.isfinite(x) for x in stats["mean"] + stats["std"]) or not all(x > 0 for x in stats["std"]):
        raise ValueError("Invalid normalization")
    context = {"dataset_fingerprint": integrity["dataset_fingerprint"],
               "stats_fingerprint": fingerprint(stats), "implementation_fingerprint": training_implementation_fingerprint()}
    return context, stats, integrity


def validate_record(record, config, require_complete=True):
    config.validate()
    if record.get("kind") != "OFFICIAL_EXPERIMENT" or record.get("schema_version") != 1:
        raise ValueError("Incompatible legacy or smoke artifact")
    if record.get("config") != config.to_dict() or record.get("fingerprint") != config.fingerprint:
        raise ValueError("Invalid configuration/fingerprint")
    if any("test" in key.lower() for key in record["history"]):
        raise ValueError("Test metrics in ordinary experiment")
    n = record.get("last_epoch", -1)
    if not isinstance(n, int) or not 0 <= n <= config.epochs:
        raise ValueError("Invalid epoch accounting")
    if require_complete and (record.get("status") != "COMPLETE" or n != config.epochs or not record.get("ended_at")):
        raise ValueError("Incomplete experiment")
    if record.get("status") not in ("PARTIAL", "COMPLETE"):
        raise ValueError("Invalid status")
    if record["status"] == "COMPLETE" and (n != config.epochs or not record.get("ended_at")):
        raise ValueError("Completion status contradicts epoch history")
    if record["status"] == "PARTIAL" and (n == config.epochs or record.get("ended_at") is not None):
        raise ValueError("Partial status contradicts epoch history")
    h = record["history"]
    for key in (*METRICS, "lr", "next_lr", "epoch_seconds"):
        values = h.get(key, [])
        if len(values) != n or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
            raise ValueError(f"Invalid history: {key}")
        if key.endswith("acc") and any(not 0 <= x <= 1 for x in values):
            raise ValueError("Accuracy outside [0,1]")
        if ("loss" in key or key == "epoch_seconds") and any(x < 0 for x in values):
            raise ValueError("Negative loss/time")
    for index in range(n):
        expected = config.lr * (1 + math.cos(math.pi * index / config.epochs)) / 2 if config.scheduler == "cosine" else config.lr
        expected_next = config.lr * (1 + math.cos(math.pi * (index + 1) / config.epochs)) / 2 if config.scheduler == "cosine" else config.lr
        if not math.isclose(h["lr"][index], expected, rel_tol=1e-7, abs_tol=1e-10) or not math.isclose(h["next_lr"][index], expected_next, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("Wrong scheduler trajectory")
    if not record.get("environment") or not record.get("started_at"):
        raise ValueError("Missing provenance")
    return record


def read_verified(output_root, config, complete=True):
    directory = Path(output_root) / config.experiment_id
    record = validate_record(read_json(directory / "record.json"), config, complete)
    if record.get("checkpoint") not in ("checkpoint_0.pt", "checkpoint_1.pt"):
        raise ValueError("Invalid checkpoint path")
    path = directory / record["checkpoint"]
    if file_hash(path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    state = load_checkpoint(path, config, restore_random=False)
    saved = state["record"]
    for key in ("config", "fingerprint", "last_epoch", "history", "status", "environment"):
        if saved[key] != record[key]:
            raise ValueError(f"Checkpoint/record mismatch: {key}")
    if state["scheduler"] is None and config.scheduler == "cosine":
        raise ValueError("Missing cosine scheduler state")
    if state["scheduler"] is not None and config.scheduler == "constant":
        raise ValueError("Unexpected scheduler state")
    if config.scheduler == "cosine":
        schedule = state["scheduler"]
        if schedule["T_max"] != config.epochs or schedule["eta_min"] != 0 or schedule["last_epoch"] != record["last_epoch"] or schedule["base_lrs"] != [config.lr]:
            raise ValueError("Scheduler checkpoint state contradicts configuration/epoch")
    expected_lr = record["history"]["next_lr"][-1] if record["last_epoch"] else config.lr
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1 or any(not math.isclose(group["lr"], expected_lr, rel_tol=1e-7, abs_tol=1e-10)
                              or group["momentum"] != config.momentum or group["weight_decay"] != config.weight_decay
                              or group["nesterov"] or group["dampening"] != 0 for group in groups):
        raise ValueError("Optimizer checkpoint state contradicts configuration/history")
    return record


EXECUTION_INSTANCE_FIELDS = frozenset({
    "slurm_job_id", "slurm_timelimit", "slurm_job_gpus", "slurm_node_list", "slurm_gres",
    "slurm_cpus_per_task", "slurm_mem_per_node_mib", "slurm_partition", "slurm_qos",
    "compute_node", "host", "cuda_visible_devices", "cuda_device_order", "logical_gpu_index",
    "nvidia_visible_index", "physical_gpu_index", "physical_gpu_uuid", "runtime_gpu_uuid",
    "gpu_free_mib", "gpu_utilization_percent", "timestamp", "started_at", "ended_at",
})


def scientific_environment(environment):
    """Conservative numerical signature, omitting only named execution-instance data.

    All unrecognized fields remain compared, including future AMP/TF32/dtype metadata.
    Different GPU models are deliberately rejected pending an explicit assessment.
    Legacy records lack per-run precision flags; their common audited producer uses
    FP32 without autocast, and config/code/software equality is required below.
    """
    required = ("python", "torch", "torchvision", "numpy", "pillow", "platform", "cpu_threads", "device")
    if not isinstance(environment, dict) or any(environment.get(key) in (None, "") for key in required):
        raise ValueError("Missing scientific execution-environment provenance")
    signature = {key: value for key, value in environment.items() if key not in EXECUTION_INSTANCE_FIELDS}
    signature["device"] = str(environment["device"]).split(":")[0]
    if signature["device"] == "cuda":
        for key in ("gpu_name", "gpu_memory_bytes", "cuda_version", "nvidia_driver"):
            if environment.get(key) in (None, ""):
                raise ValueError(f"Missing CUDA numerical provenance: {key}")
        for first, second in (("torch", "torch_version"), ("cuda_version", "cuda_runtime"),
                              ("gpu_name", "runtime_gpu_name"), ("gpu_memory_bytes", "runtime_gpu_total_bytes")):
            if second in environment and environment[first] != environment[second]:
                raise ValueError(f"Inconsistent environment aliases: {first}/{second}")
    return signature


def require_comparable(records):
    if not records:
        raise ValueError("Missing comparison records")
    signature = scientific_environment(records[0]["environment"])
    # LR/scheduler/weight decay/Mixup are intentional experimental variables. Shared
    # provenance and preprocessing/model/numerical settings must still be identical.
    shared = ("dataset_fingerprint", "stats_fingerprint", "implementation_fingerprint", "model", "weights",
              "optimizer", "batch_size", "momentum", "image_size", "flip_probability", "crop_padding",
              "num_classes", "num_workers", "deterministic", "seed")
    for record in records[1:]:
        other = scientific_environment(record["environment"])
        differences = sorted(key for key in set(signature) | set(other) if signature.get(key) != other.get(key)
                             or (key in signature) != (key in other))
        if differences:
            raise ValueError("Scientifically incompatible environments: " + ", ".join(differences))
        if any(record["config"][key] != records[0]["config"][key] for key in shared):
            raise ValueError("Incompatible shared scientific configuration/provenance")


def select_section3(records, configs):
    if [c.lr for c in configs] != registry.SECTION3_LRS or any(c.section != 3 for c in configs) or len(records) != 3:
        raise ValueError("All three corrected candidates are mandatory")
    for record, cfg in zip(records, configs):
        validate_record(record, cfg)
    require_comparable(records)
    # Stable original candidate order resolves exact ties; validation never affects this key.
    index = min(range(3), key=lambda i: records[i]["history"]["train_loss"][-1])
    return configs[index]


def select_validation(records, configs):
    if len(records) != len(configs) or not configs:
        raise ValueError("Missing candidates")
    for record, cfg in zip(records, configs):
        validate_record(record, cfg)
    require_comparable(records)
    # Predeclared final validation CE; stable registry order breaks exact ties.
    i = min(range(len(records)), key=lambda i: records[i]["history"]["val_loss"][-1])
    return configs[i]


def plan(section, context, output_root):
    c3 = registry.section3(context)
    if section == 3:
        return c3
    r3 = [read_verified(output_root, c) for c in c3]
    best_lr = select_section3(r3, c3).lr
    c4 = registry.section4(context, best_lr)
    if section == 4:
        return c4
    r4 = [read_verified(output_root, c) for c in c4]
    c5 = registry.section5(context, best_lr)
    if section == 5:
        return c5
    r5 = [read_verified(output_root, c) for c in c5]
    require_comparable(r3 + r4 + r5)
    baseline = select_validation(r4 + r5, c4 + c5)
    return [registry.section6(context, baseline)]


def completed_bundle(context, output_root):
    c3 = plan(3, context, output_root)
    c4 = plan(4, context, output_root)
    c5 = plan(5, context, output_root)
    c6 = plan(6, context, output_root)
    configs = c3 + c4 + c5 + c6
    records = [read_verified(output_root, c) for c in configs]
    require_comparable(records)
    baseline = select_validation(records[3:7], configs[3:7])
    return configs, records, baseline


def freeze(context, output_root):
    output_root = Path(output_root)
    with run_lock(output_root / "final_evaluation"):
        path = output_root / "frozen_manifest.json"
        if path.exists():
            raise FileExistsError("Frozen manifest already exists; it is immutable")
        configs, records, baseline = completed_bundle(context, output_root)
        body = {"schema_version": 1, "status": "FROZEN", "created_at": utc_now(), "context": context,
                "selected_experiment": "sec6_mixup", "baseline": baseline.experiment_id,
                "selection_rule": "Section 3 final train CE; downstream final validation CE; final prescribed Mixup model at epoch 300",
                "records": {c.experiment_id: file_hash(output_root / c.experiment_id / "record.json") for c in configs}}
        body["manifest_fingerprint"] = fingerprint(body)
        write_json(path, body)
        return body


def evaluate_final_test(data_root, output_root, device, authorized=False):
    """Explicit future action only. Claim is durable BEFORE opening the test loader."""
    if os.environ.get("AI6103_NO_FINAL_TEST") == "1":
        raise PermissionError("Phase A.6 overnight policy forbids final test evaluation")
    if not authorized:
        raise PermissionError("--authorize-final-test required")
    output_root = Path(output_root)
    with run_lock(output_root / "final_evaluation"):
        manifest = read_json(output_root / "frozen_manifest.json")
        digest = manifest.pop("manifest_fingerprint")
        if fingerprint(manifest) != digest or manifest["status"] != "FROZEN":
            raise ValueError("Frozen manifest corrupted")
        context, stats, _ = load_context(data_root, output_root)
        if manifest["context"] != context:
            raise ValueError("Frozen data/code context changed")
        configs, records, _ = completed_bundle(context, output_root)
        if manifest["records"] != {c.experiment_id: file_hash(output_root / c.experiment_id / "record.json") for c in configs}:
            raise ValueError("Frozen records changed")
        claim = output_root / "final_evaluation" / "test_started.json"
        if claim.exists():
            raise RuntimeError("Final test already started; automatic repeated evaluation forbidden")
        cfg = configs[-1]
        model = build_model(cfg.num_classes, device)
        record = records[-1]
        cp = output_root / cfg.experiment_id / record["checkpoint"]
        load_checkpoint(cp, cfg, model, restore_random=False)
        write_json(claim, {"manifest_fingerprint": digest, "started_at": utc_now(), "status": "STARTED"})
        loader = get_final_test_loader(stats["mean"], stats["std"], data_root)
        loss, accuracy = _evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
        result = {"status": "COMPLETE", "manifest_fingerprint": digest, "checkpoint_sha256": record["checkpoint_sha256"],
                  "experiment_id": cfg.experiment_id, "test_loss": loss, "test_acc": accuracy,
                  "count": len(loader.dataset), "ended_at": utc_now(), "environment": environment(device)}
        write_json(output_root / "final_evaluation" / "test_result.json", result)
        return result
