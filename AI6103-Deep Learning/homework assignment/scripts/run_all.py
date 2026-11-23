"""Explicit subcommands. No arguments or installation never starts training."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
try:
    from . import gpu_guard, execution_control
except ImportError:
    import gpu_guard, execution_control
gpu_guard.bootstrap()
import tempfile
import zipfile
import torch
import numpy as np
try:
    from . import config as registry
    from .artifacts import read_json, write_json, fingerprint, implementation_hash, utc_now
    from .dataset import inspect_dataset, compute_mean_std, get_train_val_loaders
    from .protocol import load_context, plan, read_verified, select_section3, freeze, evaluate_final_test
    from .train import device_for_training, environment, train_experiment
except ImportError:
    import config as registry
    from artifacts import read_json, write_json, fingerprint, implementation_hash, utc_now
    from dataset import inspect_dataset, compute_mean_std, get_train_val_loaders
    from protocol import load_context, plan, read_verified, select_section3, freeze, evaluate_final_test
    from train import device_for_training, environment, train_experiment


def prepare_stats(args):
    integrity = inspect_dataset(args.data_root)
    path = args.output_root / "normalization.json"
    if path.exists():
        context, _, _ = load_context(args.data_root, args.output_root)
        print("Existing verified normalization preserved", context["stats_fingerprint"])
        return
    torch.set_num_threads(2)
    mean, std = compute_mean_std(args.data_root)
    historical = args.output_root / "mean_std.json"
    legacy = read_json(historical) if historical.exists() else None
    reuse = legacy is not None and np.allclose(mean + std, legacy["mean"] + legacy["std"], atol=1e-6, rtol=0)
    result = {"status": "VERIFIED", "dataset_fingerprint": integrity["dataset_fingerprint"],
              "mean": legacy["mean"] if reuse else mean, "std": legacy["std"] if reuse else std,
              "recomputed_mean": mean, "recomputed_std": std,
              "historical_values_preserved": bool(reuse), "computed_at": utc_now(), "training_count": 7155,
              "definition": "RGB population moments of PIL bilinear 100x100 resized train images after ToTensor; float64 accumulation",
              "calculation_environment": environment(torch.device("cpu")),
              "duplicate_policy": "PRESERVE_SUPPLIED_SPLITS",
              "cross_split_duplicate_contents": integrity["cross_split_duplicate_contents"]}
    write_json(path, result)
    write_json(args.output_root / "dataset_integrity.json", integrity)
    print("Verified normalization written; legacy file preserved")


def preflight(args):
    device = torch.device("cpu")
    if gpu_guard.on_server() or os.environ.get("AI6103_PROFILE") in ("gpu6", "tc2"):
        gpu_guard.verify_cuda(torch, require_idle=True)
        device = torch.device("cuda:0")
    context, stats, integrity = load_context(args.data_root, args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_root) as tmp:
        # Probe output writes and serialization/restore without an official model run.
        p = Path(tmp) / "probe.pt"
        torch.save({"probe": torch.arange(4)}, p)
        assert torch.equal(torch.load(p, weights_only=True)["probe"], torch.arange(4))
    env = environment(device)
    free = shutil.disk_usage(args.output_root).free
    result = {"status": "GPU_READY" if torch.cuda.is_available() else "GPU_NOT_AVAILABLE",
              "environment": env, "context": context, "data_splits": integrity["splits"],
              "cross_split_duplicate_contents": integrity["cross_split_duplicate_contents"],
              "disk_free_bytes": free, "disk_minimum_bytes": 4 * 1024**3,
              "disk_ok": free >= 4 * 1024**3, "output_writable": True,
              "serialization_restore_probe": True, "official_configs": [c.to_dict() for c in registry.section3(context)],
              "test_isolation": "Train API has no test loader; final evaluation requires frozen manifest and explicit flag",
              "checked_at": utc_now()}
    if not result["disk_ok"]:
        result["status"] = "INSUFFICIENT_DISK"
    write_json(args.output_root / "qa" / "preflight.json", result)
    print({key: result[key] for key in ("status", "environment", "disk_ok", "disk_free_bytes", "output_writable", "serialization_restore_probe", "cross_split_duplicate_contents")})
    return 0 if result["status"] == "GPU_READY" else 2


def export(args):
    target = args.output_root / "exports"
    target.mkdir(parents=True, exist_ok=True)
    # Archives are rebuilt explicitly; datasets, venvs and backups are excluded.
    with zipfile.ZipFile(target / "report_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for base, prefix in ((args.output_root, "results"), (args.figure_root, "figures"), (registry.ROOT / "report", "report")):
            for path in base.rglob("*"):
                if path.is_file() and "exports" not in path.relative_to(base).parts and path.suffix in (".json", ".png", ".tex", ".bib", ".sty", ".bst", ".pdf"):
                    z.write(path, f"{prefix}/{path.relative_to(base).as_posix()}")
    if args.checkpoints:
        with zipfile.ZipFile(target / "checkpoints.zip", "w", zipfile.ZIP_STORED) as z:
            for path in args.output_root.glob("*/checkpoint_*.pt"):
                z.write(path, path.relative_to(args.output_root).as_posix())
    print(target)


def main(argv=None):
    execution_control.install_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=registry.ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=registry.ROOT / "results")
    parser.add_argument("--figure-root", type=Path, default=registry.ROOT / "figures")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prepare-stats", help="Training images only; no experiment training")
    sub.add_parser("validate")
    sub.add_parser("preflight")
    sub.add_parser("smoke-test")
    train = sub.add_parser("train")
    train.add_argument("--section", type=int, choices=(3, 4, 5, 6), required=True)
    train.add_argument("--experiment", help="Optional exact ID from the section registry")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--allow-cpu", action="store_true")
    select = sub.add_parser("select")
    select.add_argument("--section", type=int, choices=(3, 6), required=True)
    verify = sub.add_parser("verify-run")
    verify.add_argument("--section", type=int, choices=(3, 4, 5, 6), required=True)
    verify.add_argument("--experiment", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--beta-only", action="store_true")
    sub.add_parser("freeze")
    final = sub.add_parser("evaluate-final")
    final.add_argument("--authorize-final-test", action="store_true")
    archive = sub.add_parser("export")
    archive.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    args.figure_root = args.figure_root.resolve()
    if args.command == "prepare-stats":
        prepare_stats(args)
    elif args.command == "preflight":
        return preflight(args)
    elif args.command == "smoke-test":
        try:
            subprocess.run([sys.executable, str(registry.ROOT / "scripts" / "smoke.py"),
                            "--data-root", str(args.data_root), "--output-root", str(args.output_root)],
                           check=True, timeout=60)
        except subprocess.TimeoutExpired:
            write_json(args.output_root / "qa" / "smoke_test.json", {"kind": "SMOKE_TEST_ONLY", "status": "TIMEOUT", "budget_seconds": 60})
            raise RuntimeError("Tiny CPU smoke exceeded 60 seconds; child process stopped")
    elif args.command == "plot" and args.beta_only:
        try:
            from .plots import plot_beta_pdf
        except ImportError:
            from plots import plot_beta_pdf
        plot_beta_pdf(args.figure_root)
    elif args.command == "export":
        export(args)
    else:
        if args.command == "train":
            if gpu_guard.on_server() and os.environ.get("AI6103_PROFILE") != "gpu6":
                raise gpu_guard.GPUAccessError("Official server training requires GPU 6; CPU QA is not a fallback")
            if os.environ.get("AI6103_PROFILE") == "tc2" and not os.environ.get("SLURM_JOB_ID"):
                raise gpu_guard.GPUAccessError("TC2 official training requires an active SLURM allocation")
            # Reject CPU before costly data scans; never silently start CPU full training.
            device = device_for_training(args.allow_cpu)
            if device.type == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        context, stats, integrity = load_context(args.data_root, args.output_root)
        if args.command == "validate":
            statuses = {}
            for section in (3, 4, 5, 6):
                try:
                    configs = plan(section, context, args.output_root)
                    for cfg in configs:
                        path = args.output_root / cfg.experiment_id / "record.json"
                        if not path.exists():
                            statuses[cfg.experiment_id] = "MISSING"
                        else:
                            try:
                                r = read_verified(args.output_root, cfg, complete=False)
                                statuses[cfg.experiment_id] = "COMPLETE_VALID" if r["status"] == "COMPLETE" and r["last_epoch"] == cfg.epochs else "PARTIAL_VALID"
                            except (ValueError, KeyError, FileNotFoundError) as e:
                                statuses[cfg.experiment_id] = f"INVALID_OR_UNVERIFIED: {e}"
                except (ValueError, FileNotFoundError) as e:
                    statuses[f"section_{section}"] = f"DEPENDENCIES_PENDING: {e}"
            invalid = any(value.startswith("INVALID") for value in statuses.values())
            result = {"status": "VALIDATION_FAILED" if invalid else "OFFLINE_VALID", "context": context, "experiments": statuses,
                      "data": integrity, "test_evaluation": "NOT_RUN" if not (args.output_root / "final_evaluation" / "test_started.json").exists() else "SEE_FINAL_EVALUATION_RECORD"}
            write_json(args.output_root / "qa" / "validation.json", result)
            print({"status": result["status"], "experiments": statuses,
                   "splits": {s: v["count"] for s, v in integrity["splits"].items()},
                   "cross_split_duplicate_contents": integrity["cross_split_duplicate_contents"],
                   "test_evaluation": result["test_evaluation"]})
            if invalid:
                return 1
        elif args.command == "train":
            if shutil.disk_usage(args.output_root).free < 4 * 1024**3:
                raise RuntimeError("INSUFFICIENT_DISK: need at least 4 GiB free for durable checkpoints")
            configs = plan(args.section, context, args.output_root)
            if args.experiment:
                configs = [c for c in configs if c.experiment_id == args.experiment]
                if not configs:
                    raise ValueError("Unknown experiment ID for this section")
            for cfg in configs:
                record_path = args.output_root / cfg.experiment_id / "record.json"
                if args.resume and record_path.exists():
                    existing = read_verified(args.output_root, cfg, complete=False)
                    if existing["status"] == "COMPLETE":
                        read_verified(args.output_root, cfg)
                        print("Skipping verified complete", cfg.experiment_id)
                        continue
                tr, va, nc = get_train_val_loaders(stats["mean"], stats["std"], args.data_root, cfg.batch_size, cfg.seed)
                assert nc == cfg.num_classes
                train_experiment(cfg, tr, va, args.output_root, device, args.resume, args.data_root)
        elif args.command == "select":
            if args.section == 3:
                configs = plan(3, context, args.output_root)
                records = [read_verified(args.output_root, c) for c in configs]
                selected = select_section3(records, configs)
                result = {"lr": selected.lr, "rule": "Final training loss ONLY", "selected_id": selected.experiment_id}
            else:
                selected = plan(6, context, args.output_root)[0]
                result = selected.to_dict()
            print(result)
            result["context"] = context
            result["analysis_implementation_fingerprint"] = implementation_hash()
            result["comparability_policy"] = "Scientific environment equality; named execution-instance differences excluded"
            write_json(args.output_root / f"section{args.section}_selection.json", result)
        elif args.command == "verify-run":
            configs = [c for c in plan(args.section, context, args.output_root) if c.experiment_id == args.experiment]
            if len(configs) != 1:
                raise ValueError("Unknown condition")
            record = read_verified(args.output_root, configs[0])
            print({"status": "COMPLETE_VALID", "experiment": args.experiment, "last_epoch": record["last_epoch"],
                   "fingerprint": record["fingerprint"], "checkpoint_sha256": record["checkpoint_sha256"]})
        elif args.command == "freeze":
            print(freeze(context, args.output_root))
        elif args.command == "evaluate-final":
            print(evaluate_final_test(args.data_root, args.output_root, device_for_training(), args.authorize_final_test))
        elif args.command == "plot":
            try:
                from .plots import generate_official_figures
            except ImportError:
                from plots import generate_official_figures
            generate_official_figures(context, args.output_root, args.figure_root)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except execution_control.TrainingStopped as error:
        print(f"SAFE_INTERRUPTION: {error}", file=sys.stderr)
        sys.exit(75)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError, AssertionError) as error:
        print(f"SAFE_STOP: {error}", file=sys.stderr)
        sys.exit(1)
