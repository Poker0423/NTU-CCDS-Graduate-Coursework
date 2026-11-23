"""Synthetic temporary artifacts only; no data images, remote calls, or training."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import fixture_record
from scripts import artifacts, config, protocol, train

ROOT = config.ROOT
SPEC = importlib.util.spec_from_file_location("section5_ops_tests", ROOT / "deployment/tc2/section5_ops.py")
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    artifacts.write_json(output / "normalization.json", {"status": "VERIFIED", "training_count": 7155,
        "dataset_fingerprint": "SYNTHETIC_TEST_ONLY", "mean": [0.4]*3, "std": [0.2]*3})
    artifacts.write_json(output / "dataset_integrity.json", {"dataset_fingerprint": "SYNTHETIC_TEST_ONLY",
                                                          "cross_split_duplicate_contents": 0})
    context = ops.saved_context(tmp_path)
    artifacts.write_json(output / "section3_selection.json", {"lr": 0.05, "selected_id": "sec3_lr0.05",
        "rule": "Final training loss ONLY", "context": context})
    cfgs = config.section3(context) + config.section4(context, 0.05) + config.section5(context, 0.05)
    records = {cfg.experiment_id: fixture_record(cfg, loss=1 if cfg.lr == 0.05 else 2) for cfg in cfgs}
    for name, record in records.items():
        record["environment"]["slurm_job_id"] = {"sec4_constant": "42360", "sec4_cosine": "42419",
            "sec5_wd5e-4": "50001", "sec5_wd1e-4": "50002"}.get(name, "42226")
    real_read = protocol.read_verified
    def read(root, cfg, complete=True):
        if (Path(root) / cfg.experiment_id / "record.json").exists():
            return real_read(root, cfg, complete)
        return protocol.validate_record(records[cfg.experiment_id], cfg, complete)
    monkeypatch.setattr(protocol, "read_verified", read)
    monkeypatch.setattr(ops, "completed_job", lambda job: {"job_id": str(job), "state": "COMPLETED", "exit_code": "0:0"})
    return tmp_path, context, records


def test_section5_registry_differs_only_in_prescribed_decay(prepared):
    root, context, _ = prepared
    _, candidates, _, _ = ops.checked_plan(root)
    assert [(c.experiment_id, c.weight_decay) for c in candidates] == [("sec5_wd5e-4", 5e-4), ("sec5_wd1e-4", 1e-4)]
    a, b = [c.to_dict() for c in candidates]
    assert {key for key in a if a[key] != b[key]} == {"experiment_id", "weight_decay"}
    assert all(c.epochs == 300 and c.batch_size == 128 and c.lr == 0.05 and c.scheduler == "cosine" for c in candidates)


@pytest.mark.parametrize("missing", ["sec3_lr0.5", "sec3_lr0.05", "sec3_lr0.001", "sec4_constant", "sec4_cosine"])
def test_missing_upstream_cannot_be_bypassed(prepared, missing):
    root, _, records = prepared
    records.pop(missing)
    with pytest.raises(KeyError):
        ops.checked_plan(root)


def test_running_or_incomplete_cosine_blocks_preflight(prepared, monkeypatch):
    root, _, records = prepared
    records["sec4_cosine"]["status"] = "PARTIAL"
    with pytest.raises(ValueError):
        ops.preflight(root, "sec5_wd5e-4")
    records["sec4_cosine"]["status"] = "COMPLETE"
    def active(job):
        if job == "42419":
            raise ValueError("RUNNING")
        return {}
    monkeypatch.setattr(ops, "completed_job", active)
    with pytest.raises(ValueError, match="RUNNING"):
        ops.preflight(root, "sec5_wd5e-4")


def test_selection_artifact_must_match_recomputed_lr_and_context(prepared):
    root, context, _ = prepared
    path = root / "results/section3_selection.json"
    original = artifacts.read_json(path)
    artifacts.write_json(path, {**original, "lr": 0.5})
    with pytest.raises(ValueError, match="selection"):
        ops.checked_plan(root)
    artifacts.write_json(path, {**original, "context": {**context, "stats_fingerprint": "WRONG"}})
    with pytest.raises(ValueError, match="selection"):
        ops.checked_plan(root)


@pytest.mark.parametrize("artifact", ["record.json", "checkpoint_0.pt", "checkpoint_1.pt", ".run.lock", "unrecognized.tmp"])
def test_existing_or_orphan_results_are_never_overwritten(prepared, artifact):
    root, _, _ = prepared
    directory = root / "results/sec5_wd5e-4"
    directory.mkdir()
    path = directory / artifact
    path.write_bytes(b"PRESERVE")
    with pytest.raises(FileExistsError):
        ops.preflight(root, "sec5_wd5e-4")
    assert path.read_bytes() == b"PRESERVE"


def test_no_test_files_or_images_are_needed_for_preflight(prepared, monkeypatch):
    root, _, _ = prepared
    monkeypatch.setattr(protocol, "load_context", lambda *_: pytest.fail("Do not scan dataset"))
    assert not (root / "data").exists()
    assert ops.preflight(root, "sec5_wd1e-4")["status"] == "SECTION5_PREFLIGHT_PASS"


def test_no_gpu_context_without_explicit_test_integrity_policy(prepared, monkeypatch):
    root, _, _ = prepared
    monkeypatch.delenv("AI6103_SECTION5_ALLOW_TEST_INTEGRITY", raising=False)
    monkeypatch.setattr(ops.gpu_guard, "verify_cuda", lambda *_: pytest.fail("No GPU initialization allowed"))
    with pytest.raises(PermissionError, match="test integrity"):
        ops.preflight(root, "sec5_wd5e-4", check_gpu=True)


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("RUNNING", "0:0"), ("TIMEOUT", "0:15"), ("COMPLETED", "1:0")])
def test_accounting_requires_actual_success(state, code):
    def runner(command, **kwargs):
        assert command[0] == "sacct" and kwargs["check"] is True
        return SimpleNamespace(stdout=f"50001|{state}|{code}|\n")
    with pytest.raises(ValueError):
        ops.completed_job("50001", runner)


def test_accounting_does_not_treat_missing_job_as_success():
    with pytest.raises(ValueError):
        ops.completed_job("50001", lambda *a, **k: SimpleNamespace(stdout=""))
    result = ops.completed_job("50001", lambda *a, **k: SimpleNamespace(stdout="50001|COMPLETED|0:0|\n50001.batch|COMPLETED|0:0|\n"))
    assert result["state"] == "COMPLETED"


def save_tiny_checkpoint(root, cfg, record):
    directory = root / "results" / cfg.experiment_id
    directory.mkdir()
    model = torch.nn.Linear(1, 1)
    optimizer = train.build_optimizer(model, cfg)
    scheduler = train.build_scheduler(optimizer, cfg)
    scheduler.last_epoch = 300
    optimizer.param_groups[0]["lr"] = 0.0
    record["checkpoint"] = "checkpoint_0.pt"
    train.save_checkpoint(directory / record["checkpoint"], model, optimizer, scheduler, record)
    record["checkpoint_sha256"] = artifacts.file_hash(directory / record["checkpoint"])
    artifacts.write_json(directory / "record.json", record)
    return directory


def test_verify_reuses_real_checkpoint_contract_and_preserves_inputs(prepared):
    root, context, records = prepared
    cfg = config.section5(context, 0.05)[0]
    directory = save_tiny_checkpoint(root, cfg, records[cfg.experiment_id])
    before = {p.name: artifacts.file_hash(p) for p in directory.iterdir()}
    _, _, report = ops.verify(root, cfg.experiment_id, "50001")
    assert report["status"] == "SECTION5_COMPLETE_VALID" and report["last_epoch"] == 300
    assert {p.name: artifacts.file_hash(p) for p in directory.iterdir()} == before
    with pytest.raises(ValueError, match="Job ID"):
        ops.verify(root, cfg.experiment_id, "50002")
    (directory / "checkpoint_0.pt").write_bytes(b"BAD")
    with pytest.raises(ValueError, match="checksum"):
        ops.verify(root, cfg.experiment_id, "50001")


def test_comparison_has_metrics_curves_baseline_and_no_automatic_selection(prepared):
    root, context, records = prepared
    # Explicitly synthetic histories; no model computation or real experimental claims.
    for cfg in [config.section4(context, 0.05)[1], *config.section5(context, 0.05)]:
        save_tiny_checkpoint(root, cfg, records[cfg.experiment_id])
    path = root / "results/qa/section5/90001"
    result = ops.compare(root, "50001", "50002", path)
    report = artifacts.read_json(path / "comparison.json")
    assert result["status"] == "SECTION5_COMPARISON_COMPLETED"
    assert set(report["metrics"]) == {"sec4_cosine", "sec5_wd5e-4", "sec5_wd1e-4"}
    assert report["selection_performed"] is False and report["test_evaluation"] == "NOT_RUN"
    assert len((path / "curves.csv").read_text().splitlines()) == 901
    assert (path / "section5_comparison.png").stat().st_size > 10000
    with pytest.raises(FileExistsError):
        ops.compare(root, "50001", "50002", path)


@pytest.mark.parametrize("name,experiment", [("section5_wd5e4.sbatch", "sec5_wd5e-4"), ("section5_wd1e4.sbatch", "sec5_wd1e-4")])
def test_job_scripts_run_exactly_one_condition_without_resume(name, experiment):
    source = (ROOT / "deployment/tc2" / name).read_text()
    train_commands = [line for line in source.splitlines() if line.startswith("python ") and "run_all.py train" in line]
    assert train_commands == [f"python -B -u scripts/run_all.py train --section 5 --experiment {experiment}"]
    assert "--resume" not in source and "--allow-cpu" not in source
    assert "export CUDA_VISIBLE_DEVICES=" not in source and "CUDA_DEVICE_ORDER=" not in source
    assert 'export AI6103_PROFILE=tc2' in source and 'AI6103_NO_FINAL_TEST=1' in source
    assert source.index("conda activate") < source.index("set -u")
    assert source.index("section5_ops.py preflight") < source.index("run_all.py train")
    assert "flock -n 9" in source and "|| true" not in source
    for flag in ("--partition=MGPU-TC2", "--qos=normal", "--gres=gpu:1", "--cpus-per-task=2", "--mem=20G", "--time=05:30:00"):
        assert "#SBATCH " + flag in source


def test_dependency_and_comparison_jobs_never_start_training():
    for name in ("section5_preflight.sbatch", "section5_verify.sbatch", "section5_compare.sbatch"):
        source = (ROOT / "deployment/tc2" / name).read_text()
        assert "run_all.py train" not in source and "--check-gpu" not in source
        assert "export AI6103_PROFILE=cpu" in source and "export CUDA_VISIBLE_DEVICES=-1" in source
        assert "evaluate-final" not in source
