"""Temporary synthetic artifacts and bounded CPU units; never official training."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import fixture_record
from deployment.tc2 import section6_ops as ops
from scripts import artifacts, config, protocol, train


def save_tiny(root, cfg, record):
    directory = root / "results" / cfg.experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(1, 1)
    optimizer = train.build_optimizer(model, cfg)
    scheduler = train.build_scheduler(optimizer, cfg)
    if scheduler:
        scheduler.last_epoch = cfg.epochs
        optimizer.param_groups[0]["lr"] = 0.0
    record["checkpoint"] = "checkpoint_0.pt"
    record["metric_semantics"] = "Mixup optimization CE and soft-label-weighted train accuracy" if cfg.mixup else "Augmented train CE and accuracy"
    record["validation_semantics"] = "Clean validation cross-entropy and top-1 accuracy"
    train.save_checkpoint(directory / "checkpoint_0.pt", model, optimizer, scheduler, record)
    record["checkpoint_sha256"] = artifacts.file_hash(directory / "checkpoint_0.pt")
    artifacts.write_json(directory / "record.json", record)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    output = tmp_path / "results"
    artifacts.write_json(output / "normalization.json", {"status": "VERIFIED", "training_count": 7155,
        "dataset_fingerprint": "TEST_ONLY", "mean": [0.4]*3, "std": [0.2]*3})
    artifacts.write_json(output / "dataset_integrity.json", {"dataset_fingerprint": "TEST_ONLY", "cross_split_duplicate_contents": 0})
    context = ops.prior.saved_context(tmp_path)
    artifacts.write_json(output / "section3_selection.json", {"lr": 0.05, "selected_id": "sec3_lr0.05",
        "rule": "Final training loss ONLY", "context": context})
    candidates = config.section3(context) + config.section4(context, 0.05) + config.section5(context, 0.05)
    records = {}
    for index, cfg in enumerate(candidates):
        record = fixture_record(cfg, loss=1 if cfg.lr == 0.05 else 2, val=8-index)
        record["environment"]["slurm_job_id"] = str(50000 + index)
        records[cfg.experiment_id] = (cfg, record)
        save_tiny(tmp_path, cfg, record)
    monkeypatch.setattr(ops.prior, "completed_job", lambda job: {"job_id": str(job), "state": "COMPLETED", "exit_code": "0:0"})
    return tmp_path, context, records


def test_baseline_dynamic_final_validation_ce_not_accuracy_or_best_epoch(prepared):
    root, _, records = prepared
    cfg, record = records["sec4_constant"]
    record["history"]["val_acc"] = [0.99]*300
    record["history"]["val_loss"][0] = 0.001
    save_tiny(root, cfg, record)
    resolved, baseline, _, evidence = ops.checked_plan(root)
    assert baseline.experiment_id == "sec5_wd1e-4"
    assert resolved.mixup and resolved.mixup_alpha == 0.2 and resolved.lr == 0.05
    assert resolved.scheduler == baseline.scheduler and resolved.weight_decay == baseline.weight_decay
    assert len(evidence["candidates"]) == 4
    assert evidence["policy"]["assignment_status"].startswith("Experimental design")
    assert evidence["candidates"][0]["final_validation_accuracy"] == 0.99


@pytest.mark.parametrize("winner", ["sec4_constant", "sec4_cosine", "sec5_wd5e-4", "sec5_wd1e-4"])
def test_each_eligible_baseline_inherits_exact_settings(prepared, winner):
    root, _, records = prepared
    cfg, record = records[winner]
    record["history"]["val_loss"][-1] = 0.0001
    save_tiny(root, cfg, record)
    mixup, base, _, _ = ops.checked_plan(root)
    assert base.experiment_id == winner
    assert (mixup.lr, mixup.scheduler, mixup.weight_decay) == (base.lr, base.scheduler, base.weight_decay)
    assert mixup.epochs == 300 and mixup.batch_size == 128


def test_final_validation_tie_preserves_registry_order(prepared):
    root, _, records = prepared
    for name in ops.POLICY["candidates"]:
        cfg, record = records[name]
        record["history"]["val_loss"][-1] = 1.0
        save_tiny(root, cfg, record)
    assert ops.checked_plan(root)[1].experiment_id == "sec4_constant"


@pytest.mark.parametrize("missing", ["sec5_wd5e-4", "sec5_wd1e-4"])
def test_missing_section5_rejected(prepared, missing):
    root, _, _ = prepared
    (root / "results" / missing / "record.json").unlink()
    with pytest.raises(FileNotFoundError):
        ops.create_selection(root, "60001")
    assert not (root / "results/qa/section6/60001").exists()


def test_incomplete_or_failed_section5_rejected(prepared, monkeypatch):
    root, _, records = prepared
    path = root / "results/sec5_wd5e-4/record.json"
    record = artifacts.read_json(path)
    record["status"] = "PARTIAL"
    artifacts.write_json(path, record)
    with pytest.raises(ValueError):
        ops.checked_plan(root)
    save_tiny(root, *records["sec5_wd5e-4"])
    def account(job):
        if job == "50005":
            raise ValueError("RUNNING/FAILED")
        return {}
    monkeypatch.setattr(ops.prior, "completed_job", account)
    with pytest.raises(ValueError, match="RUNNING/FAILED"):
        ops.checked_plan(root)


@pytest.mark.parametrize("field,value", [("mixup_alpha", 0.3), ("scheduler", "constant"), ("weight_decay", 5e-4)])
def test_incorrect_alpha_scheduler_or_decay_rejected(prepared, monkeypatch, field, value):
    root, _, _ = prepared
    original = config.section6
    monkeypatch.setattr(config, "section6", lambda context, base: replace(original(context, base), **{field:value}))
    with pytest.raises(ValueError):
        ops.checked_plan(root)


@pytest.mark.parametrize("name", ["record.json", "checkpoint_1.pt", ".run.lock"])
def test_existing_output_preserved(prepared, name):
    root, _, _ = prepared
    destination = root / "results/sec6_mixup"
    destination.mkdir()
    (destination / name).write_bytes(b"PRESERVE")
    with pytest.raises(FileExistsError):
        ops.create_selection(root, "60001")
    assert (destination / name).read_bytes() == b"PRESERVE"


def test_corrupt_upstream_checkpoint_rejected(prepared):
    root, _, _ = prepared
    (root / "results/sec5_wd1e-4/checkpoint_0.pt").write_bytes(b"CORRUPT")
    with pytest.raises(ValueError, match="checksum"):
        ops.checked_plan(root)


def test_selection_artifact_seals_candidate_losses_and_provenance(prepared):
    root, _, _ = prepared
    result = ops.create_selection(root, "60001")
    saved = artifacts.read_json(result["selection"])
    assert saved["selected_baseline_id"] == "sec5_wd1e-4"
    assert len(saved["upstream_records"]) == 7
    assert all(c["record_sha256"] and c["checkpoint_sha256"] and c["accounting"]["state"] == "COMPLETED" for c in saved["candidates"])
    assert ops.verify_selection(root, "60001")[0].experiment_id == "sec6_mixup"
    with pytest.raises(FileExistsError):
        ops.create_selection(root, "60001")
    saved["selected_baseline_id"] = "sec4_constant"
    artifacts.write_json(result["selection"], saved)
    with pytest.raises(ValueError, match="corrupted"):
        ops.verify_selection(root, "60001")


def test_changed_provenance_after_selection_rejected(prepared):
    root, _, records = prepared
    ops.create_selection(root, "60001")
    cfg, record = records["sec4_constant"]
    record["history"]["val_acc"][-1] = 0.2
    save_tiny(root, cfg, record)
    with pytest.raises(ValueError, match="changed"):
        ops.verify_selection(root, "60001")


def complete_mixup(prepared, monkeypatch):
    root, _, records = prepared
    ops.create_selection(root, "60001")
    monkeypatch.setenv("AI6103_NO_FINAL_TEST", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "60002")
    monkeypatch.setattr(gpu_guard := ops.gpu_guard, "bootstrap", lambda: "tc2")
    monkeypatch.setattr(gpu_guard, "verify_cuda", lambda *_: ({"policy": "TEST_MOCK_ONLY"}, {}))
    monkeypatch.setattr(train, "environment", lambda *_: copy.deepcopy(records["sec4_constant"][1]["environment"]))
    ops.prepare_launch(root, "60001", "60002")
    cfg = ops.checked_plan(root)[0]
    record = fixture_record(cfg)
    record["environment"]["slurm_job_id"] = "60002"
    save_tiny(root, cfg, record)
    return root, cfg, record


def test_complete_verification_and_no_checkpoint_mutation(prepared, monkeypatch):
    root, cfg, _ = complete_mixup(prepared, monkeypatch)
    path = root / "results/sec6_mixup/checkpoint_0.pt"
    before = artifacts.file_hash(path)
    assert ops.verify(root, "60001", "60002")[3]["status"] == "SECTION6_COMPLETE_VALID"
    assert artifacts.file_hash(path) == before
    with pytest.raises(FileNotFoundError):
        ops.verify(root, "60001", "60003")
    path.write_bytes(b"CORRUPTED")
    with pytest.raises(ValueError, match="checksum"):
        ops.verify(root, "60001", "60002")


def test_comparison_uses_verified_baseline_and_distinguishes_metrics(prepared, monkeypatch):
    root, _, _ = complete_mixup(prepared, monkeypatch)
    output = root / "results/qa/section6/60004"
    result = ops.compare(root, "60001", "60002", output)
    report = artifacts.read_json(output / "comparison.json")
    assert result["status"] == "SECTION6_COMPARISON_COMPLETED"
    assert report["baseline"] == "sec5_wd1e-4" and report["test_evaluation"] == "NOT_RUN"
    assert "different targets" in report["metric_caveat"]
    assert len((output / "curves.csv").read_text().splitlines()) == 601
    assert len(list(output.glob("*.svg"))) == 5 and len(list(output.glob("*.png"))) == 5
    assert not (root / "data").exists()
    with pytest.raises(FileExistsError):
        ops.compare(root, "60001", "60002", output)


def test_beta_analytical_has_no_empirical_dependency(tmp_path):
    output = tmp_path / "results/qa/section6/analytical"
    ops.analytical(tmp_path, output)
    provenance = artifacts.read_json(output / "beta_02.provenance.json")
    assert provenance["alpha"] == 0.2 and provenance["empirical_data_used"] is False
    assert provenance["endpoint_density"] == "infinite"


def test_mixup_interpolation_permutation_and_rng_restore():
    x = torch.arange(128*6, dtype=torch.float32).reshape(128, 6)
    y = torch.arange(128)
    train.seed_all(6103)
    state = train.capture_rng()
    mixed, ya, yb, lam = train.mixup_data(x, y, 0.2)
    assert 0 <= lam <= 1 and torch.equal(ya, y) and sorted(yb.tolist()) == y.tolist()
    assert torch.allclose(mixed, lam*x + (1-lam)*x[yb])
    train.restore_rng(state)
    repeated = train.mixup_data(x, y, 0.2)
    assert repeated[3] == lam and torch.equal(repeated[0], mixed) and torch.equal(repeated[2], yb)


@pytest.mark.parametrize("alpha", [0, -0.2])
def test_invalid_mixup_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        train.mixup_data(torch.zeros(2, 3), torch.tensor([0, 1]), alpha)


def test_no_selection_or_training_after_final_stage(prepared):
    root, _, _ = prepared
    artifacts.write_json(root / "results/final_evaluation/test_started.json", {"status": "STARTED"})
    with pytest.raises(ValueError, match="final-test"):
        ops.checked_plan(root)


def test_slurm_jobs_single_experiment_and_no_final_test():
    for name in ("preflight", "mixup", "verify", "compare"):
        text = (config.ROOT / f"deployment/tc2/section6_{name}.sbatch").read_text()
        for directive in ("partition=MGPU-TC2", "qos=normal", "gres=gpu:1", "cpus-per-task=2", "mem=20G", "time=05:30:00"):
            assert "#SBATCH --" + directive in text
        assert text.index("conda activate") < text.index("set -u")
        assert "AI6103_NO_FINAL_TEST=1" in text and "evaluate-final" not in text
        assert "--resume" not in text and "--allow-cpu" not in text and "|| true" not in text
        if name == "mixup":
            assert "export AI6103_PROFILE=tc2" in text and "export CUDA_VISIBLE_DEVICES" not in text
            assert text.count("run_all.py train") == 1 and "train --section 6 --experiment sec6_mixup" in text
            assert text.index("launch-check") < text.index("run_all.py train")
        else:
            assert "run_all.py train" not in text and "CUDA_VISIBLE_DEVICES=-1" in text
