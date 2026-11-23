"""Failure-path checks use small TEST_ONLY checkpoints, never real experiment artifacts."""
from dataclasses import replace
import pytest
import torch
from scripts import train, protocol, config
from scripts.artifacts import file_hash, read_json, write_json
from conftest import fixture_record


def make_saved(tmp_path, cfg):
    directory = tmp_path / cfg.experiment_id
    record = fixture_record(cfg)
    record["checkpoint"] = "checkpoint_1.pt"
    cp = directory / record["checkpoint"]
    model = torch.nn.Linear(1, 1)
    optimizer = train.build_optimizer(model, cfg)
    train.save_checkpoint(cp, model, optimizer, None, record)
    record["checkpoint_sha256"] = file_hash(cp)
    write_json(directory / "record.json", record)
    return directory, record


def test_checksum_corruption_rejected(configs, tmp_path):
    directory, record = make_saved(tmp_path, configs[0])
    protocol.read_verified(tmp_path, configs[0])
    with (directory / record["checkpoint"]).open("ab") as f:
        f.write(b"TEST_ONLY_CORRUPTION")
    with pytest.raises(ValueError, match="checksum"):
        protocol.read_verified(tmp_path, configs[0])


def test_optimizer_config_tampering_rejected(configs, tmp_path):
    directory, record = make_saved(tmp_path, configs[0])
    cp = directory / record["checkpoint"]
    state = torch.load(cp, weights_only=False)
    state["optimizer"]["param_groups"][0]["lr"] = 0.01
    torch.save(state, cp)
    record["checkpoint_sha256"] = file_hash(cp)
    write_json(directory / "record.json", record)
    with pytest.raises(ValueError, match="Optimizer checkpoint state"):
        protocol.read_verified(tmp_path, configs[0])


def test_record_and_checkpoint_mismatch_rejected(configs, tmp_path):
    directory, record = make_saved(tmp_path, configs[0])
    record["history"]["train_loss"][-1] = 0.01
    write_json(directory / "record.json", record)
    with pytest.raises(ValueError, match="Checkpoint/record mismatch"):
        protocol.read_verified(tmp_path, configs[0])


def test_incompatible_resume_preserves_existing_artifacts(configs, tmp_path):
    cfg = configs[0]
    directory, record = make_saved(tmp_path, cfg)
    before = file_hash(directory / "record.json")
    ds = torch.utils.data.TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    with pytest.raises(ValueError, match="Incompatible existing record"):
        train.train_experiment(replace(cfg, seed=7), dl, dl, tmp_path, torch.device("cpu"), resume=True)
    assert file_hash(directory / "record.json") == before
    assert file_hash(directory / record["checkpoint"]) == record["checkpoint_sha256"]


def test_frozen_training_refused_before_initialization(configs, tmp_path, monkeypatch):
    write_json(tmp_path / "frozen_manifest.json", {"status": "FROZEN"})
    monkeypatch.setattr(train, "build_model", lambda *_: pytest.fail("model initialized after freeze"))
    ds = torch.utils.data.TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    with pytest.raises(ValueError, match="frozen"):
        train.train_experiment(configs[0], dl, dl, tmp_path, torch.device("cpu"))
