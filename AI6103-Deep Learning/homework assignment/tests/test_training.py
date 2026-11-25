import copy
import inspect
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from scripts import train, protocol, config
from scripts.artifacts import read_json, write_json, file_hash
from conftest import fixture_record


def loader(split="train"):
    ds = torch.utils.data.TensorDataset(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), torch.tensor([0, 1, 0]))
    ds.split = split
    return torch.utils.data.DataLoader(ds, batch_size=2, generator=torch.Generator().manual_seed(5))


def test_real_model_and_optimizer(configs, monkeypatch):
    constructor = train.efficientnet_v2_s
    seen = []
    def spy(*args, **kwargs):
        seen.append(kwargs)
        return constructor(*args, **kwargs)
    monkeypatch.setattr(train, "efficientnet_v2_s", spy)
    model = train.build_model(120)
    assert seen == [{"weights": None}]
    assert model.classifier[1].out_features == 120
    assert model.__class__.__name__ == "EfficientNet"
    optimizer = train.build_optimizer(model, configs[0])
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9


def test_sample_weighted_evaluation_and_training():
    logits = torch.tensor([[4., 0.], [0., 4.], [0., 4.]])
    labels = torch.tensor([0, 1, 0])
    ds = torch.utils.data.TensorDataset(logits, labels)
    batches = torch.utils.data.DataLoader(ds, batch_size=2)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    expected = torch.nn.functional.cross_entropy(logits, labels).item()
    loss, acc = train.evaluate_validation(model, batches, torch.nn.CrossEntropyLoss(), "cpu")
    assert loss == pytest.approx(expected, abs=1e-6) and acc == pytest.approx(2 / 3)
    loss, acc = train.train_one_epoch(model, batches, torch.optim.SGD(model.parameters(), lr=0), torch.nn.CrossEntropyLoss(), "cpu")
    assert loss == pytest.approx(expected, abs=1e-6) and acc == pytest.approx(2 / 3)


def test_mixup_weighted_objective(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    x, y = torch.tensor([[4., 0.], [0., 4.]]), torch.tensor([0, 1])
    ya, yb, lam = y, y.flip(0), 0.75
    actual_mixup = train.mixup_data
    monkeypatch.setattr(train, "mixup_data", lambda *_: (x, ya, yb, lam))
    ds = torch.utils.data.TensorDataset(x, y)
    result = train.train_one_epoch(model, torch.utils.data.DataLoader(ds, batch_size=2),
        torch.optim.SGD(model.parameters(), lr=0), torch.nn.CrossEntropyLoss(), "cpu", True)
    expected = lam * torch.nn.functional.cross_entropy(x, ya) + (1 - lam) * torch.nn.functional.cross_entropy(x, yb)
    assert result[0] == pytest.approx(expected.item()) and result[1] == 0.75
    with pytest.raises(ValueError):
        actual_mixup(x, y, 0)


def test_cosine_300_epoch_timing(context):
    cfg = config.section4(context, 0.05)[1]
    optimizer = train.build_optimizer(torch.nn.Linear(1, 1), cfg)
    scheduler = train.build_scheduler(optimizer, cfg)
    used = []
    for i in range(300):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used[0] == cfg.lr and used[150] == pytest.approx(cfg.lr / 2)
    assert used[-1] > 0 and optimizer.param_groups[0]["lr"] == 0
    assert scheduler.last_epoch == 300


def test_checkpoint_full_roundtrip(configs, tmp_path):
    cfg = configs[0]
    train.seed_all(10)
    tr, va = loader(), loader("val")
    model = torch.nn.Linear(2, 120)
    optimizer = train.build_optimizer(model, cfg)
    train.train_one_epoch(model, tr, optimizer, torch.nn.CrossEntropyLoss(), "cpu")
    record = fixture_record(cfg, last_epoch=1)
    path = tmp_path / "cp.pt"
    train.save_checkpoint(path, model, optimizer, None, record, (tr, va))
    expected_rng = (random.random(), np.random.rand(), torch.rand(1), next(iter(tr))[0])
    expected_weights = copy.deepcopy(model.state_dict())
    restored = torch.nn.Linear(2, 120)
    new_optimizer = train.build_optimizer(restored, cfg)
    state = train.load_checkpoint(path, cfg, restored, new_optimizer, None, (tr, va))
    assert state["record"]["last_epoch"] == 1
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected_weights.items())
    assert new_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    for old, new in zip(optimizer.state.values(), new_optimizer.state.values()):
        assert torch.equal(old["momentum_buffer"], new["momentum_buffer"])
    actual = (random.random(), np.random.rand(), torch.rand(1), next(iter(tr))[0])
    assert expected_rng[:2] == actual[:2]
    assert torch.equal(expected_rng[2], actual[2]) and torch.equal(expected_rng[3], actual[3])
    from dataclasses import replace
    with pytest.raises(ValueError):
        train.load_checkpoint(path, replace(cfg, seed=42), restore_random=False)


def test_checkpoint_scheduler_roundtrip(context, tmp_path):
    cfg = config.section4(context, 0.05)[1]
    model = torch.nn.Linear(1, 1)
    optimizer = train.build_optimizer(model, cfg)
    scheduler = train.build_scheduler(optimizer, cfg)
    optimizer.step(); scheduler.step()
    train.save_checkpoint(tmp_path / "cp.pt", model, optimizer, scheduler, fixture_record(cfg, last_epoch=1))
    new_optimizer = train.build_optimizer(model, cfg)
    new_scheduler = train.build_scheduler(new_optimizer, cfg)
    train.load_checkpoint(tmp_path / "cp.pt", cfg, model, new_optimizer, new_scheduler)
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("section", (3, 4, 5, 6))
def test_all_training_sections_never_use_test(section, context, monkeypatch, tmp_path):
    cfg = (config.section3(context) if section == 3 else config.section4(context, 0.05) if section == 4 else config.section5(context, 0.05) if section == 5 else [config.section6(context, config.section4(context, 0.05)[1])])[0]
    monkeypatch.setattr(train, "build_model", lambda *_: torch.nn.Linear(2, 120))
    def simulated_epoch(model, loader, optimizer, *args):
        optimizer.step()  # no gradients: only exercise scheduler's step-order bookkeeping
        return 2., 0.1
    monkeypatch.setattr(train, "train_one_epoch", simulated_epoch)
    monkeypatch.setattr(train, "evaluate_validation", lambda *_: (3., 0.1))
    # Stop after one committed simulated epoch; no real epoch/official training is run.
    original = train.save_checkpoint
    def stop(path, model, optimizer, scheduler, record, loaders):
        if record["last_epoch"] > 1:
            raise RuntimeError("TEST_ONLY_INTERRUPT")
        original(path, model, optimizer, scheduler, record, loaders)
    monkeypatch.setattr(train, "save_checkpoint", stop)
    monkeypatch.setattr(protocol, "evaluate_final_test", lambda *_: pytest.fail("test evaluation reached"))
    monkeypatch.setattr(protocol, "get_final_test_loader", lambda *_: pytest.fail("test loader reached"))
    with pytest.raises(RuntimeError, match="TEST_ONLY_INTERRUPT"):
        train.train_experiment(cfg, loader(), loader("val"), tmp_path, torch.device("cpu"))
    record = read_json(tmp_path / cfg.experiment_id / "record.json")
    assert record["status"] == "PARTIAL" and record["last_epoch"] == 1
    assert all("test" not in key for key in record["history"])
    assert "test_loader" not in inspect.signature(train.train_experiment).parameters


def test_resume_simulated_epochs_reproduce_rng(configs, tmp_path, monkeypatch):
    cfg = configs[0]
    monkeypatch.setattr(train, "build_model", lambda *_: torch.nn.Linear(2, 120))
    def pseudo_epoch(model, loader, optimizer, *args):
        # TEST-ONLY RNG fixture, no real model training and no optimizer step.
        return float(random.random() + np.random.rand() + torch.rand(1).item()), 0.1
    monkeypatch.setattr(train, "train_one_epoch", pseudo_epoch)
    monkeypatch.setattr(train, "evaluate_validation", lambda *_: (3., 0.1))
    full = train.train_experiment(cfg, loader(), loader("val"), tmp_path / "full", torch.device("cpu"))
    save = train.save_checkpoint
    def interrupt(path, model, optimizer, scheduler, record, loaders):
        if record["last_epoch"] == 3:
            raise KeyboardInterrupt()
        save(path, model, optimizer, scheduler, record, loaders)
    monkeypatch.setattr(train, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train.train_experiment(cfg, loader(), loader("val"), tmp_path / "interrupted", torch.device("cpu"))
    partial = read_json(tmp_path / "interrupted" / cfg.experiment_id / "record.json")
    assert partial["last_epoch"] == 2 and partial["status"] == "PARTIAL"
    monkeypatch.setattr(train, "save_checkpoint", save)
    resumed = train.train_experiment(cfg, loader(), loader("val"), tmp_path / "interrupted", torch.device("cpu"), resume=True)
    for metric in train.METRICS:
        assert full["history"][metric] == resumed["history"][metric]
    with pytest.raises(FileExistsError):
        train.train_experiment(cfg, loader(), loader("val"), tmp_path / "full", torch.device("cpu"))
    verified = protocol.read_verified(tmp_path / "full", cfg)
    assert verified["last_epoch"] == 15


def test_lock_and_test_guards(configs, tmp_path):
    with train.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with train.run_lock(tmp_path):
                pass
    assert not (tmp_path / ".run.lock").exists()
    with pytest.raises(ValueError):
        train.train_experiment(configs[0], loader(), loader("test"), tmp_path, torch.device("cpu"))
    with pytest.raises(ValueError):
        train.evaluate_validation(torch.nn.Linear(2, 120), loader("test"), torch.nn.CrossEntropyLoss(), "cpu")
    with pytest.raises(PermissionError):
        protocol.evaluate_final_test(tmp_path, tmp_path, torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        protocol.evaluate_final_test(tmp_path, tmp_path, torch.device("cpu"), authorized=True)
    assert not (tmp_path / "final_evaluation" / "test_started.json").exists()


def test_no_gpu_safe_stop(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU_NOT_AVAILABLE"):
        train.device_for_training()
    assert train.device_for_training(True).type == "cpu"
