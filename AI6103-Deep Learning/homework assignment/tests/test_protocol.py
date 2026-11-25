import copy
from dataclasses import replace
import pytest
from scripts import config, protocol
from conftest import fixture_record


def test_official_registry(context):
    assert config.SECTION3_LRS == [0.5, 0.05, 0.001]
    c3 = config.section3(context)
    assert all(c.batch_size == 128 and c.epochs == 15 and c.weight_decay == 0 and c.scheduler == "constant" for c in c3)
    c4 = config.section4(context, 0.05)
    c5 = config.section5(context, 0.05)
    assert [c.scheduler for c in c4] == ["constant", "cosine"]
    assert [c.weight_decay for c in c5] == [5e-4, 1e-4]
    for baseline in c4 + c5:
        mixup = config.section6(context, baseline)
        assert mixup.mixup and mixup.mixup_alpha == 0.2
        assert (mixup.lr, mixup.scheduler, mixup.weight_decay) == (baseline.lr, baseline.scheduler, baseline.weight_decay)
        assert mixup.epochs == 300


def test_lr_only_training_loss(configs):
    records = [fixture_record(c, loss=l, val=v) for c, l, v in zip(configs, (3, 1, 2), (0.1, 8, 0.2))]
    records[0]["history"]["train_acc"] = [0.99] * 15
    records[0]["history"]["val_acc"] = [0.99] * 15
    assert protocol.select_section3(records, configs).lr == 0.05
    for record in records:
        record["history"]["val_acc"] = [0] * 15
        record["history"]["val_loss"] = [999] * 15
    assert protocol.select_section3(records, configs).lr == 0.05


def test_deterministic_ties(configs):
    assert protocol.select_section3([fixture_record(c) for c in configs], configs).lr == 0.5


@pytest.mark.parametrize("mutation", ["partial", "wrong_lr", "legacy", "nan", "short_history", "wrong_fingerprint", "test_metric", "wrong_scheduler", "environment"])
def test_invalid_selection_rejected(configs, mutation):
    records = [fixture_record(c) for c in configs]
    r = records[1]
    if mutation == "partial":
        r["status"] = "PARTIAL"
    elif mutation == "wrong_lr":
        r["config"]["lr"] = 0.005
    elif mutation == "legacy":
        r.pop("kind")
    elif mutation == "nan":
        r["history"]["train_loss"][-1] = float("nan")
    elif mutation == "short_history":
        r["history"]["train_loss"].pop()
    elif mutation == "wrong_fingerprint":
        r["fingerprint"] = "bad"
    elif mutation == "test_metric":
        r["history"]["test_acc"] = [0.1]
    elif mutation == "wrong_scheduler":
        r["history"]["lr"][-1] = 0.0
    elif mutation == "environment":
        r["environment"]["torch"] = "different"
    with pytest.raises(ValueError):
        protocol.select_section3(records, configs)


def test_missing_and_old_candidates(configs):
    with pytest.raises(ValueError):
        protocol.select_section3([fixture_record(c) for c in configs[:2]], configs[:2])
    with pytest.raises(ValueError):
        replace(configs[1], lr=0.005).validate()


def test_false_complete_status_rejected_even_in_partial_validation(configs):
    record = fixture_record(configs[0], last_epoch=2)
    record["status"] = "COMPLETE"
    record["ended_at"] = "TEST_ONLY"
    with pytest.raises(ValueError, match="Completion status"):
        protocol.validate_record(record, configs[0], require_complete=False)


def test_downstream_validation_uses_all_conditions(context):
    configs = config.section4(context, 0.05) + config.section5(context, 0.05)
    records = [fixture_record(c, val=v) for c, v in zip(configs, (1, 2, 3, 4))]
    assert protocol.select_validation(records, configs).experiment_id == "sec4_constant"
    mixup = config.section6(context, protocol.select_validation(records, configs))
    assert mixup.scheduler == "constant" and mixup.weight_decay == 0


def test_required_dependencies_not_skipped(context, tmp_path):
    with pytest.raises(FileNotFoundError):
        protocol.plan(4, context, tmp_path)
    with pytest.raises(FileNotFoundError):
        protocol.freeze(context, tmp_path)
    assert not (tmp_path / "frozen_manifest.json").exists()
