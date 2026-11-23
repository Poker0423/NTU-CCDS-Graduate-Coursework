import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import fixture_record
from scripts import artifacts, config, protocol, run_all, train


def cuda_records(configs):
    records = [fixture_record(cfg, loss=loss) for cfg, loss in zip(configs,
        (4.536550395693669, 4.080613163148999, 4.790708950797134))]
    for index, record in enumerate(records):
        record["environment"].update({"device": "cuda:0", "gpu_name": "NVIDIA A40",
            "gpu_memory_bytes": 47709159424, "cuda_version": "12.4", "nvidia_driver": "570.148.08",
            "slurm_job_id": ("42226", "42220", "42226")[index],
            "slurm_timelimit": ("01:00:00", "05:30:00", "01:00:00")[index],
            "gpu_free_mib": (45231, 45231, 39481)[index], "gpu_utilization_percent": index})
    return records


def test_real_loss_values_and_legitimate_instance_differences_select_lr_005(configs):
    records = cuda_records(configs)
    before = copy.deepcopy(records)
    assert protocol.select_section3(records, configs).lr == 0.05
    assert records == before
    records[0]["history"]["val_loss"] = [0.0] * 15
    records[1]["history"]["val_loss"] = [999.0] * 15
    assert protocol.select_section3(records, configs).lr == 0.05


@pytest.mark.parametrize("key,value", [
    ("torch", "2.7.0+cu126"), ("torchvision", "0.22.0"), ("python", "3.11.0"),
    ("numpy", "incompatible"), ("pillow", "incompatible"), ("cuda_version", "12.6"),
    ("nvidia_driver", "different"), ("gpu_name", "NVIDIA L40S"),
    ("gpu_memory_bytes", 20000000000), ("cpu_threads", 8), ("device", "cpu"),
    ("amp", True), ("parameter_dtype", "float16"), ("allow_tf32", True),
    ("float32_matmul_precision", "medium"), ("unknown_numerical_setting", "changed"),
])
def test_incompatible_or_new_numerical_settings_are_not_silently_ignored(configs, key, value):
    records = cuda_records(configs)
    records[1]["environment"][key] = value
    with pytest.raises(ValueError):
        protocol.select_section3(records, configs)


def test_shared_dataset_and_precision_config_must_match(context):
    configs = config.section3(context)
    configs[1] = replace(configs[1], dataset_fingerprint="OTHER_DATA")
    records = cuda_records(configs)
    with pytest.raises(ValueError, match="provenance"):
        protocol.select_section3(records, configs)


def test_missing_provenance_and_inconsistent_aliases_fail(configs):
    records = cuda_records(configs)
    records[1]["environment"].pop("torch")
    with pytest.raises(ValueError, match="Missing"):
        protocol.select_section3(records, configs)
    records = cuda_records(configs)
    records[1]["environment"]["runtime_gpu_name"] = "NVIDIA L40S"
    with pytest.raises(ValueError, match="aliases"):
        protocol.select_section3(records, configs)


def test_audited_analysis_patch_retains_training_lineage_without_faking_raw_hash():
    assert protocol.training_implementation_fingerprint() == "e4b846f4cc8a45fadc162e686cd78330a7758880761af4e53e8138b90491c8b1"
    assert artifacts.implementation_hash() != protocol.training_implementation_fingerprint()


def test_unapproved_code_drift_is_not_accepted_by_lineage(monkeypatch):
    original = protocol.file_hash
    monkeypatch.setattr(protocol, "file_hash", lambda path: "drift" if Path(path).name == "train.py" else original(path))
    with pytest.raises(ValueError, match="audited"):
        protocol.training_implementation_fingerprint()


def test_selection_cli_reads_verified_checkpoints_and_does_not_change_inputs(tmp_path, monkeypatch, context):
    configs = config.section3(context)
    record_paths = []
    for cfg, loss in zip(configs, (4.536550395693669, 4.080613163148999, 4.790708950797134)):
        directory = tmp_path / cfg.experiment_id
        directory.mkdir()
        record = fixture_record(cfg, loss=loss)
        record["environment"].update(slurm_job_id="different_" + cfg.experiment_id, gpu_free_mib=loss * 1000)
        record["checkpoint"] = "checkpoint_1.pt"
        model = torch.nn.Linear(1, 1)
        optimizer = train.build_optimizer(model, cfg)
        train.save_checkpoint(directory / record["checkpoint"], model, optimizer, None, record)
        record["checkpoint_sha256"] = artifacts.file_hash(directory / record["checkpoint"])
        artifacts.write_json(directory / "record.json", record)
        record_paths.extend((directory / "record.json", directory / record["checkpoint"]))
    before = {path: artifacts.file_hash(path) for path in record_paths}
    monkeypatch.setattr(run_all, "load_context", lambda *_: (context, {}, {}))
    assert run_all.main(["--output-root", str(tmp_path), "select", "--section", "3"]) == 0
    selection = artifacts.read_json(tmp_path / "section3_selection.json")
    assert selection["lr"] == 0.05 and selection["rule"] == "Final training loss ONLY"
    assert selection["analysis_implementation_fingerprint"] == artifacts.implementation_hash()
    assert {path: artifacts.file_hash(path) for path in record_paths} == before
    checkpoint = record_paths[1]
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        run_all.main(["--output-root", str(tmp_path), "select", "--section", "3"])


def test_selection_sbatch_has_only_selection_command_and_no_job_chaining():
    source = (config.ROOT / "deployment/tc2/select_section3.sbatch").read_text()
    commands = [line for line in source.splitlines() if line.startswith("python ")]
    assert commands == ["python -B -u scripts/run_all.py select --section 3"]
    assert "export AI6103_PROFILE=cpu" in source and "export CUDA_VISIBLE_DEVICES=-1" in source
    assert "--partition=MGPU-TC2" in source and "--qos=normal" in source
    assert "sha256sum --check" in source and "SECTION3_SELECTION_VERIFIED" in source
    assert "afterok" not in source and "evaluate-final" not in source
