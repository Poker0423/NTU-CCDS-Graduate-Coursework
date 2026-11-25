import ast
import csv
from pathlib import Path
import re
import nbformat
import numpy as np
from PIL import Image
import pytest
import torch
from scripts import dataset, fill_latex, protocol, train, config
from scripts.artifacts import write_json, read_json, file_hash
from conftest import fixture_record

ROOT = Path(__file__).resolve().parents[1]


def make_data(root):
    counter = 0
    for split in ("train", "val", "test"):
        directory = root / split / "images"
        directory.mkdir(parents=True)
        rows = []
        for label in ("zebra_dog", "alpha_dog"):
            identifier = str(counter)
            Image.new("RGB", (12, 14), (counter * 30, 100, 120)).save(directory / f"{identifier}.jpg")
            rows.append({"id": identifier, "breed": label})
            counter += 1
        with (root / f"{split}_labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("id", "breed"))
            writer.writeheader(); writer.writerows(rows)


def test_notebook_valid_and_safe_defaults():
    notebook = nbformat.read(ROOT / "notebooks" / "AI6103_HW_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
    source = "\n".join(c.source for c in notebook.cells)
    assert "RUN_SECTION = None" in source and "AUTHORIZE_FINAL_TEST = False" in source
    assert "def train_one_epoch" not in source and "class DogBreedDataset" not in source


def test_data_class_mapping_transform_and_no_test_loader(tmp_path):
    make_data(tmp_path)
    info = dataset.inspect_dataset(tmp_path, {"train": 2, "val": 2, "test": 2}, 2)
    assert info["class_mapping"] == {"alpha_dog": 0, "zebra_dog": 1}
    assert info["content_hashes_checked"] == 6 and not info["duplicate_contents"]
    tr, va, nc = dataset.get_train_val_loaders([0.5]*3, [0.2]*3, tmp_path, 2)
    assert nc == 2 and tr.dataset.split == "train" and va.dataset.split == "val"
    assert tr.dataset.labels == [1, 0] == va.dataset.labels
    assert next(iter(tr))[0].shape == (2, 3, 100, 100)
    tf, ev = dataset.make_transforms([0.5]*3, [0.2]*3)
    assert tf.transforms[0].size == (100, 100)
    assert tf.transforms[1].p == 0.5
    crop = tf.transforms[2]
    assert crop.size == (100, 100) and crop.padding == 12 and crop.padding_mode == "constant" and crop.fill == 0
    assert len(ev.transforms) == 3 and not any(isinstance(t, dataset.transforms.RandomCrop) for t in ev.transforms)


@pytest.mark.parametrize("issue", ("missing", "unknown_label", "duplicate_id", "extra_image"))
def test_dataset_corruption_rejected(tmp_path, issue):
    make_data(tmp_path)
    if issue == "missing":
        (tmp_path / "val/images/2.jpg").unlink()
    elif issue == "unknown_label":
        p = tmp_path / "val_labels.csv"
        p.write_text(p.read_text().replace("alpha_dog", "unknown"))
    elif issue == "duplicate_id":
        p = tmp_path / "val_labels.csv"
        p.write_text(p.read_text().replace("3,", "2,"))
    else:
        Image.new("RGB", (2, 2)).save(tmp_path / "val/images/extra.jpg")
    with pytest.raises(ValueError):
        dataset.inspect_dataset(tmp_path, {"train": 2, "val": 2, "test": 2}, 2)


def test_supplied_data_evidence():
    info = read_json(ROOT / "results/dataset_integrity.json")
    assert {s: v["count"] for s, v in info["splits"].items()} == config.EXPECTED_SPLITS
    assert len(info["class_mapping"]) == 120 and info["no_duplicate_ids"]
    assert info["cross_split_duplicate_contents"] == 10
    archive = read_json(ROOT / "results/archive_integrity.json")
    assert archive["checked_files"] == 10225 and not archive["missing"] and not archive["mismatches"]
    stats = read_json(ROOT / "results/normalization.json")
    historical = read_json(ROOT / "results/mean_std.json")
    assert stats["mean"] == historical["mean"] and stats["std"] == historical["std"]
    assert stats["dataset_fingerprint"] == info["dataset_fingerprint"]
    assert np.max(np.abs(np.array(stats["recomputed_mean"] + stats["recomputed_std"]) - np.array(stats["mean"] + stats["std"]))) < 1e-6


def test_stats_training_only(tmp_path, monkeypatch):
    make_data(tmp_path)
    real = dataset.DogBreedDataset
    seen = []
    def spy(split, *args, **kwargs):
        seen.append(split)
        return real(split, *args, **kwargs)
    monkeypatch.setattr(dataset, "DogBreedDataset", spy)
    mean, std = dataset.compute_mean_std(tmp_path)
    assert seen == ["train"] and len(mean) == len(std) == 3


def test_report_honesty_and_bindings():
    source = (ROOT / "report/main.tex").read_text(encoding="utf-8")
    assert "0.005" not in source and "0.5,0.05,0.001" in source
    assert "training loss" in source
    assert "Takeaway" not in source and not re.search(r"\\section\{.*7", source)
    for n in ("Three", "Four", "Five", "Six"):
        assert "\\Section" + n + "Rows" in source
        assert "\\Discussion" + n in source
    for claim in ("exhibits the lowest training loss", "Cosine annealing achieves lower", "accuracy improvements confirm", "necessarily diverges"):
        assert claim not in source
    bindings = (ROOT / "report/bindings.tex").read_text()
    assert "PRE-EXPERIMENT DRAFT" in bindings and "Pending" in bindings
    assert "Experimental figure pending" in bindings
    # Balanced braces after removing escaped braces and comments.
    for text in (source, bindings):
        stripped = re.sub(r"(?<!\\)%[^\n]*", "", text)
        stripped = re.sub(r"\\[{}]", "", stripped)
        level = 0
        for char in stripped:
            level += (char == "{") - (char == "}")
            assert level >= 0
        assert level == 0
    refs = (ROOT / "report/refs.bib").read_text()
    for citation in re.findall(r"\\cite(?:t|p)?\{([^}]+)\}", source):
        assert citation in refs


def test_final_report_refuses_incomplete_runs(tmp_path, context, monkeypatch):
    monkeypatch.setattr(fill_latex, "load_context", lambda *_: (context, {"mean": [0]*3, "std": [1]*3}, {}))
    with pytest.raises(FileNotFoundError):
        fill_latex.generate("final", tmp_path, tmp_path, ROOT / "figures", tmp_path)
    assert not (tmp_path / "main_filled.tex").exists()


def test_beta_analytical_pdf():
    from scipy.stats import beta
    from scipy.integrate import quad
    assert beta.pdf(0.1, 0.2, 0.2) == pytest.approx(beta.pdf(0.9, 0.2, 0.2))
    assert np.isinf(beta.pdf(0, 0.2, 0.2)) and np.isinf(beta.pdf(1, 0.2, 0.2))
    integral, _ = quad(lambda x: beta.pdf(x, 0.2, 0.2), 0, 1)
    assert integral == pytest.approx(1, abs=1e-6)
    assert file_hash(ROOT / "figures/beta_pdf.png") == read_json(ROOT / "figures/beta_pdf.provenance.json")["sha256"]


def test_plots_readability_only_in_temporary_tests(configs, tmp_path):
    from scripts.plots import plot_curves
    path = tmp_path / "TEST_ONLY_SYNTHETIC.png"
    plot_curves([fixture_record(c) for c in configs], ["A", "B", "C"], "TEST ONLY", path)
    image = Image.open(path)
    assert image.width >= 900 and image.height >= 1400


def test_final_evaluation_claim_precedes_test_loader(context, tmp_path, monkeypatch):
    configs = config.section3(context) + config.section4(context, 0.05) + config.section5(context, 0.05)
    baseline = configs[4]
    configs += [config.section6(context, baseline)]
    records = [fixture_record(c) for c in configs]
    for c, r in zip(configs, records):
        r["checkpoint"] = "checkpoint_0.pt"
        r["checkpoint_sha256"] = "TEST_ONLY"
        write_json(tmp_path / c.experiment_id / "record.json", r)
    monkeypatch.setattr(protocol, "completed_bundle", lambda *_: (configs, records, baseline))
    monkeypatch.setattr(protocol, "load_context", lambda *_: (context, {"mean": [0]*3, "std": [1]*3}, {}))
    monkeypatch.setattr(protocol, "build_model", lambda *_: torch.nn.Linear(1, 1))
    monkeypatch.setattr(protocol, "load_checkpoint", lambda *_args, **_kwargs: None)
    def spy(*_):
        assert (tmp_path / "final_evaluation/test_started.json").exists()
        class SyntheticTestOnly:
            dataset = range(1534)
        return SyntheticTestOnly()
    monkeypatch.setattr(protocol, "get_final_test_loader", spy)
    monkeypatch.setattr(protocol, "_evaluate", lambda *_: (2.0, 0.1))
    protocol.freeze(context, tmp_path)
    result = protocol.evaluate_final_test(tmp_path, tmp_path, torch.device("cpu"), authorized=True)
    assert result["status"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="already started"):
        protocol.evaluate_final_test(tmp_path, tmp_path, torch.device("cpu"), authorized=True)
    with pytest.raises(FileExistsError):
        protocol.freeze(context, tmp_path)
