"""Hard-bounded by the parent process; two real train images, one optimizer step."""
import argparse
import tempfile
from pathlib import Path
import time
try:
    from .gpu_guard import bootstrap
except ImportError:
    from gpu_guard import bootstrap
bootstrap()
import torch
try:
    from .artifacts import read_json, write_json
    from .config import section3
    from .dataset import get_train_val_loaders
    from .train import build_model, build_optimizer, save_checkpoint, load_checkpoint, seed_all, evaluate_validation
except ImportError:
    from artifacts import read_json, write_json
    from config import section3
    from dataset import get_train_val_loaders
    from train import build_model, build_optimizer, save_checkpoint, load_checkpoint, seed_all, evaluate_validation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_all(6103)
    start = time.monotonic()
    stats = read_json(args.output_root / "normalization.json")
    tr, va, nc = get_train_val_loaders(stats["mean"], stats["std"], args.data_root, batch_size=2)
    cfg = section3({"dataset_fingerprint": "SMOKE_TEST_ONLY", "stats_fingerprint": "SMOKE_TEST_ONLY",
                    "implementation_fingerprint": "SMOKE_TEST_ONLY"})[0]
    model = build_model(nc)
    optimizer = build_optimizer(model, cfg)
    model.train()
    images, labels = next(iter(tr))
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    # Restrict validation to a single REAL two-image batch.
    val_images, val_labels = next(iter(va))
    small = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_images, val_labels), batch_size=2)
    val_loss, _ = evaluate_validation(model, small, torch.nn.CrossEntropyLoss(), "cpu")
    record = {"kind": "SMOKE_TEST_ONLY", "config": cfg.to_dict(), "fingerprint": cfg.fingerprint,
              "last_epoch": 0, "history": {}, "status": "SMOKE_TEST_ONLY"}
    with tempfile.TemporaryDirectory(dir=args.output_root) as tmp:
        cp = Path(tmp) / "smoke.pt"
        save_checkpoint(cp, model, optimizer, None, record, (tr, va))
        # Deliberately perturb a parameter, then restore the full state.
        parameter = next(model.parameters())
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1)
        load_checkpoint(cp, cfg, model, optimizer, None, (tr, va))
        assert torch.equal(parameter, original)
    result = {"kind": "SMOKE_TEST_ONLY", "status": "PASS", "device": "cpu",
              "train_examples": 2, "validation_examples": 2, "optimizer_steps": 1,
              "finite_losses": bool(torch.isfinite(loss)) and bool(torch.isfinite(torch.tensor(val_loss))),
              "checkpoint_round_trip": True, "seconds": time.monotonic() - start}
    assert result["finite_losses"]
    write_json(args.output_root / "qa" / "smoke_test.json", result)
    print(result)


if __name__ == "__main__":
    main()
